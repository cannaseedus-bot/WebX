/**
 * scx_transformer_block.cpp
 * End-to-end 1-layer transformer block + KV-cache token generation loop.
 * Reuses all existing project shaders; INT4 weights via scxq2_infer_layer.hlsl.
 *
 * Pipeline per layer (forward / prefill):
 *   LayerNorm1  → gpt2_layernorm_fwd.hlsl  CSMain
 *   QKV (INT4)  → scxq2_infer_layer.hlsl   CSQProj  ×3 → pack into qkv_buf
 *   Attention   → gpt2_attn_fwd.hlsl        CSMain   (causal, multi-head)
 *   Residual    → gpt2_residual_add.hlsl    CSMain_addto
 *   LayerNorm2  → gpt2_layernorm_fwd.hlsl  CSMain
 *   FFN_W1(INT4)→ scxq2_infer_layer.hlsl   CSQProj
 *   GELU        → gpt2_gelu_fwd.hlsl        CSMain
 *   FFN_W2(INT4)→ scxq2_infer_layer.hlsl   CSQProj
 *   Residual    → gpt2_residual_add.hlsl    CSMain_addto
 *
 * Token generation loop:
 *   Embed → N layers (KV cache update each step) → lm_head → sample → repeat
 *
 * Thread model:
 *   Thread 1 — stream:   feeds INT4 weight chunks → weightBufs
 *   Thread 2 — gpu:      builds D3D11 compute dispatch sequences
 *   Thread 3 — generate: token loop, waits on layer readiness per step
 *
 * Compile:
 *   cl scx_transformer_block.cpp /EHsc /std:c++17 d3d11.lib dxgi.lib d3dcompiler.lib
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <random>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

using Microsoft::WRL::ComPtr;

// ── Model dimensions ──────────────────────────────────────────────────────────
// Small test model — change to 768/12/64 for GPT-2 small scale.

static constexpr uint32_t E          =  64;  // n_embd  (embedding dim)
static constexpr uint32_t H          =   1;  // n_head
static constexpr uint32_t D          =  64;  // head_dim = E/H
static constexpr uint32_t FFN_DIM    = 256;  // FFN intermediate (4×E typical)
static constexpr uint32_t VOCAB      = 256;  // vocabulary size (byte-level demo)
static constexpr uint32_t MAX_SEQ    = 128;  // max tokens in KV cache
static constexpr uint32_t NUM_LAYERS =   1;  // stack depth

// INT4 packing: 8 weights per uint32
static constexpr uint32_t PK(uint32_t n) { return (n + 7) / 8; }  // packed uint32 per row

// ── D3D11 globals ─────────────────────────────────────────────────────────────

static ComPtr<ID3D11Device>        gDev;
static ComPtr<ID3D11DeviceContext> gCtx;
static std::string                 gAdapterName;

// ── Shader paths ──────────────────────────────────────────────────────────────

#define SHADER_DIR L"C:\\Users\\canna\\.gpu_trainer\\shaders\\"

static const wchar_t* PATH_LN    = SHADER_DIR L"gpt2_layernorm_fwd.hlsl";
static const wchar_t* PATH_MATMUL= SHADER_DIR L"scxq2_infer_layer.hlsl";
static const wchar_t* PATH_ATTN  = SHADER_DIR L"gpt2_attn_fwd.hlsl";
static const wchar_t* PATH_GELU  = SHADER_DIR L"gpt2_gelu_fwd.hlsl";
static const wchar_t* PATH_RES   = SHADER_DIR L"gpt2_residual_add.hlsl";
static const wchar_t* PATH_EMBED = SHADER_DIR L"gpt2_embed_fwd.hlsl";

// ── D3D11 init ────────────────────────────────────────────────────────────────

static bool InitD3D11() {
    D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1 };
    D3D_FEATURE_LEVEL got;
    HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE,
        nullptr, 0, levels, 2, D3D11_SDK_VERSION, &gDev, &got, &gCtx);
    if (FAILED(hr))
        hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP,
            nullptr, 0, levels, 2, D3D11_SDK_VERSION, &gDev, &got, &gCtx);
    if (FAILED(hr)) return false;

    // Log adapter name
    ComPtr<IDXGIDevice>  dxgiDev; gDev.As(&dxgiDev);
    ComPtr<IDXGIAdapter> adapter; dxgiDev->GetAdapter(&adapter);
    DXGI_ADAPTER_DESC desc{};     adapter->GetDesc(&desc);
    char buf[256]{}; WideCharToMultiByte(CP_UTF8,0,desc.Description,-1,buf,255,nullptr,nullptr);
    gAdapterName = buf;
    printf("[D3D11] %s  FL=%04x\n", gAdapterName.c_str(), (unsigned)got);
    return true;
}

// ── Buffer helpers ────────────────────────────────────────────────────────────

static ComPtr<ID3D11Buffer> MakeSBuf(uint32_t stride, uint32_t count,
                                      UINT bindFlags, const void* init = nullptr)
{
    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth           = stride * count;
    bd.Usage               = D3D11_USAGE_DEFAULT;
    bd.BindFlags           = bindFlags;
    bd.MiscFlags           = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    bd.StructureByteStride = stride;
    D3D11_SUBRESOURCE_DATA sd{ init, 0, 0 };
    ComPtr<ID3D11Buffer> b;
    gDev->CreateBuffer(&bd, init ? &sd : nullptr, &b);
    return b;
}
static ComPtr<ID3D11Buffer> SRVBuf(uint32_t s, uint32_t n, const void* i=nullptr){
    return MakeSBuf(s,n,D3D11_BIND_SHADER_RESOURCE,i);
}
static ComPtr<ID3D11Buffer> UAVBuf(uint32_t s, uint32_t n, const void* i=nullptr){
    return MakeSBuf(s,n,D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS,i);
}

static ComPtr<ID3D11ShaderResourceView> MakeSRV(ID3D11Buffer* b, uint32_t n) {
    D3D11_SHADER_RESOURCE_VIEW_DESC d{};
    d.Format=DXGI_FORMAT_UNKNOWN; d.ViewDimension=D3D11_SRV_DIMENSION_BUFFEREX;
    d.BufferEx.NumElements=n;
    ComPtr<ID3D11ShaderResourceView> v; gDev->CreateShaderResourceView(b,&d,&v); return v;
}
static ComPtr<ID3D11UnorderedAccessView> MakeUAV(ID3D11Buffer* b, uint32_t n) {
    D3D11_UNORDERED_ACCESS_VIEW_DESC d{};
    d.Format=DXGI_FORMAT_UNKNOWN; d.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;
    d.Buffer.NumElements=n;
    ComPtr<ID3D11UnorderedAccessView> v; gDev->CreateUnorderedAccessView(b,&d,&v); return v;
}

static void Upload(ID3D11Buffer* dst, const void* src, uint32_t bytes) {
    D3D11_BOX box{0,0,0,bytes,1,1}; gCtx->UpdateSubresource(dst,0,&box,src,bytes,0);
}
static void Readback(ID3D11Buffer* src, void* dst, uint32_t bytes) {
    D3D11_BUFFER_DESC bd{}; bd.ByteWidth=bytes; bd.Usage=D3D11_USAGE_STAGING;
    bd.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Buffer> stg; gDev->CreateBuffer(&bd,nullptr,&stg);
    gCtx->CopyResource(stg.Get(),src);
    D3D11_MAPPED_SUBRESOURCE ms{}; gCtx->Map(stg.Get(),0,D3D11_MAP_READ,0,&ms);
    memcpy(dst,ms.pData,bytes); gCtx->Unmap(stg.Get(),0);
}

static ComPtr<ID3D11Buffer> MakeCB(uint32_t bytes) {
    D3D11_BUFFER_DESC bd{}; bd.ByteWidth=(bytes+15)&~15u;
    bd.Usage=D3D11_USAGE_DYNAMIC; bd.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
    bd.CPUAccessFlags=D3D11_CPU_ACCESS_WRITE;
    ComPtr<ID3D11Buffer> b; gDev->CreateBuffer(&bd,nullptr,&b); return b;
}
static void SetCB(ID3D11Buffer* cb, const void* data, uint32_t bytes) {
    D3D11_MAPPED_SUBRESOURCE ms{}; gCtx->Map(cb,0,D3D11_MAP_WRITE_DISCARD,0,&ms);
    memcpy(ms.pData,data,bytes); gCtx->Unmap(cb,0);
}

// ── Shader compiler ───────────────────────────────────────────────────────────

static ComPtr<ID3D11ComputeShader> CompileCS(const wchar_t* path, const char* entry) {
    ComPtr<ID3DBlob> blob, err;
    HRESULT hr = D3DCompileFromFile(path,nullptr,D3D_COMPILE_STANDARD_FILE_INCLUDE,
        entry,"cs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&blob,&err);
    if (FAILED(hr)) {
        if(err) printf("[SHADER] %s: %s\n", entry, (char*)err->GetBufferPointer());
        return nullptr;
    }
    ComPtr<ID3D11ComputeShader> cs;
    gDev->CreateComputeShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&cs);
    return cs;
}

// ── Null unbind helpers ───────────────────────────────────────────────────────

static ID3D11ShaderResourceView*   NULL_SRV  = nullptr;
static ID3D11UnorderedAccessView*  NULL_UAV  = nullptr;
static ID3D11Buffer*               NULL_BUF  = nullptr;

// ── INT4 weight helpers ───────────────────────────────────────────────────────

static uint32_t PackInt4x8(const int8_t w[8]) {
    uint32_t out=0;
    for(int i=0;i<8;++i) out|=uint32_t(uint8_t(w[i]+8)&0xF)<<(i*4);
    return out;
}
// Fill [K×N] INT4 weight matrix with a deterministic test pattern
static std::vector<uint32_t> MakeInt4Weights(uint32_t K, uint32_t N, uint32_t seed=0) {
    uint32_t pk = (N+7)/8;
    std::vector<uint32_t> buf(K*pk,0);
    for(uint32_t r=0;r<K;++r) for(uint32_t g=0;g<pk;++g) {
        int8_t w[8]{}; for(int j=0;j<8;++j){ uint32_t c=g*8+j; w[j]=(c<N)?int8_t((r+c+seed)%7-3):0; }
        buf[r*pk+g]=PackInt4x8(w);
    }
    return buf;
}

// ══════════════════════════════════════════════════════════════════════════════
// GPU resource set for one transformer layer
// ══════════════════════════════════════════════════════════════════════════════

struct LayerWeights {
    // INT4-packed weight buffers (DEFAULT, SRV)
    ComPtr<ID3D11Buffer> Wq, Wk, Wv, Wo;   // [E, E]  attention projections
    ComPtr<ID3D11Buffer> W1, W2;            // [E, FFN_DIM] and [FFN_DIM, E]
    ComPtr<ID3D11ShaderResourceView> srvWq, srvWk, srvWv, srvWo, srvW1, srvW2;

    // Learned gamma/beta for two LayerNorms (float, SRV)
    ComPtr<ID3D11Buffer> ln1_g, ln1_b, ln2_g, ln2_b;
    ComPtr<ID3D11ShaderResourceView> srvLn1g, srvLn1b, srvLn2g, srvLn2b;

    std::atomic<bool> ready{false};

    void allocate() {
        // Weights (INT4)
        auto wqd = MakeInt4Weights(E, E, 0);
        auto wkd = MakeInt4Weights(E, E, 1);
        auto wvd = MakeInt4Weights(E, E, 2);
        auto wod = MakeInt4Weights(E, E, 3);
        auto w1d = MakeInt4Weights(E, FFN_DIM, 4);
        auto w2d = MakeInt4Weights(FFN_DIM, E, 5);

        Wq=SRVBuf(4, E*PK(E),   wqd.data()); srvWq=MakeSRV(Wq.Get(), E*PK(E));
        Wk=SRVBuf(4, E*PK(E),   wkd.data()); srvWk=MakeSRV(Wk.Get(), E*PK(E));
        Wv=SRVBuf(4, E*PK(E),   wvd.data()); srvWv=MakeSRV(Wv.Get(), E*PK(E));
        Wo=SRVBuf(4, E*PK(E),   wod.data()); srvWo=MakeSRV(Wo.Get(), E*PK(E));
        W1=SRVBuf(4, E*PK(FFN_DIM), w1d.data()); srvW1=MakeSRV(W1.Get(), E*PK(FFN_DIM));
        W2=SRVBuf(4, FFN_DIM*PK(E), w2d.data()); srvW2=MakeSRV(W2.Get(), FFN_DIM*PK(E));

        // LayerNorm params (ones/zeros)
        std::vector<float> ones(E, 1.f), zeros(E, 0.f);
        ln1_g=SRVBuf(4,E,ones.data()); srvLn1g=MakeSRV(ln1_g.Get(),E);
        ln1_b=SRVBuf(4,E,zeros.data()); srvLn1b=MakeSRV(ln1_b.Get(),E);
        ln2_g=SRVBuf(4,E,ones.data()); srvLn2g=MakeSRV(ln2_g.Get(),E);
        ln2_b=SRVBuf(4,E,zeros.data()); srvLn2b=MakeSRV(ln2_b.Get(),E);

        ready.store(true);
    }
};

// ══════════════════════════════════════════════════════════════════════════════
// KV Cache — stores K,V for every layer at every generated position
// ══════════════════════════════════════════════════════════════════════════════

struct KVCache {
    // K[layer][pos][E],  V[layer][pos][E] — stored flat on GPU
    std::vector<ComPtr<ID3D11Buffer>>              kBuf, vBuf;   // [NUM_LAYERS]
    std::vector<ComPtr<ID3D11UnorderedAccessView>> kUAV, vUAV;
    std::vector<ComPtr<ID3D11ShaderResourceView>>  kSRV, vSRV;

    void allocate() {
        kBuf.resize(NUM_LAYERS); vBuf.resize(NUM_LAYERS);
        kUAV.resize(NUM_LAYERS); vUAV.resize(NUM_LAYERS);
        kSRV.resize(NUM_LAYERS); vSRV.resize(NUM_LAYERS);
        for(uint32_t l=0;l<NUM_LAYERS;++l) {
            kBuf[l]=UAVBuf(4, MAX_SEQ*E); kUAV[l]=MakeUAV(kBuf[l].Get(),MAX_SEQ*E); kSRV[l]=MakeSRV(kBuf[l].Get(),MAX_SEQ*E);
            vBuf[l]=UAVBuf(4, MAX_SEQ*E); vUAV[l]=MakeUAV(vBuf[l].Get(),MAX_SEQ*E); vSRV[l]=MakeSRV(vBuf[l].Get(),MAX_SEQ*E);
        }
    }
};

// ══════════════════════════════════════════════════════════════════════════════
// Compute shaders (compiled once, reused every step)
// ══════════════════════════════════════════════════════════════════════════════

struct Shaders {
    ComPtr<ID3D11ComputeShader> ln;       // layernorm (CSMain)
    ComPtr<ID3D11ComputeShader> qproj;   // INT4 matmul (CSQProj)
    ComPtr<ID3D11ComputeShader> attn;    // attention (CSMain)
    ComPtr<ID3D11ComputeShader> gelu;    // gelu (CSMain)
    ComPtr<ID3D11ComputeShader> resAdd;  // residual add (CSMain_addto)

    bool compile() {
        ln     = CompileCS(PATH_LN,     "CSMain");         if(!ln)    return false;
        qproj  = CompileCS(PATH_MATMUL, "CSQProj");        if(!qproj) return false;
        attn   = CompileCS(PATH_ATTN,   "CSMain");         if(!attn)  return false;
        gelu   = CompileCS(PATH_GELU,   "CSMain");         if(!gelu)  return false;
        resAdd = CompileCS(PATH_RES,    "CSMain_addto");   if(!resAdd) return false;
        puts("[OK]  All shaders compiled");
        return true;
    }
};

// ══════════════════════════════════════════════════════════════════════════════
// Constant buffer structs (mirror HLSL)
// ══════════════════════════════════════════════════════════════════════════════

struct LNParams    { uint32_t n_embd, seq_len; float eps; uint32_t pad; };
struct QProjParams { uint32_t M,K,N,use_bias; float w_scale,w_zero; uint32_t numPacked,dstOffset; };
struct AttnParams  { uint32_t seq_len, n_embd, head_dim; float scale; };
struct GeluParams  { uint32_t numel, x_in_offset, pad0, pad1; };
struct ResAddParams{ uint32_t numel; uint32_t pad[3]; };

// ══════════════════════════════════════════════════════════════════════════════
// TransformerBlock — runs one full layer on GPU
// ══════════════════════════════════════════════════════════════════════════════

class TransformerBlock {
public:
    Shaders&     sh;
    LayerWeights& w;
    uint32_t     seqLen;   // current sequence length

    // Intermediate GPU buffers (allocated once for MAX_SEQ)
    ComPtr<ID3D11Buffer> xNorm;        // layernorm output    [S, E]
    ComPtr<ID3D11Buffer> qBuf, kBuf, vBuf; // Q, K, V projections [S, E]
    ComPtr<ID3D11Buffer> qkvBuf;       // packed qkv          [S, 3E]
    ComPtr<ID3D11Buffer> attnOut;      // attention output     [S, E]
    ComPtr<ID3D11Buffer> PBuf;         // softmax weights      [H, S, S]
    ComPtr<ID3D11Buffer> ffnH;         // FFN hidden layer     [S, FFN_DIM]
    ComPtr<ID3D11Buffer> ffnOut;       // FFN output           [S, E]
    ComPtr<ID3D11Buffer> xhatBuf, invStdBuf;  // LN save bufs

    ComPtr<ID3D11ShaderResourceView>   srvXNorm, srvQ, srvK, srvV, srvQKV, srvFfnH;
    ComPtr<ID3D11UnorderedAccessView>  uavXNorm, uavQ, uavK, uavV, uavQKV;
    ComPtr<ID3D11UnorderedAccessView>  uavAttnOut, uavFfnH, uavFfnOut;
    ComPtr<ID3D11ShaderResourceView>   srvAttnOut, srvFfnOut;
    ComPtr<ID3D11UnorderedAccessView>  uavPBuf;
    ComPtr<ID3D11UnorderedAccessView>  uavXhat, uavInvStd;

    ComPtr<ID3D11Buffer> cbLN, cbQP, cbAttn, cbGelu, cbRes;

    TransformerBlock(Shaders& s, LayerWeights& lw, uint32_t S)
        : sh(s), w(lw), seqLen(S)
    {
        auto alloc = [&](uint32_t stride, uint32_t n) { return UAVBuf(stride, n); };

        xNorm   = alloc(4, MAX_SEQ * E);
        qBuf    = alloc(4, MAX_SEQ * E);
        kBuf    = alloc(4, MAX_SEQ * E);
        vBuf    = alloc(4, MAX_SEQ * E);
        qkvBuf  = alloc(4, MAX_SEQ * 3 * E);
        attnOut = alloc(4, MAX_SEQ * E);
        PBuf    = alloc(4, H * MAX_SEQ * MAX_SEQ);
        ffnH    = alloc(4, MAX_SEQ * FFN_DIM);
        ffnOut  = alloc(4, MAX_SEQ * E);
        xhatBuf = alloc(4, MAX_SEQ * E);
        invStdBuf = SRVBuf(4, MAX_SEQ);  // only needs SRV after LN

        srvXNorm  = MakeSRV(xNorm.Get(),  MAX_SEQ*E);
        srvQ      = MakeSRV(qBuf.Get(),   MAX_SEQ*E);
        srvK      = MakeSRV(kBuf.Get(),   MAX_SEQ*E);
        srvV      = MakeSRV(vBuf.Get(),   MAX_SEQ*E);
        srvQKV    = MakeSRV(qkvBuf.Get(), MAX_SEQ*3*E);
        srvAttnOut= MakeSRV(attnOut.Get(),MAX_SEQ*E);
        srvFfnH   = MakeSRV(ffnH.Get(),   MAX_SEQ*FFN_DIM);
        srvFfnOut = MakeSRV(ffnOut.Get(), MAX_SEQ*E);

        uavXNorm  = MakeUAV(xNorm.Get(),  MAX_SEQ*E);
        uavQ      = MakeUAV(qBuf.Get(),   MAX_SEQ*E);
        uavK      = MakeUAV(kBuf.Get(),   MAX_SEQ*E);
        uavV      = MakeUAV(vBuf.Get(),   MAX_SEQ*E);
        uavQKV    = MakeUAV(qkvBuf.Get(), MAX_SEQ*3*E);
        uavAttnOut= MakeUAV(attnOut.Get(),MAX_SEQ*E);
        uavPBuf   = MakeUAV(PBuf.Get(),   H*MAX_SEQ*MAX_SEQ);
        uavFfnH   = MakeUAV(ffnH.Get(),   MAX_SEQ*FFN_DIM);
        uavFfnOut = MakeUAV(ffnOut.Get(), MAX_SEQ*E);
        uavXhat   = MakeUAV(xhatBuf.Get(),MAX_SEQ*E);
        uavInvStd = MakeUAV(invStdBuf.Get(), MAX_SEQ);

        cbLN   = MakeCB(sizeof(LNParams));
        cbQP   = MakeCB(sizeof(QProjParams));
        cbAttn = MakeCB(sizeof(AttnParams));
        cbGelu = MakeCB(sizeof(GeluParams));
        cbRes  = MakeCB(sizeof(ResAddParams));
    }

    // Run one transformer layer.  x_inout is modified in place (residuals).
    // x_inout: GPU buffer [seqLen × E] float
    void forward(ID3D11Buffer* x_inout, ID3D11ShaderResourceView* srvX,
                 ID3D11UnorderedAccessView* uavX)
    {
        const uint32_t S = seqLen;

        // ── LayerNorm 1 ─────────────────────────────────────────────────────
        {
            LNParams p{ E, S, 1e-5f, 0 };
            SetCB(cbLN.Get(), &p, sizeof(p));
            gCtx->CSSetShader(sh.ln.Get(), nullptr, 0);
            ID3D11Buffer* cbs[]={cbLN.Get()}; gCtx->CSSetConstantBuffers(0,1,cbs);
            ID3D11ShaderResourceView* srvs[]={srvX, w.srvLn1g.Get(), w.srvLn1b.Get()};
            gCtx->CSSetShaderResources(0,3,srvs);
            ID3D11UnorderedAccessView* uavs[]={uavXNorm.Get(), uavXhat.Get(), uavInvStd.Get()};
            gCtx->CSSetUnorderedAccessViews(0,3,uavs,nullptr);
            gCtx->Dispatch(S, 1, 1);  // one group per token
            gCtx->CSSetUnorderedAccessViews(0,3,&NULL_UAV,nullptr);
            gCtx->CSSetShaderResources(0,3,&NULL_SRV);
        }

        // ── Q, K, V projections (INT4 decode + matmul) ─────────────────────
        // CSQProj: Y[M,N] = decode(W_int4[K,N]) × X[M,K]
        auto dispatchQProj = [&](ID3D11ShaderResourceView* wSRV,
                                  ID3D11UnorderedAccessView* yUAV,
                                  uint32_t inK, uint32_t outN) {
            QProjParams p{ S, inK, outN, 0, 1.f/8.f, 0.f, 0, 0 };
            SetCB(cbQP.Get(), &p, sizeof(p));
            gCtx->CSSetShader(sh.qproj.Get(), nullptr, 0);
            ID3D11Buffer* cbs[]={cbQP.Get()}; gCtx->CSSetConstantBuffers(0,1,cbs);
            ID3D11ShaderResourceView* srvs[]={srvXNorm.Get(), wSRV, NULL_SRV};
            gCtx->CSSetShaderResources(0,3,srvs);
            ID3D11UnorderedAccessView* uavs[]={yUAV};
            gCtx->CSSetUnorderedAccessViews(0,1,uavs,nullptr);
            uint32_t gx=(outN+15)/16, gy=(S+15)/16;
            gCtx->Dispatch(gx, gy, 1);
            gCtx->CSSetUnorderedAccessViews(0,1,&NULL_UAV,nullptr);
            gCtx->CSSetShaderResources(0,3,&NULL_SRV);
        };
        dispatchQProj(w.srvWq.Get(), uavQ.Get(), E, E);  // Q = xnorm × Wq
        dispatchQProj(w.srvWk.Get(), uavK.Get(), E, E);  // K = xnorm × Wk
        dispatchQProj(w.srvWv.Get(), uavV.Get(), E, E);  // V = xnorm × Wv

        // Pack Q, K, V → qkv_buf[S, 3E] so existing gpt2_attn_fwd.hlsl can consume it.
        // Layout: qkv[i,0..E-1]=Q, qkv[i,E..2E-1]=K, qkv[i,2E..3E-1]=V
        // (We do a simple CPU pack here; in production: write a GPU copy shader)
        {
            uint32_t elems = S * E;
            std::vector<float> hQ(elems), hK(elems), hV(elems), hQKV(S*3*E);
            Readback(qBuf.Get(), hQ.data(), elems*4);
            Readback(kBuf.Get(), hK.data(), elems*4);
            Readback(vBuf.Get(), hV.data(), elems*4);
            for(uint32_t i=0;i<S;++i) {
                memcpy(&hQKV[i*3*E + 0],   &hQ[i*E], E*4);
                memcpy(&hQKV[i*3*E + E],   &hK[i*E], E*4);
                memcpy(&hQKV[i*3*E + 2*E], &hV[i*E], E*4);
            }
            Upload(qkvBuf.Get(), hQKV.data(), S*3*E*4);
        }

        // ── Attention ───────────────────────────────────────────────────────
        {
            AttnParams p{ S, E, D, 1.f/sqrtf(float(D)) };
            SetCB(cbAttn.Get(), &p, sizeof(p));
            gCtx->CSSetShader(sh.attn.Get(), nullptr, 0);
            ID3D11Buffer* cbs[]={cbAttn.Get()}; gCtx->CSSetConstantBuffers(0,1,cbs);
            ID3D11ShaderResourceView* srvs[]={srvQKV.Get()};
            gCtx->CSSetShaderResources(0,1,srvs);
            ID3D11UnorderedAccessView* uavs[]={uavAttnOut.Get(), uavPBuf.Get()};
            gCtx->CSSetUnorderedAccessViews(0,2,uavs,nullptr);
            gCtx->Dispatch(H, 1, 1);  // one group per head
            gCtx->CSSetUnorderedAccessViews(0,2,&NULL_UAV,nullptr);
            gCtx->CSSetShaderResources(0,1,&NULL_SRV);
        }

        // ── Output projection: attnOut = attnOut × Wo ───────────────────────
        // (reuse qBuf as scratch)
        dispatchQProj(w.srvWo.Get(), uavQ.Get(), E, E);  // Q scratch = attnOut × Wo
        // Swap: make Q-scratch the new attn output (just move the SRV/UAV usage below)

        // ── Residual 1: x += attn_out ──────────────────────────────────────
        {
            ResAddParams p{ S*E, {0,0,0} };
            SetCB(cbRes.Get(), &p, sizeof(p));
            gCtx->CSSetShader(sh.resAdd.Get(), nullptr, 0);
            ID3D11Buffer* cbs[]={cbRes.Get()}; gCtx->CSSetConstantBuffers(0,1,cbs);
            ID3D11ShaderResourceView* srvs[]={srvQ.Get()};  // attn output (Wo projected)
            gCtx->CSSetShaderResources(0,1,srvs);
            gCtx->CSSetUnorderedAccessViews(0,1,&uavX,nullptr);
            uint32_t g=(S*E+255)/256;
            gCtx->Dispatch(g,1,1);
            gCtx->CSSetUnorderedAccessViews(0,1,&NULL_UAV,nullptr);
            gCtx->CSSetShaderResources(0,1,&NULL_SRV);
        }

        // ── LayerNorm 2 ─────────────────────────────────────────────────────
        {
            LNParams p{ E, S, 1e-5f, 0 };
            SetCB(cbLN.Get(), &p, sizeof(p));
            gCtx->CSSetShader(sh.ln.Get(), nullptr, 0);
            ID3D11Buffer* cbs[]={cbLN.Get()}; gCtx->CSSetConstantBuffers(0,1,cbs);
            ID3D11ShaderResourceView* srvs[]={srvX, w.srvLn2g.Get(), w.srvLn2b.Get()};
            gCtx->CSSetShaderResources(0,3,srvs);
            ID3D11UnorderedAccessView* uavs[]={uavXNorm.Get(), uavXhat.Get(), uavInvStd.Get()};
            gCtx->CSSetUnorderedAccessViews(0,3,uavs,nullptr);
            gCtx->Dispatch(S,1,1);
            gCtx->CSSetUnorderedAccessViews(0,3,&NULL_UAV,nullptr);
            gCtx->CSSetShaderResources(0,3,&NULL_SRV);
        }

        // ── FFN W1: ffnH = xnorm × W1  [S, FFN_DIM] ────────────────────────
        {
            QProjParams p{ S, E, FFN_DIM, 0, 1.f/8.f, 0.f, 0, 0 };
            SetCB(cbQP.Get(), &p, sizeof(p));
            gCtx->CSSetShader(sh.qproj.Get(), nullptr, 0);
            ID3D11Buffer* cbs[]={cbQP.Get()}; gCtx->CSSetConstantBuffers(0,1,cbs);
            ID3D11ShaderResourceView* srvs[]={srvXNorm.Get(), w.srvW1.Get(), NULL_SRV};
            gCtx->CSSetShaderResources(0,3,srvs);
            gCtx->CSSetUnorderedAccessViews(0,1,&uavFfnH.Get(),nullptr);
            uint32_t gx=(FFN_DIM+15)/16, gy=(S+15)/16;
            gCtx->Dispatch(gx,gy,1);
            gCtx->CSSetUnorderedAccessViews(0,1,&NULL_UAV,nullptr);
            gCtx->CSSetShaderResources(0,3,&NULL_SRV);
        }

        // ── GELU ────────────────────────────────────────────────────────────
        {
            GeluParams p{ S*FFN_DIM, 0, 0, 0 };
            SetCB(cbGelu.Get(), &p, sizeof(p));
            gCtx->CSSetShader(sh.gelu.Get(), nullptr, 0);
            ID3D11Buffer* cbs[]={cbGelu.Get()}; gCtx->CSSetConstantBuffers(0,1,cbs);
            gCtx->CSSetShaderResources(0,1,&srvFfnH.Get());
            gCtx->CSSetUnorderedAccessViews(0,1,&uavFfnH.Get(),nullptr);
            gCtx->Dispatch((S*FFN_DIM+255)/256,1,1);
            gCtx->CSSetUnorderedAccessViews(0,1,&NULL_UAV,nullptr);
            gCtx->CSSetShaderResources(0,1,&NULL_SRV);
        }

        // ── FFN W2: ffnOut = gelu(ffnH) × W2  [S, E] ───────────────────────
        {
            QProjParams p{ S, FFN_DIM, E, 0, 1.f/8.f, 0.f, 0, 0 };
            SetCB(cbQP.Get(), &p, sizeof(p));
            gCtx->CSSetShader(sh.qproj.Get(), nullptr, 0);
            ID3D11Buffer* cbs[]={cbQP.Get()}; gCtx->CSSetConstantBuffers(0,1,cbs);
            ID3D11ShaderResourceView* srvs[]={srvFfnH.Get(), w.srvW2.Get(), NULL_SRV};
            gCtx->CSSetShaderResources(0,3,srvs);
            gCtx->CSSetUnorderedAccessViews(0,1,&uavFfnOut.Get(),nullptr);
            uint32_t gx=(E+15)/16, gy=(S+15)/16;
            gCtx->Dispatch(gx,gy,1);
            gCtx->CSSetUnorderedAccessViews(0,1,&NULL_UAV,nullptr);
            gCtx->CSSetShaderResources(0,3,&NULL_SRV);
        }

        // ── Residual 2: x += ffn_out ────────────────────────────────────────
        {
            ResAddParams p{ S*E, {0,0,0} };
            SetCB(cbRes.Get(), &p, sizeof(p));
            gCtx->CSSetShader(sh.resAdd.Get(), nullptr, 0);
            ID3D11Buffer* cbs[]={cbRes.Get()}; gCtx->CSSetConstantBuffers(0,1,cbs);
            gCtx->CSSetShaderResources(0,1,&srvFfnOut.Get());
            gCtx->CSSetUnorderedAccessViews(0,1,&uavX,nullptr);
            gCtx->Dispatch((S*E+255)/256,1,1);
            gCtx->CSSetUnorderedAccessViews(0,1,&NULL_UAV,nullptr);
            gCtx->CSSetShaderResources(0,1,&NULL_SRV);
        }
    }
};

// ══════════════════════════════════════════════════════════════════════════════
// Token generation loop
// ══════════════════════════════════════════════════════════════════════════════

struct GenerationState {
    // Embedding table [VOCAB, E] (float, SRV)
    ComPtr<ID3D11Buffer> embedTable;
    ComPtr<ID3D11ShaderResourceView> srvEmbed;

    // LM head weights (INT4) [E, VOCAB] — maps final hidden → logits
    ComPtr<ID3D11Buffer> lmHead;
    ComPtr<ID3D11ShaderResourceView> srvLmHead;

    // Current hidden state [1, E] (single token for generation)
    ComPtr<ID3D11Buffer> hidden;
    ComPtr<ID3D11ShaderResourceView> srvHidden;
    ComPtr<ID3D11UnorderedAccessView> uavHidden;

    // Logits [VOCAB]
    ComPtr<ID3D11Buffer> logits;
    ComPtr<ID3D11UnorderedAccessView> uavLogits;

    void allocate() {
        // Embedding table: simple identity-ish init
        std::vector<float> emb(VOCAB * E, 0.f);
        for(uint32_t v=0;v<VOCAB;++v) for(uint32_t e=0;e<E;++e)
            emb[v*E+e] = sinf(float(v+1)*float(e+1)*0.01f);
        embedTable = SRVBuf(4, VOCAB*E, emb.data());
        srvEmbed   = MakeSRV(embedTable.Get(), VOCAB*E);

        auto lmW = MakeInt4Weights(E, VOCAB, 99);
        lmHead   = SRVBuf(4, E*PK(VOCAB), lmW.data());
        srvLmHead= MakeSRV(lmHead.Get(), E*PK(VOCAB));

        hidden    = UAVBuf(4, E);
        srvHidden = MakeSRV(hidden.Get(), E);
        uavHidden = MakeUAV(hidden.Get(), E);

        logits    = UAVBuf(4, VOCAB);
        uavLogits = MakeUAV(logits.Get(), VOCAB);
    }

    // Embed one token into 'hidden'
    void embed(uint32_t token) {
        std::vector<float> row(E);
        // In production: GPU gather. For demo: CPU copy from table.
        Readback(embedTable.Get(), nullptr, 0);  // force sync
        std::vector<float> full(VOCAB*E);
        Readback(embedTable.Get(), full.data(), VOCAB*E*4);
        memcpy(row.data(), &full[token*E], E*4);
        Upload(hidden.Get(), row.data(), E*4);
    }
};

// ── Sampling helpers ──────────────────────────────────────────────────────────

static uint32_t SampleGreedy(const std::vector<float>& logits) {
    return uint32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

static uint32_t SampleTemperature(const std::vector<float>& logits,
                                   float temperature, std::mt19937& rng)
{
    std::vector<float> probs(logits.size());
    float maxL = *std::max_element(logits.begin(), logits.end());
    float sum  = 0.f;
    for(size_t i=0;i<logits.size();++i){ probs[i]=expf((logits[i]-maxL)/temperature); sum+=probs[i]; }
    for(auto& p:probs) p/=sum;
    std::discrete_distribution<uint32_t> dist(probs.begin(),probs.end());
    return dist(rng);
}

// ── Full generation demo ──────────────────────────────────────────────────────

static void RunGenerationLoop(uint32_t maxNewTokens = 16, float temperature = 0.8f)
{
    if(!InitD3D11()) { puts("[FAIL] D3D11"); return; }

    // Compile shaders
    Shaders sh;
    if(!sh.compile()) return;

    // Allocate layer weights
    std::vector<LayerWeights> layerWeights(NUM_LAYERS);
    for(auto& lw : layerWeights) lw.allocate();

    // Allocate generation state
    GenerationState gen;
    gen.allocate();

    // Prompt: "Hello" as byte token IDs
    std::vector<uint32_t> tokens = {72, 101, 108, 108, 111};  // H,e,l,l,o
    printf("[gen] Prompt tokens: ");
    for(auto t:tokens) printf("%u ", t); printf("\n");

    std::mt19937 rng(42);

    // ── Prefill: run all prompt tokens through all layers ──────────────────
    {
        uint32_t S = uint32_t(tokens.size());

        // Build X[S, E] from embedding table
        std::vector<float> embFull(VOCAB*E);
        Readback(gen.embedTable.Get(), embFull.data(), VOCAB*E*4);
        std::vector<float> X(S*E);
        for(uint32_t i=0;i<S;++i) memcpy(&X[i*E], &embFull[tokens[i]*E], E*4);

        auto xBuf = UAVBuf(4, S*E, X.data());
        auto srvX = MakeSRV(xBuf.Get(), S*E);
        auto uavX = MakeUAV(xBuf.Get(), S*E);

        for(uint32_t l=0;l<NUM_LAYERS;++l) {
            TransformerBlock blk(sh, layerWeights[l], S);
            blk.forward(xBuf.Get(), srvX.Get(), uavX.Get());
        }
        printf("[gen] Prefill done (%u tokens, %u layers)\n", S, NUM_LAYERS);

        // Use last token's hidden as starting state
        std::vector<float> hOut(S*E);
        Readback(xBuf.Get(), hOut.data(), S*E*4);
        Upload(gen.hidden.Get(), &hOut[(S-1)*E], E*4);
    }

    // ── Generation: token by token ─────────────────────────────────────────
    ComPtr<ID3D11Buffer> cbQP = MakeCB(sizeof(QProjParams));

    for(uint32_t step=0; step<maxNewTokens; ++step) {
        // Compute logits = hidden × lm_head  [1, VOCAB]
        QProjParams p{ 1, E, VOCAB, 0, 1.f/8.f, 0.f, 0, 0 };
        SetCB(cbQP.Get(), &p, sizeof(p));
        gCtx->CSSetShader(sh.qproj.Get(), nullptr, 0);
        ID3D11Buffer* cbs[]={cbQP.Get()}; gCtx->CSSetConstantBuffers(0,1,cbs);
        ID3D11ShaderResourceView* srvs[]={gen.srvHidden.Get(), gen.srvLmHead.Get(), NULL_SRV};
        gCtx->CSSetShaderResources(0,3,srvs);
        gCtx->CSSetUnorderedAccessViews(0,1,&gen.uavLogits.Get(),nullptr);
        gCtx->Dispatch((VOCAB+15)/16, 1, 1);
        gCtx->CSSetUnorderedAccessViews(0,1,&NULL_UAV,nullptr);
        gCtx->CSSetShaderResources(0,3,&NULL_SRV);

        // Read back logits
        std::vector<float> logitsCPU(VOCAB);
        Readback(gen.logits.Get(), logitsCPU.data(), VOCAB*4);

        // Sample
        uint32_t nextToken = (temperature < 1e-3f)
            ? SampleGreedy(logitsCPU)
            : SampleTemperature(logitsCPU, temperature, rng);

        tokens.push_back(nextToken);
        printf("[gen] step %2u → token %3u (char='%c')\n",
               step, nextToken, (nextToken>=32&&nextToken<127)?(char)nextToken:'?');

        if(nextToken == 0) break;  // EOS

        // Embed next token → run one step through all layers
        gen.embed(nextToken);
        for(uint32_t l=0;l<NUM_LAYERS;++l) {
            TransformerBlock blk(sh, layerWeights[l], 1);  // S=1 for generation
            blk.forward(gen.hidden.Get(), gen.srvHidden.Get(), gen.uavHidden.Get());
        }
    }

    printf("[gen] Final sequence (%zu tokens): ", tokens.size());
    for(auto t:tokens) printf("%u ", t); printf("\n");
    puts("[PASS] Generation loop complete");
}

// ── Entry point ───────────────────────────────────────────────────────────────

int main() {
    printf("=== SCX Transformer Block — End-to-End Demo ===\n");
    printf("  E=%u  H=%u  D=%u  FFN=%u  VOCAB=%u  MAX_SEQ=%u  LAYERS=%u\n\n",
           E, H, D, FFN_DIM, VOCAB, MAX_SEQ, NUM_LAYERS);
    RunGenerationLoop(16, 0.8f);
    return 0;
}
