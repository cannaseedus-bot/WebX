// xshard_tile_test.cpp
// Step 1 (upload loop):  disk → GPU → CopyResource → memcmp       (no shader)
// Step 2 (--shader):     disk → SRV → identity CS → UAV → memcmp
// Step 3 (--qk):         Q+K tiles → CS dot-product → 64×64 scores vs CPU ref
// Step 4 (--softmax):    Q+K → fused scale+softmax → probabilities vs CPU ref
// Step 5 (--v):          Q+K+V → softmax → P×V → 64×1024 context vs CPU ref
// Step 6 (--proj):       ctx[64,1024] × W_o[64,1024]^T → out[64,64] vs CPU ref
// Step 7 (--fold):       16 O-proj partials[16×64×64] + residual → [64,1024] LAYER_FOLD
//
// Usage:
//   xshard_tile_test.exe <q_shard> <iters>
//   xshard_tile_test.exe <q_shard> <iters> --shader
//   xshard_tile_test.exe <q_shard> <iters> --qk     <k_shard>
//   xshard_tile_test.exe <q_shard> <iters> --softmax <k_shard>
//   xshard_tile_test.exe <q_shard> <iters> --v       <k_shard> <v_shard>
//   xshard_tile_test.exe <q_shard> <iters> --proj    <o_shard>
//   xshard_tile_test.exe <q_shard> <iters> --fold    <o_shard>

#include "d3d11_engine.h"
#include "xshard.h"

#include <d3dcompiler.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

// ── embedded HLSL shaders ─────────────────────────────────────────────────────

// Step 2: identity copy (typed Buffer<float>, no STRUCTURED flag needed)
static const char* kTileCopyHLSL = R"hlsl(
Buffer<float>   inBuf  : register(t0);
RWBuffer<float> outBuf : register(u0);
[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i < 65536u) outBuf[i] = inBuf[i];
}
)hlsl";

// Step 3: Q×K^T dot product
// Tile layout: each tile is [64 rows × 1024 cols] = 65536 floats (reinterpreted)
// Q[row, i] = Q_buf[row * 1024 + i]   row ∈ [0,64)  i ∈ [0,1024)
// K[col, i] = K_buf[col * 1024 + i]   col ∈ [0,64)
// Out[row, col] = dot(Q[row,:], K[col,:])   shape [64,64] = 4096 floats
static const char* kQKdotHLSL = R"hlsl(
Buffer<float>   Q   : register(t0);   // [64 x 1024]
Buffer<float>   K   : register(t1);   // [64 x 1024]
RWBuffer<float> Out : register(u0);   // [64 x 64]

#define D_MODEL 1024u
#define SEQ     64u

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint row = id.x;
    uint col = id.y;
    if (row >= SEQ || col >= SEQ) return;
    float sum = 0.0f;
    for (uint i = 0u; i < D_MODEL; i++)
        sum += Q[row * D_MODEL + i] * K[col * D_MODEL + i];
    Out[row * SEQ + col] = sum;
}
)hlsl";

// Step 4: Q×K^T + scale(1/√1024) + row-wise numerically-stable softmax
//
// Layout: same as --qk  (Q[64,1024], K[64,1024] → P[64,64])
//
// FIX vs naive 8×8 kernel: the spec's 8×8 grouping only covers 8 of 64 columns
// per row, so its max/sum reduction is incomplete — wrong softmax.
// Correct approach: ONE GROUP PER ROW, 64 threads (one per col).
//   [numthreads(64,1,1)]  +  Dispatch(64,1,1)
//   groupshared[64] covers the full row → reduction is correct.
static const char* kSoftmaxHLSL = R"hlsl(
Buffer<float>   Q   : register(t0);   // [64 x 1024]
Buffer<float>   K   : register(t1);   // [64 x 1024]
RWBuffer<float> Out : register(u0);   // [64 x 64]

#define D_MODEL    1024u
#define SEQ        64u
#define INV_SQRT_D 0.03125f            // 1/sqrt(1024) = 1/32

groupshared float sdata[SEQ];          // 256 bytes — well under SM5 limit

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID,
          uint3 tid : SV_GroupThreadID)
{
    uint row = gid.x;   // 0..63  (one group per output row)
    uint col = tid.x;   // 0..63  (one thread per output column)

    // 1. Dot product + scale
    float s = 0.0f;
    for (uint i = 0u; i < D_MODEL; i++)
        s += Q[row * D_MODEL + i] * K[col * D_MODEL + i];
    s *= INV_SQRT_D;

    sdata[col] = s;
    GroupMemoryBarrierWithGroupSync();

    // 2. Parallel max reduction (log2(64) = 6 passes)
    for (uint stride = 32u; stride > 0u; stride >>= 1u) {
        if (col < stride)
            sdata[col] = max(sdata[col], sdata[col + stride]);
        GroupMemoryBarrierWithGroupSync();
    }
    float m = sdata[0];
    GroupMemoryBarrierWithGroupSync();   // guard before reusing sdata

    // 3. exp(s - max)
    float e = exp(s - m);
    sdata[col] = e;
    GroupMemoryBarrierWithGroupSync();

    // 4. Parallel sum reduction
    for (uint stride = 32u; stride > 0u; stride >>= 1u) {
        if (col < stride)
            sdata[col] += sdata[col + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    float denom = sdata[0];

    // 5. Normalize
    Out[row * SEQ + col] = e / denom;
}
)hlsl";

// Step 5: P × V  (attention head context)
// P[64,64] × V[64,1024] → ctx[64,1024] = 65536 floats = 256KB (one full tile)
// P[row,k] = P_buf[row*64 + k]
// V[k,col] = V_buf[k*1024 + col]
// ctx[row,col] = sum_{k=0}^{63} P[row,k] * V[k,col]
static const char* kVmulHLSL = R"hlsl(
Buffer<float>   P   : register(t0);   // [64 x 64]   softmax weights
Buffer<float>   V   : register(t1);   // [64 x 1024] value tile
RWBuffer<float> Out : register(u0);   // [64 x 1024] context vectors

#define SEQ     64u
#define D_MODEL 1024u

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint row = id.x;   // 0..63
    uint col = id.y;   // 0..1023
    if (row >= SEQ || col >= D_MODEL) return;
    float s = 0.f;
    for (uint k = 0u; k < SEQ; k++)
        s += P[row * SEQ + k] * V[k * D_MODEL + col];
    Out[row * D_MODEL + col] = s;
}
)hlsl";

// Step 6: O-projection tile
// ctx[SEQ, N_EMBD] × W_o[OUT, N_EMBD]^T → out[SEQ, OUT]
// One tile: SEQ=64, N_EMBD=1024, OUT=64  (64 output dims per tile)
// Full O-projection needs 16 such tiles summed (LAYER_FOLD, next gate)
// Math: out[s,j] = sum_{i=0}^{1023} ctx[s,i] * W_o[j,i]   (no scale — plain GEMM)
static const char* kOProjHLSL = R"hlsl(
Buffer<float>   Ctx : register(t0);   // [64 x 1024] attention context
Buffer<float>   Wo  : register(t1);   // [64 x 1024] O-weight tile (64 output rows)
RWBuffer<float> Out : register(u0);   // [64 x 64]   projected output (64 seq × 64 out)

#define SEQ    64u
#define N_EMBD 1024u

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint s = id.x;   // 0..63  sequence position
    uint j = id.y;   // 0..63  output dim within tile
    if (s >= SEQ || j >= SEQ) return;
    float sum = 0.f;
    for (uint i = 0u; i < N_EMBD; i++)
        sum += Ctx[s * N_EMBD + i] * Wo[j * N_EMBD + i];
    Out[s * SEQ + j] = sum;
}
)hlsl";

// Step 7: LAYER_FOLD — concatenate 16 O-proj partial outputs + residual add
// Partial[h][s][j] packed as Partial[h*4096 + s*64 + j]
// out[s, h*64+j] = Partial[h*4096 + s*64 + j] + Residual[s*1024 + h*64 + j]
static const char* kFoldHLSL = R"hlsl(
Buffer<float>   Partial  : register(t0);  // [16 x 64 x 64] = 65536 floats
Buffer<float>   Residual : register(t1);  // [64 x 1024]    = 65536 floats
RWBuffer<float> Out      : register(u0);  // [64 x 1024]    = 65536 floats

#define SEQ      64u
#define N_EMBD   1024u
#define HEAD_DIM 64u

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint idx = id.x;
    if (idx >= SEQ * N_EMBD) return;
    uint s   = idx / N_EMBD;
    uint col = idx % N_EMBD;
    uint h   = col / HEAD_DIM;
    uint j   = col % HEAD_DIM;
    Out[idx] = Partial[h * SEQ * HEAD_DIM + s * HEAD_DIM + j] + Residual[idx];
}
)hlsl";

// ── helpers ──────────────────────────────────────────────────────────────────

static void die(const char* msg) {
    fprintf(stderr, "[tile_test] FATAL: %s\n", msg); exit(1);
}
static void chk(HRESULT hr, const char* label) {
    if (FAILED(hr)) {
        fprintf(stderr, "[tile_test] %s  hr=0x%08X\n", label, (unsigned)hr);
        exit(1);
    }
}
static ComPtr<ID3D11ComputeShader> compile_cs(ID3D11Device* dev,
                                               const char* src,
                                               const char* name) {
    ComPtr<ID3DBlob> blob, errs;
    HRESULT hr = D3DCompile(src, strlen(src), name, nullptr, nullptr,
                            "main", "cs_5_0",
                            D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &blob, &errs);
    if (FAILED(hr)) {
        if (errs) fprintf(stderr, "%s\n", (char*)errs->GetBufferPointer());
        die("D3DCompile failed");
    }
    ComPtr<ID3D11ComputeShader> cs;
    chk(dev->CreateComputeShader(blob->GetBufferPointer(),
                                  blob->GetBufferSize(), nullptr, &cs),
        "CreateComputeShader");
    return cs;
}
static ComPtr<ID3D11Buffer> make_buf(ID3D11Device* dev, UINT bytes,
                                      D3D11_USAGE usage, UINT bind, UINT cpu) {
    D3D11_BUFFER_DESC d{};
    d.ByteWidth = bytes; d.Usage = usage;
    d.BindFlags = bind; d.CPUAccessFlags = cpu;
    ComPtr<ID3D11Buffer> b;
    chk(dev->CreateBuffer(&d, nullptr, &b), "CreateBuffer");
    return b;
}
static ComPtr<ID3D11ShaderResourceView> make_srv(ID3D11Device* dev,
                                                  ID3D11Buffer* buf,
                                                  UINT n_elems) {
    D3D11_SHADER_RESOURCE_VIEW_DESC d{};
    d.Format              = DXGI_FORMAT_R32_FLOAT;
    d.ViewDimension       = D3D11_SRV_DIMENSION_BUFFER;
    d.Buffer.NumElements  = n_elems;
    ComPtr<ID3D11ShaderResourceView> srv;
    chk(dev->CreateShaderResourceView(buf, &d, &srv), "CreateSRV");
    return srv;
}
static ComPtr<ID3D11UnorderedAccessView> make_uav(ID3D11Device* dev,
                                                   ID3D11Buffer* buf,
                                                   UINT n_elems) {
    D3D11_UNORDERED_ACCESS_VIEW_DESC d{};
    d.Format             = DXGI_FORMAT_R32_FLOAT;
    d.ViewDimension      = D3D11_UAV_DIMENSION_BUFFER;
    d.Buffer.NumElements = n_elems;
    ComPtr<ID3D11UnorderedAccessView> uav;
    chk(dev->CreateUnorderedAccessView(buf, &d, &uav), "CreateUAV");
    return uav;
}

// ── xshard loader ─────────────────────────────────────────────────────────────

struct ShardFile {
    FILE*        fp              = nullptr;
    XShardHeader header          = {};
    uint64_t     tile_data_bytes = 0;
    uint64_t     tile_stride     = 0;

    bool open(const char* path) {
        fp = fopen(path, "rb");
        if (!fp) { fprintf(stderr, "[shard] cannot open %s\n", path); return false; }
        if (fread(&header, 1, sizeof(header), fp) != sizeof(header))
            { fclose(fp); return false; }
        if (!xshard_valid_magic(header))
            { fprintf(stderr, "[shard] bad magic\n"); fclose(fp); return false; }
        tile_data_bytes = (uint64_t)header.tile_size * 4;
        tile_stride     = xshard_tile_bytes(header);
        return true;
    }
    bool read_tile(uint32_t t, void* buf) {
        uint64_t off = xshard_tile_offset(header, t);
        if (fseek(fp, (long)off, SEEK_SET) != 0) return false;
        return fread(buf, 1, tile_data_bytes, fp) == tile_data_bytes;
    }
    ~ShardFile() { if (fp) fclose(fp); }
};

// ── CPU reference: O-projection tile ─────────────────────────────────────────
// ctx[64,1024] × W_o[64,1024]^T → out[64,64]  (no scale — plain GEMM)
static void cpu_proj(const float* ctx, const float* Wo, float* out) {
    for (int s = 0; s < 64; ++s)
        for (int j = 0; j < 64; ++j) {
            float sum = 0.f;
            for (int i = 0; i < 1024; ++i)
                sum += ctx[s*1024+i] * Wo[j*1024+i];
            out[s*64+j] = sum;
        }
}

// ── CPU reference: LAYER_FOLD — concatenate 16 partial O-proj outputs + residual ─────
// partials[h*4096 + s*64 + j] → out[s*1024 + h*64+j] + residual[s*1024 + h*64+j]
static void cpu_fold(const float* partials, const float* residual, float* out) {
    for (int s = 0; s < 64; ++s)
        for (int h = 0; h < 16; ++h)
            for (int j = 0; j < 64; ++j) {
                int idx = s*1024 + h*64+j;
                out[idx] = partials[h*4096 + s*64 + j] + residual[idx];
            }
}

// ── CPU reference: Q×K^T dot product ─────────────────────────────────────────
// Q[64,1024], K[64,1024] → Out[64,64]
static void cpu_qk(const float* Q, const float* K, float* out) {
    for (int r = 0; r < 64; ++r)
        for (int c = 0; c < 64; ++c) {
            float s = 0.f;
            for (int i = 0; i < 1024; ++i)
                s += Q[r*1024+i] * K[c*1024+i];
            out[r*64+c] = s;
        }
}

// ── CPU reference: P × V context multiply ────────────────────────────────────
// P[64,64] × V[64,1024] → ctx[64,1024]
static void cpu_v_mul(const float* P, const float* V, float* out) {
    for (int r = 0; r < 64; ++r)
        for (int c = 0; c < 1024; ++c) {
            float s = 0.f;
            for (int k = 0; k < 64; ++k)
                s += P[r*64+k] * V[k*1024+c];
            out[r*1024+c] = s;
        }
}

// ── CPU reference: scaled dot-product + row-wise stable softmax ───────────────
// Q[64,1024], K[64,1024] → P[64,64]  (rows sum to 1.0)
static void cpu_softmax(const float* Q, const float* K, float* out) {
    for (int r = 0; r < 64; ++r) {
        float scores[64];
        float maxval = -1e30f;
        for (int c = 0; c < 64; ++c) {
            float s = 0.f;
            for (int i = 0; i < 1024; ++i)
                s += Q[r*1024+i] * K[c*1024+i];
            scores[c] = s * 0.03125f;   // scale by 1/sqrt(1024)
            if (scores[c] > maxval) maxval = scores[c];
        }
        float sum = 0.f;
        for (int c = 0; c < 64; ++c) { scores[c] = expf(scores[c] - maxval); sum += scores[c]; }
        for (int c = 0; c < 64; ++c) out[r*64+c] = scores[c] / sum;
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    const char* q_path     = (argc > 1) ? argv[1]
        : "C:\\Users\\canna\\.gpu_trainer\\shards\\layer_00_q.xshard";
    const int   ITERS      = (argc > 2) ? atoi(argv[2]) : 10000;
    bool use_shader = false, use_qk = false, use_softmax = false, use_v = false, use_proj = false, use_fold = false;
    const char* k_path = nullptr;
    const char* v_path = nullptr;
    for (int i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "--shader")  == 0) use_shader = true;
        if (strcmp(argv[i], "--qk")      == 0) { use_qk      = true; k_path = argv[++i]; }
        if (strcmp(argv[i], "--softmax") == 0) { use_softmax = true; k_path = argv[++i]; }
        if (strcmp(argv[i], "--v")       == 0) { use_v       = true; k_path = argv[++i]; v_path = argv[++i]; }
        if (strcmp(argv[i], "--proj")    == 0) { use_proj    = true; k_path = argv[++i]; }
        if (strcmp(argv[i], "--fold")    == 0) { use_fold    = true; k_path = argv[++i]; }
    }
    const bool need_k = use_qk || use_softmax || use_v || use_proj || use_fold;
    const bool need_v = use_v;

    const char* mode = use_fold     ? "LAYER_FOLD: 16 O-proj partials → [64,1024] + residual add"
                     : use_proj     ? "O-projection tile: ctx×W_oᵀ → [64,64]"
                     : use_v        ? "Q×Kᵀ softmax → P×V context (full attn head)"
                     : use_softmax  ? "Q×Kᵀ + scale + softmax"
                     : use_qk       ? "Q×K dot-product"
                     : use_shader   ? "identity CS"
                     :               "upload only";
    printf("[tile_test] q_shard: %s\n", q_path);
    if (need_k) printf("[tile_test] k_shard: %s\n", k_path);
    if (need_v) printf("[tile_test] v_shard: %s\n", v_path);
    printf("[tile_test] iters  : %d\n", ITERS);
    printf("[tile_test] mode   : %s\n\n", mode);

    // ── open shards ───────────────────────────────────────────────────────────
    ShardFile q_shard;
    if (!q_shard.open(q_path)) die("cannot open Q shard");

    ShardFile k_shard;
    if (need_k && (!k_path || !k_shard.open(k_path))) die("cannot open K shard");

    ShardFile v_shard;
    if (need_v && (!v_path || !v_shard.open(v_path))) die("cannot open V shard");

    const uint32_t tile_count = q_shard.header.tile_count;
    const size_t   tile_bytes = (size_t)q_shard.tile_data_bytes;  // 262144
    const uint32_t tile_elems = q_shard.header.tile_size;          // 65536

    printf("[tile_test] tile: %u × %zu bytes (%u floats)\n\n",
           tile_count, tile_bytes, tile_elems);

    std::vector<uint8_t> q_cpu(tile_bytes);
    std::vector<uint8_t> k_cpu(tile_bytes);
    std::vector<uint8_t> v_cpu(tile_bytes);

    // ── D3D11 init ────────────────────────────────────────────────────────────
    D3D11Engine eng;
    if (!eng.init(false, true)) die("D3D11 init failed");
    ID3D11Device*        dev = eng.rawDevice();
    ID3D11DeviceContext* ctx = eng.rawCtx();
    printf("[tile_test] adapter: %s\n\n", eng.adapterName().c_str());

    // ── common: Q tile buffer + staging readback ──────────────────────────────
    auto gQBuf    = make_buf(dev, (UINT)tile_bytes,
                             D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0);
    auto gReadback = make_buf(dev, (UINT)tile_bytes,
                              D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ);

    // ── mode-specific setup ───────────────────────────────────────────────────
    ComPtr<ID3D11ShaderResourceView>  gQ_SRV, gK_SRV, gIn_SRV;
    ComPtr<ID3D11Buffer>              gKBuf, gOutBuf, gOutReadback;
    ComPtr<ID3D11UnorderedAccessView> gOut_UAV, gIdent_UAV;
    ComPtr<ID3D11Buffer>              gIdentOut;
    ComPtr<ID3D11ComputeShader>       gCS;
    // --v mode extras
    ComPtr<ID3D11Buffer>              gVBuf, gPBuf, gCtxBuf, gCtxReadback;
    ComPtr<ID3D11ShaderResourceView>  gV_SRV, gP_SRV;
    ComPtr<ID3D11UnorderedAccessView> gP_UAV, gCtx_UAV;
    ComPtr<ID3D11ComputeShader>       gSoftmaxCS, gVmulCS;
    // --proj mode extras
    ComPtr<ID3D11Buffer>              gWoBuf, gProjOut, gProjReadback;
    ComPtr<ID3D11ShaderResourceView>  gCtx_SRV, gWo_SRV;
    ComPtr<ID3D11UnorderedAccessView> gProj_UAV;
    ComPtr<ID3D11ComputeShader>       gProjCS;
    // --fold mode extras
    std::vector<float>                fold_packed;    // 16×4096 floats, pre-computed at setup
    ComPtr<ID3D11Buffer>              gPartialBuf, gResidualBuf, gFoldOut, gFoldReadback;
    ComPtr<ID3D11ShaderResourceView>  gPartial_SRV, gResidual_SRV;
    ComPtr<ID3D11UnorderedAccessView> gFold_UAV;
    ComPtr<ID3D11ComputeShader>       gFoldCS;

    if (use_shader) {
        // identity: same 256KB in → 256KB out
        gIn_SRV   = make_srv(dev, gQBuf.Get(), tile_elems);
        gIdentOut = make_buf(dev, (UINT)tile_bytes,
                             D3D11_USAGE_DEFAULT, D3D11_BIND_UNORDERED_ACCESS, 0);
        gIdent_UAV = make_uav(dev, gIdentOut.Get(), tile_elems);
        gCS = compile_cs(dev, kTileCopyHLSL, "tile_copy.hlsl");
        printf("[tile_test] identity shader compiled\n");
        printf("[tile_test] dispatch: %u groups × 256 threads\n\n", tile_elems/256);
    }
    else if (use_qk) {
        // Q×K: two 256KB inputs → 16KB output
        constexpr UINT OUT_ELEMS = 64 * 64;               // 4096 floats
        constexpr UINT OUT_BYTES = OUT_ELEMS * 4;          // 16384 bytes

        gQ_SRV = make_srv(dev, gQBuf.Get(), tile_elems);
        gKBuf  = make_buf(dev, (UINT)tile_bytes,
                          D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0);
        gK_SRV = make_srv(dev, gKBuf.Get(), tile_elems);

        gOutBuf = make_buf(dev, OUT_BYTES,
                           D3D11_USAGE_DEFAULT, D3D11_BIND_UNORDERED_ACCESS, 0);
        gOut_UAV = make_uav(dev, gOutBuf.Get(), OUT_ELEMS);

        gOutReadback = make_buf(dev, OUT_BYTES,
                                D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ);

        gCS = compile_cs(dev, kQKdotHLSL, "qk_tile.hlsl");
        printf("[tile_test] Q×K shader compiled\n");
        printf("[tile_test] dispatch: 8×8 groups × 8×8 threads = 64×64 outputs (4096 floats, 16KB)\n\n");
    }
    else if (use_softmax) {
        // softmax: same two 256KB inputs → 16KB probability output
        constexpr UINT OUT_ELEMS = 64 * 64;
        constexpr UINT OUT_BYTES = OUT_ELEMS * 4;

        gQ_SRV = make_srv(dev, gQBuf.Get(), tile_elems);
        gKBuf  = make_buf(dev, (UINT)tile_bytes,
                          D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0);
        gK_SRV = make_srv(dev, gKBuf.Get(), tile_elems);

        gOutBuf      = make_buf(dev, OUT_BYTES,
                                D3D11_USAGE_DEFAULT, D3D11_BIND_UNORDERED_ACCESS, 0);
        gOut_UAV     = make_uav(dev, gOutBuf.Get(), OUT_ELEMS);
        gOutReadback = make_buf(dev, OUT_BYTES,
                                D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ);

        gCS = compile_cs(dev, kSoftmaxHLSL, "qk_softmax.hlsl");
        printf("[tile_test] softmax shader compiled\n");
        printf("[tile_test] dispatch: 64 groups × 64 threads (one group per output row)\n\n");
    }
    else if (use_v) {
        // Full attention head: Q+K → softmax → P[64,64] → P×V → ctx[64,1024]
        constexpr UINT P_ELEMS   = 64 * 64;          // 4096 floats  (16KB)
        constexpr UINT CTX_ELEMS = 64 * 1024;        // 65536 floats (256KB = one tile)

        gQ_SRV = make_srv(dev, gQBuf.Get(), tile_elems);

        gKBuf  = make_buf(dev, (UINT)tile_bytes, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0);
        gK_SRV = make_srv(dev, gKBuf.Get(), tile_elems);

        // P = softmax output / vmul input
        gPBuf  = make_buf(dev, P_ELEMS * 4, D3D11_USAGE_DEFAULT,
                          D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS, 0);
        gP_SRV = make_srv(dev, gPBuf.Get(), P_ELEMS);
        gP_UAV = make_uav(dev, gPBuf.Get(), P_ELEMS);

        gVBuf  = make_buf(dev, (UINT)tile_bytes, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0);
        gV_SRV = make_srv(dev, gVBuf.Get(), tile_elems);

        gCtxBuf      = make_buf(dev, CTX_ELEMS * 4, D3D11_USAGE_DEFAULT, D3D11_BIND_UNORDERED_ACCESS, 0);
        gCtx_UAV     = make_uav(dev, gCtxBuf.Get(), CTX_ELEMS);
        gCtxReadback = make_buf(dev, CTX_ELEMS * 4, D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ);

        gSoftmaxCS = compile_cs(dev, kSoftmaxHLSL, "qk_softmax.hlsl");
        gVmulCS    = compile_cs(dev, kVmulHLSL,    "v_mul.hlsl");
        printf("[tile_test] softmax + V-mul shaders compiled\n");
        printf("[tile_test] softmax dispatch: 64×1×1 groups × 64 threads\n");
        printf("[tile_test] v-mul  dispatch: 8×128×1 groups × 8×8 threads → 64×1024 context\n\n");
    }
    else if (use_proj) {
        // O-projection: ctx[64,1024] × W_o[64,1024]^T → out[64,64]
        // ctx input = q_shard tile (reuses gQBuf as ctx — same [64,1024] layout)
        // W_o input = k_path shard (c_proj weight tile, same [64,1024] layout)
        constexpr UINT OUT_ELEMS = 64 * 64;   // 4096 floats = 16KB
        constexpr UINT OUT_BYTES = OUT_ELEMS * 4;

        gCtx_SRV = make_srv(dev, gQBuf.Get(), tile_elems);   // ctx reuses Q buffer

        gWoBuf  = make_buf(dev, (UINT)tile_bytes, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0);
        gWo_SRV = make_srv(dev, gWoBuf.Get(), tile_elems);

        gProjOut      = make_buf(dev, OUT_BYTES, D3D11_USAGE_DEFAULT, D3D11_BIND_UNORDERED_ACCESS, 0);
        gProj_UAV     = make_uav(dev, gProjOut.Get(), OUT_ELEMS);
        gProjReadback = make_buf(dev, OUT_BYTES, D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ);

        gProjCS = compile_cs(dev, kOProjHLSL, "o_proj.hlsl");
        printf("[tile_test] O-projection shader compiled\n");
        printf("[tile_test] dispatch: 8×8×1 groups × 8×8 threads → 64×64 outputs (4096 floats, 16KB)\n\n");
    }

    else if (use_fold) {
        // Pre-compute all 16 O-proj partials on CPU using Q tile 0 as context
        constexpr UINT PARTIAL_ELEMS = 16 * 64 * 64;   // 65536 floats = 256KB
        constexpr UINT PARTIAL_BYTES = PARTIAL_ELEMS * 4;
        constexpr UINT FOLD_ELEMS    = 64 * 1024;       // 65536 floats = 256KB
        constexpr UINT FOLD_BYTES    = FOLD_ELEMS * 4;

        if (!q_shard.read_tile(0, q_cpu.data())) die("read Q tile 0 for fold setup");
        fold_packed.resize(PARTIAL_ELEMS);
        printf("[tile_test] pre-computing 16 O-proj partials (CPU, Q tile 0 as ctx)...\n");
        for (uint32_t h = 0; h < 16; ++h) {
            if (!k_shard.read_tile(h, k_cpu.data())) die("read O tile for fold setup");
            cpu_proj((const float*)q_cpu.data(), (const float*)k_cpu.data(),
                     fold_packed.data() + h * 4096);
        }
        printf("[tile_test] partials done  (16 x 16KB = 256KB packed)\n");

        gPartialBuf   = make_buf(dev, PARTIAL_BYTES, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0);
        gResidualBuf  = make_buf(dev, FOLD_BYTES,    D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0);
        gFoldOut      = make_buf(dev, FOLD_BYTES,    D3D11_USAGE_DEFAULT, D3D11_BIND_UNORDERED_ACCESS, 0);
        gFoldReadback = make_buf(dev, FOLD_BYTES,    D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ);

        gPartial_SRV  = make_srv(dev, gPartialBuf.Get(),  PARTIAL_ELEMS);
        gResidual_SRV = make_srv(dev, gResidualBuf.Get(), FOLD_ELEMS);
        gFold_UAV     = make_uav(dev, gFoldOut.Get(),     FOLD_ELEMS);

        // Upload packed partials once — constant across all iterations
        ctx->UpdateSubresource(gPartialBuf.Get(), 0, nullptr, fold_packed.data(), 0, 0);

        gFoldCS = compile_cs(dev, kFoldHLSL, "layer_fold.hlsl");
        printf("[tile_test] LAYER_FOLD shader compiled\n");
        printf("[tile_test] dispatch: 256x1x1 groups x 256 threads -> 65536 outputs (64x1024, 256KB)\n\n");
    }

    // ── loop ─────────────────────────────────────────────────────────────────
    int mismatches = 0, device_losses = 0;
    float max_err = 0.f;
    const float TOL = 1e-2f;   // fp32 accumulation over 1024 elements

    for (int iter = 0; iter < ITERS; ++iter) {
        const uint32_t tidx = (uint32_t)(iter % tile_count);

        if (!q_shard.read_tile(tidx, q_cpu.data())) { fprintf(stderr, "read Q failed\n"); break; }

        ctx->UpdateSubresource(gQBuf.Get(), 0, nullptr, q_cpu.data(), 0, 0);

        ID3D11Buffer* readback_src = nullptr;
        size_t        cmp_bytes    = tile_bytes;

        if (!use_shader && !use_qk && !use_softmax && !use_v && !use_proj && !use_fold) {
            // upload-only: read back from gQBuf
            readback_src = gQBuf.Get();
        }
        else if (use_shader) {
            // identity CS
            ID3D11ShaderResourceView*  srv = gIn_SRV.Get();
            ID3D11UnorderedAccessView* uav = gIdent_UAV.Get();
            ctx->CSSetShader(gCS.Get(), nullptr, 0);
            ctx->CSSetShaderResources(0, 1, &srv);
            ctx->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
            ctx->Dispatch(tile_elems / 256, 1, 1);
            ID3D11UnorderedAccessView* nu = nullptr;
            ID3D11ShaderResourceView*  ns = nullptr;
            ctx->CSSetUnorderedAccessViews(0, 1, &nu, nullptr);
            ctx->CSSetShaderResources(0, 1, &ns);
            ctx->CSSetShader(nullptr, nullptr, 0);
            readback_src = gIdentOut.Get();
        }
        else if (use_qk) {
            // Q×K dot product
            if (!k_shard.read_tile(tidx, k_cpu.data())) { fprintf(stderr,"read K failed\n"); break; }
            ctx->UpdateSubresource(gKBuf.Get(), 0, nullptr, k_cpu.data(), 0, 0);

            ID3D11ShaderResourceView*  srvs[2] = { gQ_SRV.Get(), gK_SRV.Get() };
            ID3D11UnorderedAccessView* uav      = gOut_UAV.Get();
            ctx->CSSetShader(gCS.Get(), nullptr, 0);
            ctx->CSSetShaderResources(0, 2, srvs);
            ctx->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
            ctx->Dispatch(8, 8, 1);   // 8×8 groups × 8×8 threads = 64×64

            ID3D11UnorderedAccessView* nu    = nullptr;
            ID3D11ShaderResourceView*  ns[2] = { nullptr, nullptr };
            ctx->CSSetUnorderedAccessViews(0, 1, &nu, nullptr);
            ctx->CSSetShaderResources(0, 2, ns);
            ctx->CSSetShader(nullptr, nullptr, 0);

            readback_src = gOutBuf.Get();
            cmp_bytes    = 64 * 64 * 4;   // 16KB
        }
        else if (use_softmax) {
            // fused scale + softmax
            if (!k_shard.read_tile(tidx, k_cpu.data())) { fprintf(stderr,"read K failed\n"); break; }
            ctx->UpdateSubresource(gKBuf.Get(), 0, nullptr, k_cpu.data(), 0, 0);

            ID3D11ShaderResourceView*  srvs[2] = { gQ_SRV.Get(), gK_SRV.Get() };
            ID3D11UnorderedAccessView* uav      = gOut_UAV.Get();
            ctx->CSSetShader(gCS.Get(), nullptr, 0);
            ctx->CSSetShaderResources(0, 2, srvs);
            ctx->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
            ctx->Dispatch(64, 1, 1);   // 64 groups × 64 threads — one group per row

            ID3D11UnorderedAccessView* nu    = nullptr;
            ID3D11ShaderResourceView*  ns[2] = { nullptr, nullptr };
            ctx->CSSetUnorderedAccessViews(0, 1, &nu, nullptr);
            ctx->CSSetShaderResources(0, 2, ns);
            ctx->CSSetShader(nullptr, nullptr, 0);

            readback_src = gOutBuf.Get();
            cmp_bytes    = 64 * 64 * 4;
        }
        else if (use_v) {
            // Pass 1: softmax(Q, K) → gPBuf
            if (!k_shard.read_tile(tidx, k_cpu.data())) { fprintf(stderr,"read K failed\n"); break; }
            ctx->UpdateSubresource(gKBuf.Get(), 0, nullptr, k_cpu.data(), 0, 0);
            {
                ID3D11ShaderResourceView*  srvs[2] = { gQ_SRV.Get(), gK_SRV.Get() };
                ID3D11UnorderedAccessView* uav      = gP_UAV.Get();
                ctx->CSSetShader(gSoftmaxCS.Get(), nullptr, 0);
                ctx->CSSetShaderResources(0, 2, srvs);
                ctx->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
                ctx->Dispatch(64, 1, 1);
                ID3D11UnorderedAccessView* nu    = nullptr;
                ID3D11ShaderResourceView*  ns[2] = { nullptr, nullptr };
                ctx->CSSetUnorderedAccessViews(0, 1, &nu, nullptr);
                ctx->CSSetShaderResources(0, 2, ns);
                ctx->CSSetShader(nullptr, nullptr, 0);
            }
            // Pass 2: P × V → gCtxBuf
            if (!v_shard.read_tile(tidx, v_cpu.data())) { fprintf(stderr,"read V failed\n"); break; }
            ctx->UpdateSubresource(gVBuf.Get(), 0, nullptr, v_cpu.data(), 0, 0);
            {
                ID3D11ShaderResourceView*  srvs[2] = { gP_SRV.Get(), gV_SRV.Get() };
                ID3D11UnorderedAccessView* uav      = gCtx_UAV.Get();
                ctx->CSSetShader(gVmulCS.Get(), nullptr, 0);
                ctx->CSSetShaderResources(0, 2, srvs);
                ctx->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
                ctx->Dispatch(8, 128, 1);   // 64 rows × 1024 cols
                ID3D11UnorderedAccessView* nu    = nullptr;
                ID3D11ShaderResourceView*  ns[2] = { nullptr, nullptr };
                ctx->CSSetUnorderedAccessViews(0, 1, &nu, nullptr);
                ctx->CSSetShaderResources(0, 2, ns);
                ctx->CSSetShader(nullptr, nullptr, 0);
            }
            readback_src = gCtxBuf.Get();
            cmp_bytes    = 64 * 1024 * 4;   // 256KB
        }
        else if (use_proj) {
            // O-projection: ctx (gQBuf) × W_o (gWoBuf) → out[64,64]
            if (!k_shard.read_tile(tidx, k_cpu.data())) { fprintf(stderr,"read W_o failed\n"); break; }
            ctx->UpdateSubresource(gWoBuf.Get(), 0, nullptr, k_cpu.data(), 0, 0);

            ID3D11ShaderResourceView*  srvs[2] = { gCtx_SRV.Get(), gWo_SRV.Get() };
            ID3D11UnorderedAccessView* uav      = gProj_UAV.Get();
            ctx->CSSetShader(gProjCS.Get(), nullptr, 0);
            ctx->CSSetShaderResources(0, 2, srvs);
            ctx->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
            ctx->Dispatch(8, 8, 1);   // 64×64 outputs

            ID3D11UnorderedAccessView* nu    = nullptr;
            ID3D11ShaderResourceView*  ns[2] = { nullptr, nullptr };
            ctx->CSSetUnorderedAccessViews(0, 1, &nu, nullptr);
            ctx->CSSetShaderResources(0, 2, ns);
            ctx->CSSetShader(nullptr, nullptr, 0);

            readback_src = gProjOut.Get();
            cmp_bytes    = 64 * 64 * 4;   // 16KB
        }
        else if (use_fold) {
            // Residual comes from Q shard (varies each iter, gives different data paths)
            ctx->UpdateSubresource(gResidualBuf.Get(), 0, nullptr, q_cpu.data(), 0, 0);

            ID3D11ShaderResourceView*  srvs[2] = { gPartial_SRV.Get(), gResidual_SRV.Get() };
            ID3D11UnorderedAccessView* uav      = gFold_UAV.Get();
            ctx->CSSetShader(gFoldCS.Get(), nullptr, 0);
            ctx->CSSetShaderResources(0, 2, srvs);
            ctx->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
            ctx->Dispatch(256, 1, 1);   // 256 groups × 256 threads = 65536

            ID3D11UnorderedAccessView* nu    = nullptr;
            ID3D11ShaderResourceView*  ns[2] = { nullptr, nullptr };
            ctx->CSSetUnorderedAccessViews(0, 1, &nu, nullptr);
            ctx->CSSetShaderResources(0, 2, ns);
            ctx->CSSetShader(nullptr, nullptr, 0);

            readback_src = gFoldOut.Get();
            cmp_bytes    = 64 * 1024 * 4;  // 256KB
        }

        // readback (Map stalls CPU until GPU done — reliable sync on Intel iGPU)
        ID3D11Buffer* staging = use_fold                        ? gFoldReadback.Get()
                             : use_v                          ? gCtxReadback.Get()
                             : use_proj                       ? gProjReadback.Get()
                             : (use_qk || use_softmax)        ? gOutReadback.Get()
                             :                                  gReadback.Get();
        ctx->CopyResource(staging, readback_src);
        D3D11_MAPPED_SUBRESOURCE mr{};
        HRESULT hr = ctx->Map(staging, 0, D3D11_MAP_READ, 0, &mr);
        if (FAILED(hr)) {
            fprintf(stderr, "[tile_test] Map failed iter=%d hr=0x%08X device=0x%08X\n",
                    iter, (unsigned)hr, (unsigned)dev->GetDeviceRemovedReason());
            ++device_losses; break;
        }

        bool mismatch = false;
        if (use_qk) {
            // Float tolerance comparison against CPU reference
            std::vector<float> ref(64*64);
            cpu_qk((const float*)q_cpu.data(), (const float*)k_cpu.data(), ref.data());
            const float* gpu = (const float*)mr.pData;
            for (int e = 0; e < 64*64; ++e) {
                float err = fabsf(gpu[e] - ref[e]);
                if (err > max_err) max_err = err;
                if (err > TOL) {
                    if (!mismatch)
                        fprintf(stderr, "[tile_test] MISMATCH iter=%d tile=%u  "
                                "e=%d gpu=%.4f ref=%.4f err=%.2e\n",
                                iter, tidx, e, gpu[e], ref[e], err);
                    mismatch = true;
                    break;
                }
            }
        } else if (use_v) {
            // CPU reference: softmax(Q,K) → P, then P×V → ctx
            std::vector<float> P_ref(64*64);
            cpu_softmax((const float*)q_cpu.data(), (const float*)k_cpu.data(), P_ref.data());
            std::vector<float> ref(64*1024);
            cpu_v_mul(P_ref.data(), (const float*)v_cpu.data(), ref.data());
            const float* gpu = (const float*)mr.pData;
            const float V_TOL = 1e-3f;
            for (int e = 0; e < 64*1024; ++e) {
                float err = fabsf(gpu[e] - ref[e]);
                if (err > max_err) max_err = err;
                if (err > V_TOL) {
                    if (!mismatch)
                        fprintf(stderr, "[tile_test] MISMATCH iter=%d tile=%u "
                                "e=%d gpu=%.6f ref=%.6f err=%.2e\n",
                                iter, tidx, e, gpu[e], ref[e], err);
                    mismatch = true; break;
                }
            }
        } else if (use_softmax) {
            // Compare vs CPU softmax + verify row sums ≈ 1.0
            std::vector<float> ref(64*64);
            cpu_softmax((const float*)q_cpu.data(), (const float*)k_cpu.data(), ref.data());
            const float* gpu = (const float*)mr.pData;
            const float SOFTMAX_TOL = 1e-4f;
            const float SUM_TOL    = 1e-5f;
            for (int r = 0; r < 64 && !mismatch; ++r) {
                float row_sum = 0.f;
                for (int c = 0; c < 64; ++c) {
                    int e = r*64+c;
                    float err = fabsf(gpu[e] - ref[e]);
                    if (err > max_err) max_err = err;
                    if (err > SOFTMAX_TOL) {
                        fprintf(stderr, "[tile_test] MISMATCH iter=%d tile=%u "
                                "r=%d c=%d gpu=%.6f ref=%.6f err=%.2e\n",
                                iter, tidx, r, c, gpu[e], ref[e], err);
                        mismatch = true; break;
                    }
                    row_sum += gpu[e];
                }
                if (!mismatch && fabsf(row_sum - 1.f) > SUM_TOL) {
                    fprintf(stderr, "[tile_test] ROW SUM FAIL iter=%d tile=%u "
                            "r=%d sum=%.8f\n", iter, tidx, r, row_sum);
                    mismatch = true;
                }
            }
        } else if (use_proj) {
            // O-projection: cpu_proj(ctx=q_cpu, Wo=k_cpu) → ref[64,64]
            std::vector<float> ref(64*64);
            cpu_proj((const float*)q_cpu.data(), (const float*)k_cpu.data(), ref.data());
            const float* gpu = (const float*)mr.pData;
            for (int e = 0; e < 64*64; ++e) {
                float err = fabsf(gpu[e] - ref[e]);
                if (err > max_err) max_err = err;
                if (err > TOL) {
                    if (!mismatch)
                        fprintf(stderr, "[tile_test] MISMATCH iter=%d tile=%u "
                                "e=%d gpu=%.4f ref=%.4f err=%.2e\n",
                                iter, tidx, e, gpu[e], ref[e], err);
                    mismatch = true; break;
                }
            }
        } else if (use_fold) {
            // LAYER_FOLD: cpu_fold(packed_partials, q_cpu_as_residual) → ref[64,1024]
            std::vector<float> ref(64*1024);
            cpu_fold(fold_packed.data(), (const float*)q_cpu.data(), ref.data());
            const float* gpu = (const float*)mr.pData;
            for (int e = 0; e < 64*1024; ++e) {
                float err = fabsf(gpu[e] - ref[e]);
                if (err > max_err) max_err = err;
                if (err > TOL) {
                    if (!mismatch)
                        fprintf(stderr, "[tile_test] MISMATCH iter=%d tile=%u "
                                "e=%d gpu=%.4f ref=%.4f err=%.2e\n",
                                iter, tidx, e, gpu[e], ref[e], err);
                    mismatch = true; break;
                }
            }
        } else {
            // Bit-exact comparison
            mismatch = (memcmp(q_cpu.data(), mr.pData, cmp_bytes) != 0);
            if (mismatch) {
                const float* a = (const float*)q_cpu.data();
                const float* b = (const float*)mr.pData;
                for (uint32_t e = 0; e < tile_elems; ++e)
                    if (a[e] != b[e]) {
                        fprintf(stderr, "[tile_test] MISMATCH iter=%d tile=%u "
                                "e=%u cpu=%.6f gpu=%.6f\n", iter, tidx, e, a[e], b[e]);
                        break;
                    }
            }
        }
        ctx->Unmap(staging, 0);

        if (mismatch) { ++mismatches; if (mismatches >= 3) { fprintf(stderr,"(stopping)\n"); break; } }

        if ((iter+1) % 1000 == 0) {
            HRESULT dr = dev->GetDeviceRemovedReason();
            if (use_qk || use_softmax || use_v || use_proj || use_fold)
                printf("[tile_test] iter %5d/%d  tile=%u  device=%s  max_err=%.2e\n",
                       iter+1, ITERS, tidx, (dr==S_OK)?"OK":"LOST", max_err);
            else
                printf("[tile_test] iter %5d/%d  tile=%u  device=%s\n",
                       iter+1, ITERS, tidx, (dr==S_OK)?"OK":"LOST");
            if (dr != S_OK) { ++device_losses; break; }
        }
    }

    printf("\n");
    if (mismatches == 0 && device_losses == 0) {
        if (use_fold)
            printf("[tile_test] PASS — %d iterations, 0 mismatches, max_err=%.2e (tol=%.2e), LAYER_FOLD stable\n",
                   ITERS, max_err, TOL);
        else if (use_proj)
            printf("[tile_test] PASS — %d iterations, 0 mismatches, max_err=%.2e (tol=%.2e), O-proj stable\n",
                   ITERS, max_err, TOL);
        else if (use_v)
            printf("[tile_test] PASS — %d iterations, 0 mismatches, max_err=%.2e (tol=1e-3), full attn head stable\n",
                   ITERS, max_err);
        else if (use_softmax)
            printf("[tile_test] PASS — %d iterations, 0 mismatches, max_err=%.2e (tol=1e-4), row sums OK, device stable\n",
                   ITERS, max_err);
        else if (use_qk)
            printf("[tile_test] PASS — %d iterations, 0 mismatches, max_err=%.2e (tol=%.2e), device stable\n",
                   ITERS, max_err, TOL);
        else
            printf("[tile_test] PASS — %d iterations, 0 mismatches, device stable\n", ITERS);
        return 0;
    }
    printf("[tile_test] FAIL — mismatches=%d device_losses=%d\n", mismatches, device_losses);
    return 1;
}
