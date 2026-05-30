// json_runtime_test.cpp
// JSON Runtime — opcode executor for the ASX streaming engine
//
// Three-plane architecture:
//   [JSON Intent]  .jrun program file  → declarative list of ops
//   [ASX RAM]      JRunExecutor        → slot scheduling, residency
//   [GPU]          GPUWindow           → D3D11 compute dispatch
//
// Supported opcodes:
//   ATTN_HEAD  layer head q k v  — full head: load→softmax→vmul→verify
//   (future)   LOAD_SHARD, UPLOAD_TILE, DISPATCH, READBACK  — primitives
//
// Usage:
//   json_runtime_test.exe <program.jrun>
//   json_runtime_test.exe --gen <out.jrun> <q_shard> <k_shard> <v_shard> [n_heads]

#include "d3d11_engine.h"
#include "xshard.h"

#include <d3dcompiler.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>

using Clock = std::chrono::steady_clock;
static float elapsed_ms(Clock::time_point t0) {
    return std::chrono::duration<float,std::milli>(Clock::now()-t0).count();
}

// ── embedded shaders (proven in --v and asx_ram_test) ─────────────────────────

static const char* kSoftmaxHLSL = R"hlsl(
Buffer<float>   Q   : register(t0);
Buffer<float>   K   : register(t1);
RWBuffer<float> Out : register(u0);
#define D_MODEL 1024u
#define SEQ     64u
#define SCALE   0.03125f
groupshared float sdata[SEQ];
[numthreads(64,1,1)]
void main(uint3 gid:SV_GroupID, uint3 tid:SV_GroupThreadID) {
    uint row=gid.x, col=tid.x;
    float s=0.f;
    for(uint i=0u;i<D_MODEL;i++) s+=Q[row*D_MODEL+i]*K[col*D_MODEL+i];
    s*=SCALE; sdata[col]=s; GroupMemoryBarrierWithGroupSync();
    for(uint stride=32u;stride>0u;stride>>=1u){
        if(col<stride) sdata[col]=max(sdata[col],sdata[col+stride]);
        GroupMemoryBarrierWithGroupSync();
    }
    float m=sdata[0]; GroupMemoryBarrierWithGroupSync();
    float e=exp(s-m); sdata[col]=e; GroupMemoryBarrierWithGroupSync();
    for(uint stride=32u;stride>0u;stride>>=1u){
        if(col<stride) sdata[col]+=sdata[col+stride];
        GroupMemoryBarrierWithGroupSync();
    }
    Out[row*SEQ+col]=e/sdata[0];
}
)hlsl";

static const char* kVmulHLSL = R"hlsl(
Buffer<float>   P   : register(t0);
Buffer<float>   V   : register(t1);
RWBuffer<float> Out : register(u0);
#define SEQ     64u
#define D_MODEL 1024u
[numthreads(8,8,1)]
void main(uint3 id:SV_DispatchThreadID){
    uint row=id.x, col=id.y;
    if(row>=SEQ||col>=D_MODEL) return;
    float s=0.f;
    for(uint k=0u;k<SEQ;k++) s+=P[row*SEQ+k]*V[k*D_MODEL+col];
    Out[row*D_MODEL+col]=s;
}
)hlsl";

// ── D3D11 helpers ─────────────────────────────────────────────────────────────

static void die(const char* m){fprintf(stderr,"[jrt] FATAL: %s\n",m);exit(1);}
static void chk(HRESULT hr,const char* l){
    if(FAILED(hr)){fprintf(stderr,"[jrt] %s hr=0x%08X\n",l,(unsigned)hr);exit(1);}
}
static ComPtr<ID3D11Buffer> mk_buf(ID3D11Device* dev,UINT bytes,D3D11_USAGE u,UINT b,UINT c){
    D3D11_BUFFER_DESC d{};d.ByteWidth=bytes;d.Usage=u;d.BindFlags=b;d.CPUAccessFlags=c;
    ComPtr<ID3D11Buffer> r;chk(dev->CreateBuffer(&d,nullptr,&r),"CreateBuffer");return r;
}
static ComPtr<ID3D11ShaderResourceView> mk_srv(ID3D11Device* dev,ID3D11Buffer* buf,UINT n){
    D3D11_SHADER_RESOURCE_VIEW_DESC d{};d.Format=DXGI_FORMAT_R32_FLOAT;
    d.ViewDimension=D3D11_SRV_DIMENSION_BUFFER;d.Buffer.NumElements=n;
    ComPtr<ID3D11ShaderResourceView> v;chk(dev->CreateShaderResourceView(buf,&d,&v),"SRV");return v;
}
static ComPtr<ID3D11UnorderedAccessView> mk_uav(ID3D11Device* dev,ID3D11Buffer* buf,UINT n){
    D3D11_UNORDERED_ACCESS_VIEW_DESC d{};d.Format=DXGI_FORMAT_R32_FLOAT;
    d.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;d.Buffer.NumElements=n;
    ComPtr<ID3D11UnorderedAccessView> v;chk(dev->CreateUnorderedAccessView(buf,&d,&v),"UAV");return v;
}
static ComPtr<ID3D11ComputeShader> mk_cs(ID3D11Device* dev,const char* src,const char* name){
    ComPtr<ID3DBlob> blob,errs;
    HRESULT hr=D3DCompile(src,strlen(src),name,nullptr,nullptr,"main","cs_5_0",
                          D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&blob,&errs);
    if(FAILED(hr)){if(errs)fprintf(stderr,"%s\n",(char*)errs->GetBufferPointer());die("D3DCompile");}
    ComPtr<ID3D11ComputeShader> cs;
    chk(dev->CreateComputeShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&cs),"CreateCS");
    return cs;
}

// ── GPU window ─────────────────────────────────────────────────────────────────

struct GPUWin {
    ComPtr<ID3D11Buffer>              qBuf,kBuf,vBuf,pBuf,ctxBuf,ctxStg;
    ComPtr<ID3D11ShaderResourceView>  q_sv,k_sv,p_sv,v_sv;
    ComPtr<ID3D11UnorderedAccessView> p_uv,ctx_uv;
    ComPtr<ID3D11ComputeShader>       smx_cs,vmul_cs;

    void init(ID3D11Device* dev){
        qBuf  =mk_buf(dev,65536*4,D3D11_USAGE_DEFAULT,D3D11_BIND_SHADER_RESOURCE,0);
        kBuf  =mk_buf(dev,65536*4,D3D11_USAGE_DEFAULT,D3D11_BIND_SHADER_RESOURCE,0);
        vBuf  =mk_buf(dev,65536*4,D3D11_USAGE_DEFAULT,D3D11_BIND_SHADER_RESOURCE,0);
        pBuf  =mk_buf(dev,4096*4, D3D11_USAGE_DEFAULT,
                      D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS,0);
        ctxBuf=mk_buf(dev,65536*4,D3D11_USAGE_DEFAULT,D3D11_BIND_UNORDERED_ACCESS,0);
        ctxStg=mk_buf(dev,65536*4,D3D11_USAGE_STAGING,0,D3D11_CPU_ACCESS_READ);
        q_sv=mk_srv(dev,qBuf.Get(),65536); k_sv=mk_srv(dev,kBuf.Get(),65536);
        v_sv=mk_srv(dev,vBuf.Get(),65536); p_sv=mk_srv(dev,pBuf.Get(),4096);
        p_uv=mk_uav(dev,pBuf.Get(),4096); ctx_uv=mk_uav(dev,ctxBuf.Get(),65536);
        smx_cs =mk_cs(dev,kSoftmaxHLSL,"jrt_softmax.hlsl");
        vmul_cs=mk_cs(dev,kVmulHLSL,   "jrt_vmul.hlsl");
    }

    void upload(ID3D11DeviceContext* ctx,const float* q,const float* k,const float* v){
        ctx->UpdateSubresource(qBuf.Get(),0,nullptr,q,0,0);
        ctx->UpdateSubresource(kBuf.Get(),0,nullptr,k,0,0);
        ctx->UpdateSubresource(vBuf.Get(),0,nullptr,v,0,0);
    }

    void dispatch(ID3D11DeviceContext* ctx){
        // softmax(Q,K) → pBuf
        {ID3D11ShaderResourceView* s[2]={q_sv.Get(),k_sv.Get()};
         ID3D11UnorderedAccessView* u=p_uv.Get();
         ctx->CSSetShader(smx_cs.Get(),nullptr,0);
         ctx->CSSetShaderResources(0,2,s);
         ctx->CSSetUnorderedAccessViews(0,1,&u,nullptr);
         ctx->Dispatch(64,1,1);
         ID3D11UnorderedAccessView* nu=nullptr;ID3D11ShaderResourceView* ns[2]={nullptr,nullptr};
         ctx->CSSetUnorderedAccessViews(0,1,&nu,nullptr);
         ctx->CSSetShaderResources(0,2,ns);
         ctx->CSSetShader(nullptr,nullptr,0);}
        // vmul(P,V) → ctxBuf
        {ID3D11ShaderResourceView* s[2]={p_sv.Get(),v_sv.Get()};
         ID3D11UnorderedAccessView* u=ctx_uv.Get();
         ctx->CSSetShader(vmul_cs.Get(),nullptr,0);
         ctx->CSSetShaderResources(0,2,s);
         ctx->CSSetUnorderedAccessViews(0,1,&u,nullptr);
         ctx->Dispatch(8,128,1);
         ID3D11UnorderedAccessView* nu=nullptr;ID3D11ShaderResourceView* ns[2]={nullptr,nullptr};
         ctx->CSSetUnorderedAccessViews(0,1,&nu,nullptr);
         ctx->CSSetShaderResources(0,2,ns);
         ctx->CSSetShader(nullptr,nullptr,0);}
    }

    // returns false on device loss; out must be float[65536]
    bool readback(ID3D11DeviceContext* ctx, float* out){
        ctx->CopyResource(ctxStg.Get(),ctxBuf.Get());
        D3D11_MAPPED_SUBRESOURCE mr{};
        HRESULT hr=ctx->Map(ctxStg.Get(),0,D3D11_MAP_READ,0,&mr);
        if(FAILED(hr)) return false;
        memcpy(out,mr.pData,65536*4);
        ctx->Unmap(ctxStg.Get(),0);
        return true;
    }
};

// ── xshard loader ─────────────────────────────────────────────────────────────

struct Shard {
    FILE* fp=nullptr; XShardHeader hdr{};
    bool open(const char* p){
        fp=fopen(p,"rb");
        if(!fp){fprintf(stderr,"[jrt] cannot open shard: %s\n",p);return false;}
        return fread(&hdr,1,64,fp)==64 && xshard_valid_magic(hdr);
    }
    bool read(uint32_t tile, float* buf){
        uint64_t off=xshard_tile_offset(hdr,tile);
        if(fseek(fp,(long)off,SEEK_SET)!=0) return false;
        return fread(buf,4,(size_t)hdr.tile_size,fp)==hdr.tile_size;
    }
    ~Shard(){if(fp)fclose(fp);}
};

// ── CPU reference ─────────────────────────────────────────────────────────────

static void cpu_attn(const float* Q,const float* K,const float* V,float* out){
    float P[64*64];
    for(int r=0;r<64;++r){
        float s[64],mx=-1e30f;
        for(int c=0;c<64;++c){
            float d=0.f;
            for(int i=0;i<1024;++i) d+=Q[r*1024+i]*K[c*1024+i];
            s[c]=d*0.03125f; if(s[c]>mx) mx=s[c];
        }
        float sum=0.f;
        for(int c=0;c<64;++c){s[c]=expf(s[c]-mx);sum+=s[c];}
        for(int c=0;c<64;++c) P[r*64+c]=s[c]/sum;
    }
    for(int r=0;r<64;++r)
        for(int c=0;c<1024;++c){
            float d=0.f;
            for(int k=0;k<64;++k) d+=P[r*64+k]*V[k*1024+c];
            out[r*1024+c]=d;
        }
}

// ── minimal JSON parser ───────────────────────────────────────────────────────
// Handles our specific .jrun schema. Not general-purpose.

static std::string jstr(const std::string& obj, const char* key) {
    std::string pat = std::string("\"") + key + "\"";
    size_t p = obj.find(pat);
    if (p == std::string::npos) return "";
    p += pat.size();
    while (p < obj.size() && (obj[p]==' '||obj[p]=='\t'||obj[p]=='\r'||obj[p]=='\n'||obj[p]==':')) ++p;
    if (p >= obj.size() || obj[p] != '"') return "";
    ++p;
    std::string r;
    while (p < obj.size() && obj[p] != '"') {
        if (obj[p]=='\\' && p+1<obj.size()) {
            ++p;
            if (obj[p]=='\\') r+='\\';
            else if(obj[p]=='"') r+='"';
            else r+=obj[p];
        } else { r+=obj[p]; }
        ++p;
    }
    return r;
}

static int jint(const std::string& obj, const char* key) {
    std::string pat = std::string("\"") + key + "\"";
    size_t p = obj.find(pat);
    if (p == std::string::npos) return -1;
    p += pat.size();
    while (p < obj.size() && (obj[p]==' '||obj[p]=='\t'||obj[p]==':')) ++p;
    if (p >= obj.size()) return -1;
    return atoi(obj.c_str()+p);
}

// Split "program": [ ... ] array into individual object strings
static std::vector<std::string> parse_program(const std::string& json) {
    size_t arr = json.find("\"program\"");
    if (arr == std::string::npos) { fprintf(stderr,"[jrt] no 'program' key\n"); return {}; }
    arr = json.find('[', arr);
    if (arr == std::string::npos) { fprintf(stderr,"[jrt] no '[' after program\n"); return {}; }
    ++arr;
    std::vector<std::string> objs;
    int depth = 0;
    size_t start = std::string::npos;
    bool in_str = false;
    for (size_t i = arr; i < json.size(); ++i) {
        char c = json[i];
        if (c == '\\' && in_str) { ++i; continue; }
        if (c == '"') { in_str = !in_str; continue; }
        if (in_str) continue;
        if (c == '{') { if (depth++ == 0) start = i; }
        else if (c == '}') {
            if (--depth == 0 && start != std::string::npos) {
                objs.push_back(json.substr(start, i-start+1));
                start = std::string::npos;
            }
        } else if (c == ']' && depth == 0) break;
    }
    return objs;
}

static std::string read_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr,"[jrt] cannot open program: %s\n", path); return ""; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); rewind(f);
    std::string s(sz, '\0');
    fread(&s[0], 1, sz, f); fclose(f);
    return s;
}

// ── opcodes ───────────────────────────────────────────────────────────────────

enum class JOp { ATTN_HEAD, UNKNOWN };

struct JInstr {
    JOp         op      = JOp::UNKNOWN;
    uint32_t    layer   = 0;
    uint32_t    head    = 0;
    std::string q, k, v;
};

static JInstr parse_instr(const std::string& obj) {
    JInstr instr;
    std::string op = jstr(obj, "op");
    if (op == "ATTN_HEAD") instr.op = JOp::ATTN_HEAD;
    instr.layer = (uint32_t)jint(obj, "layer");
    instr.head  = (uint32_t)jint(obj, "head");
    instr.q     = jstr(obj, "q");
    instr.k     = jstr(obj, "k");
    instr.v     = jstr(obj, "v");
    return instr;
}

// ── --gen mode: write a .jrun for N heads of one layer ────────────────────────

static void gen_program(const char* out_path, const char* q, const char* k, const char* v,
                         uint32_t layer, uint32_t n_heads) {
    FILE* f = fopen(out_path, "w");
    if (!f) { fprintf(stderr,"[jrt] cannot write %s\n", out_path); exit(1); }
    fprintf(f, "{\n  \"name\": \"layer_%02u_attn\",\n", layer);
    fprintf(f, "  \"description\": \"GPT-2 medium layer %u attention — %u heads\",\n", layer, n_heads);
    fprintf(f, "  \"layer\": %u,\n  \"n_heads\": %u,\n", layer, n_heads);
    fprintf(f, "  \"program\": [\n");
    for (uint32_t h = 0; h < n_heads; ++h) {
        fprintf(f, "    {\"op\":\"ATTN_HEAD\",\"layer\":%u,\"head\":%u,"
                   "\"q\":\"%s\",\"k\":\"%s\",\"v\":\"%s\"}%s\n",
                layer, h, q, k, v, (h+1<n_heads)?",":"");
    }
    fprintf(f, "  ]\n}\n");
    fclose(f);
    printf("[jrt] wrote program: %s  (%u ATTN_HEAD ops)\n", out_path, n_heads);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    // ── --gen mode ────────────────────────────────────────────────────────────
    if (argc >= 2 && strcmp(argv[1],"--gen")==0) {
        if (argc < 6) {
            fprintf(stderr,"Usage: --gen <out.jrun> <q_shard> <k_shard> <v_shard> [layer] [n_heads]\n");
            return 1;
        }
        uint32_t layer  = argc>6 ? (uint32_t)atoi(argv[6]) : 0;
        uint32_t nheads = argc>7 ? (uint32_t)atoi(argv[7]) : 16;
        gen_program(argv[2], argv[3], argv[4], argv[5], layer, nheads);
        return 0;
    }

    // ── execute mode ──────────────────────────────────────────────────────────
    if (argc < 2) {
        fprintf(stderr,"Usage: json_runtime_test.exe <program.jrun>\n"
                       "       json_runtime_test.exe --gen <out.jrun> <q> <k> <v> [layer] [heads]\n");
        return 1;
    }

    const char* prog_path = argv[1];
    printf("[jrt] program : %s\n", prog_path);

    std::string json = read_file(prog_path);
    if (json.empty()) return 1;

    std::string prog_name = jstr(json, "name");
    printf("[jrt] name    : %s\n\n", prog_name.c_str());

    std::vector<std::string> obj_strs = parse_program(json);
    printf("[jrt] parsed %zu instruction(s)\n\n", obj_strs.size());

    std::vector<JInstr> program;
    for (auto& s : obj_strs) program.push_back(parse_instr(s));

    // ── D3D11 init ────────────────────────────────────────────────────────────
    D3D11Engine eng;
    if (!eng.init(false, true)) die("D3D11 init");
    ID3D11Device*        dev = eng.rawDevice();
    ID3D11DeviceContext* ctx = eng.rawCtx();
    printf("[jrt] adapter : %s\n\n", eng.adapterName().c_str());

    GPUWin gpu;
    gpu.init(dev);
    printf("[jrt] GPU window initialized\n");
    printf("[jrt] softmax dispatch : Dispatch(64,1,1)\n");
    printf("[jrt] vmul    dispatch : Dispatch(8,128,1)\n\n");

    // ── tile CPU buffers ──────────────────────────────────────────────────────
    std::vector<float> q_buf(65536), k_buf(65536), v_buf(65536), ctx_gpu(65536), ctx_ref(65536);

    // Shard cache: avoid reopening the same file repeatedly
    std::string open_q, open_k, open_v;
    Shard sq, sk, sv;

    // ── execution loop ────────────────────────────────────────────────────────
    int   n_ok=0, n_fail=0, n_dev_loss=0;
    float global_max_err=0.f, total_disk_ms=0.f, total_gpu_ms=0.f;
    const float TOL = 1e-3f;

    printf("[jrt] ── execution trace ─────────────────────────────────────────────\n");
    printf("[jrt]  #   op          L  H   disk_ms  gpu_ms   max_err  status\n");
    printf("[jrt] ─────────────────────────────────────────────────────────────────\n");

    for (size_t i = 0; i < program.size(); ++i) {
        const JInstr& instr = program[i];

        if (instr.op == JOp::UNKNOWN) {
            printf("[jrt] %3zu  UNKNOWN — skipped\n", i);
            continue;
        }

        // ── ATTN_HEAD ─────────────────────────────────────────────────────────
        // Slot C: load Q/K/V tile[head] from shards
        if (instr.q != open_q) { sq.~Shard(); new(&sq)Shard(); if(!sq.open(instr.q.c_str()))die("open Q"); open_q=instr.q; }
        if (instr.k != open_k) { sk.~Shard(); new(&sk)Shard(); if(!sk.open(instr.k.c_str()))die("open K"); open_k=instr.k; }
        if (instr.v != open_v) { sv.~Shard(); new(&sv)Shard(); if(!sv.open(instr.v.c_str()))die("open V"); open_v=instr.v; }

        auto t_disk = Clock::now();
        if (!sq.read(instr.head, q_buf.data())) die("read Q tile");
        if (!sk.read(instr.head, k_buf.data())) die("read K tile");
        if (!sv.read(instr.head, v_buf.data())) die("read V tile");
        float disk_ms = elapsed_ms(t_disk);

        // Slot B→A: upload + dispatch + readback
        auto t_gpu = Clock::now();
        gpu.upload(ctx, q_buf.data(), k_buf.data(), v_buf.data());
        gpu.dispatch(ctx);
        bool ok = gpu.readback(ctx, ctx_gpu.data());
        float gpu_ms = elapsed_ms(t_gpu);

        if (!ok) {
            fprintf(stderr,"[jrt] device lost at instr %zu\n", i);
            ++n_dev_loss; break;
        }

        // Verify vs CPU reference
        cpu_attn(q_buf.data(), k_buf.data(), v_buf.data(), ctx_ref.data());
        float head_err=0.f; bool mismatch=false;
        for (int e=0;e<65536;++e){
            float err=fabsf(ctx_gpu[e]-ctx_ref[e]);
            if(err>head_err) head_err=err;
            if(err>TOL){mismatch=true;break;}
        }
        if (head_err>global_max_err) global_max_err=head_err;

        total_disk_ms += disk_ms;
        total_gpu_ms  += gpu_ms;
        if (mismatch) ++n_fail; else ++n_ok;

        printf("[jrt] %3zu  ATTN_HEAD   %u  %2u  %6.2f   %6.2f  %.2e  %s\n",
               i, instr.layer, instr.head, disk_ms, gpu_ms, head_err,
               mismatch ? "FAIL" : "OK");
    }

    // ── summary ───────────────────────────────────────────────────────────────
    printf("[jrt] ─────────────────────────────────────────────────────────────────\n\n");

    int total = n_ok + n_fail;
    float thr  = total > 0 ? (float)total / ((total_disk_ms+total_gpu_ms)*0.001f) : 0.f;
    float mbps = thr * 3.f * 0.25f;   // 3 × 256KB tiles per head

    printf("[jrt] ops executed : %d\n",   total);
    printf("[jrt] passed       : %d\n",   n_ok);
    printf("[jrt] failed       : %d\n",   n_fail);
    printf("[jrt] device losses: %d\n",   n_dev_loss);
    printf("[jrt] disk time    : %.1f ms  (%.2f ms/op)\n", total_disk_ms, total?total_disk_ms/total:0.f);
    printf("[jrt] gpu  time    : %.1f ms  (%.2f ms/op)\n", total_gpu_ms,  total?total_gpu_ms/total:0.f);
    printf("[jrt] throughput   : %.1f heads/sec  (%.1f MB/sec)\n", thr, mbps);
    printf("[jrt] global err   : %.2e  (tol=%.0e)\n\n", global_max_err, TOL);

    if (n_fail==0 && n_dev_loss==0)
        printf("[jrt] PASS — %d/%d ops correct, device stable\n", n_ok, total);
    else
        printf("[jrt] FAIL\n");

    return (n_fail==0 && n_dev_loss==0) ? 0 : 1;
}
