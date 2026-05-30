// asx_ram_test.cpp
// ASX RAM Controller — 3-slot streaming pipeline for one attention layer
//
// Streams all 16 attention heads of layer 0 through a fixed GPU window.
// Slot transitions:
//
//   Sync mode:    [C] disk→cpu  →  [B] cpu ready  →  [A] GPU dispatch  →  verify
//   Prefetch mode:               [C] async disk←thread
//                 while [A] GPU dispatch runs concurrently         ↓ join
//                  [B] swap cur↔nxt →  [A] GPU dispatch  →  verify
//
// Each head produces a [64 × 1024] context vector (256KB = one tile).
// Tolerance vs CPU reference: 1e-3 (same as --v mode in xshard_tile_test).
//
// Usage: asx_ram_test.exe <q_shard> <k_shard> <v_shard> [n_passes] [--prefetch]
//   n_passes:   how many times to sweep all 16 heads (default 1, stress = 10)
//   --prefetch: enable async slot C disk I/O overlapping with GPU slot A

#include "asx_ram_controller.h"
#include "xshard.h"

#include <d3dcompiler.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <future>
#include <vector>

using Clock = std::chrono::steady_clock;
static float elapsed_ms(Clock::time_point t0) {
    return std::chrono::duration<float, std::milli>(Clock::now() - t0).count();
}

// ── embedded shaders (identical to xshard_tile_test --v mode) ─────────────────

static const char* kSoftmaxHLSL = R"hlsl(
Buffer<float>   Q   : register(t0);
Buffer<float>   K   : register(t1);
RWBuffer<float> Out : register(u0);
#define D_MODEL    1024u
#define SEQ        64u
#define INV_SQRT_D 0.03125f
groupshared float sdata[SEQ];
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint row = gid.x; uint col = tid.x;
    float s = 0.f;
    for (uint i = 0u; i < D_MODEL; i++)
        s += Q[row * D_MODEL + i] * K[col * D_MODEL + i];
    s *= INV_SQRT_D;
    sdata[col] = s;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 32u; stride > 0u; stride >>= 1u) {
        if (col < stride) sdata[col] = max(sdata[col], sdata[col+stride]);
        GroupMemoryBarrierWithGroupSync();
    }
    float m = sdata[0];
    GroupMemoryBarrierWithGroupSync();
    float e = exp(s - m);
    sdata[col] = e;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 32u; stride > 0u; stride >>= 1u) {
        if (col < stride) sdata[col] += sdata[col+stride];
        GroupMemoryBarrierWithGroupSync();
    }
    Out[row * SEQ + col] = e / sdata[0];
}
)hlsl";

static const char* kVmulHLSL = R"hlsl(
Buffer<float>   P   : register(t0);
Buffer<float>   V   : register(t1);
RWBuffer<float> Out : register(u0);
#define SEQ     64u
#define D_MODEL 1024u
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint row = id.x; uint col = id.y;
    if (row >= SEQ || col >= D_MODEL) return;
    float s = 0.f;
    for (uint k = 0u; k < SEQ; k++)
        s += P[row * SEQ + k] * V[k * D_MODEL + col];
    Out[row * D_MODEL + col] = s;
}
)hlsl";

// ── helpers ───────────────────────────────────────────────────────────────────

static void die(const char* msg) { fprintf(stderr, "[asx_ram] FATAL: %s\n", msg); exit(1); }

static void chk(HRESULT hr, const char* label) {
    if (FAILED(hr)) { fprintf(stderr, "[asx_ram] %s hr=0x%08X\n", label, (unsigned)hr); exit(1); }
}

static ComPtr<ID3D11Buffer> make_buf(ID3D11Device* dev, UINT bytes,
                                      D3D11_USAGE usage, UINT bind, UINT cpu) {
    D3D11_BUFFER_DESC d{}; d.ByteWidth=bytes; d.Usage=usage; d.BindFlags=bind; d.CPUAccessFlags=cpu;
    ComPtr<ID3D11Buffer> b; chk(dev->CreateBuffer(&d,nullptr,&b),"CreateBuffer"); return b;
}
static ComPtr<ID3D11ShaderResourceView> make_srv(ID3D11Device* dev, ID3D11Buffer* buf, UINT n) {
    D3D11_SHADER_RESOURCE_VIEW_DESC d{}; d.Format=DXGI_FORMAT_R32_FLOAT;
    d.ViewDimension=D3D11_SRV_DIMENSION_BUFFER; d.Buffer.NumElements=n;
    ComPtr<ID3D11ShaderResourceView> v; chk(dev->CreateShaderResourceView(buf,&d,&v),"SRV"); return v;
}
static ComPtr<ID3D11UnorderedAccessView> make_uav(ID3D11Device* dev, ID3D11Buffer* buf, UINT n) {
    D3D11_UNORDERED_ACCESS_VIEW_DESC d{}; d.Format=DXGI_FORMAT_R32_FLOAT;
    d.ViewDimension=D3D11_UAV_DIMENSION_BUFFER; d.Buffer.NumElements=n;
    ComPtr<ID3D11UnorderedAccessView> v; chk(dev->CreateUnorderedAccessView(buf,&d,&v),"UAV"); return v;
}
static ComPtr<ID3D11ComputeShader> compile_cs(ID3D11Device* dev, const char* src, const char* name) {
    ComPtr<ID3DBlob> blob, errs;
    HRESULT hr = D3DCompile(src,strlen(src),name,nullptr,nullptr,"main","cs_5_0",
                            D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&blob,&errs);
    if (FAILED(hr)) { if (errs) fprintf(stderr,"%s\n",(char*)errs->GetBufferPointer()); die("D3DCompile"); }
    ComPtr<ID3D11ComputeShader> cs;
    chk(dev->CreateComputeShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&cs),"CreateCS");
    return cs;
}

// ── GPUWindow implementation ──────────────────────────────────────────────────

bool GPUWindow::init(ID3D11Device* dev, const char* softmax_src, const char* vmul_src) {
    constexpr UINT TILE_ELEMS = 65536;
    constexpr UINT TILE_BYTES = TILE_ELEMS * 4;
    constexpr UINT P_ELEMS    = 64 * 64;
    constexpr UINT CTX_ELEMS  = 64 * 1024;

    qBuf = make_buf(dev, TILE_BYTES, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0);
    kBuf = make_buf(dev, TILE_BYTES, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0);
    vBuf = make_buf(dev, TILE_BYTES, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0);
    q_srv = make_srv(dev, qBuf.Get(), TILE_ELEMS);
    k_srv = make_srv(dev, kBuf.Get(), TILE_ELEMS);
    v_srv = make_srv(dev, vBuf.Get(), TILE_ELEMS);

    pBuf  = make_buf(dev, P_ELEMS*4, D3D11_USAGE_DEFAULT,
                     D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS, 0);
    p_srv = make_srv(dev, pBuf.Get(), P_ELEMS);
    p_uav = make_uav(dev, pBuf.Get(), P_ELEMS);

    ctxBuf     = make_buf(dev, CTX_ELEMS*4, D3D11_USAGE_DEFAULT, D3D11_BIND_UNORDERED_ACCESS, 0);
    ctxStaging = make_buf(dev, CTX_ELEMS*4, D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ);
    ctx_uav    = make_uav(dev, ctxBuf.Get(), CTX_ELEMS);

    softmax_cs = compile_cs(dev, softmax_src, "asx_softmax.hlsl");
    vmul_cs    = compile_cs(dev, vmul_src,    "asx_vmul.hlsl");
    return true;
}

void GPUWindow::upload(ID3D11DeviceContext* ctx, const TileSlot& slot) {
    ctx->UpdateSubresource(qBuf.Get(), 0, nullptr, slot.q.data(), 0, 0);
    ctx->UpdateSubresource(kBuf.Get(), 0, nullptr, slot.k.data(), 0, 0);
    ctx->UpdateSubresource(vBuf.Get(), 0, nullptr, slot.v.data(), 0, 0);
}

void GPUWindow::dispatch(ID3D11DeviceContext* ctx) {
    // Pass 1: softmax(Q, K) → pBuf
    {
        ID3D11ShaderResourceView*  srvs[2] = { q_srv.Get(), k_srv.Get() };
        ID3D11UnorderedAccessView* uav     = p_uav.Get();
        ctx->CSSetShader(softmax_cs.Get(), nullptr, 0);
        ctx->CSSetShaderResources(0, 2, srvs);
        ctx->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
        ctx->Dispatch(64, 1, 1);
        ID3D11UnorderedAccessView* nu    = nullptr;
        ID3D11ShaderResourceView*  ns[2] = { nullptr, nullptr };
        ctx->CSSetUnorderedAccessViews(0, 1, &nu, nullptr);
        ctx->CSSetShaderResources(0, 2, ns);
        ctx->CSSetShader(nullptr, nullptr, 0);
    }
    // Pass 2: P × V → ctxBuf
    {
        ID3D11ShaderResourceView*  srvs[2] = { p_srv.Get(), v_srv.Get() };
        ID3D11UnorderedAccessView* uav     = ctx_uav.Get();
        ctx->CSSetShader(vmul_cs.Get(), nullptr, 0);
        ctx->CSSetShaderResources(0, 2, srvs);
        ctx->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
        ctx->Dispatch(8, 128, 1);
        ID3D11UnorderedAccessView* nu    = nullptr;
        ID3D11ShaderResourceView*  ns[2] = { nullptr, nullptr };
        ctx->CSSetUnorderedAccessViews(0, 1, &nu, nullptr);
        ctx->CSSetShaderResources(0, 2, ns);
        ctx->CSSetShader(nullptr, nullptr, 0);
    }
}

bool GPUWindow::readback(ID3D11DeviceContext* ctx, float* out) {
    ctx->CopyResource(ctxStaging.Get(), ctxBuf.Get());
    D3D11_MAPPED_SUBRESOURCE mr{};
    HRESULT hr = ctx->Map(ctxStaging.Get(), 0, D3D11_MAP_READ, 0, &mr);
    if (FAILED(hr)) return false;
    memcpy(out, mr.pData, 64 * 1024 * 4);
    ctx->Unmap(ctxStaging.Get(), 0);
    return true;
}

// ── CPU references ────────────────────────────────────────────────────────────

static void cpu_softmax(const float* Q, const float* K, float* P) {
    for (int r = 0; r < 64; ++r) {
        float scores[64]; float maxv = -1e30f;
        for (int c = 0; c < 64; ++c) {
            float s = 0.f;
            for (int i = 0; i < 1024; ++i) s += Q[r*1024+i] * K[c*1024+i];
            scores[c] = s * 0.03125f;
            if (scores[c] > maxv) maxv = scores[c];
        }
        float sum = 0.f;
        for (int c = 0; c < 64; ++c) { scores[c] = expf(scores[c]-maxv); sum += scores[c]; }
        for (int c = 0; c < 64; ++c) P[r*64+c] = scores[c] / sum;
    }
}
static void cpu_v_mul(const float* P, const float* V, float* out) {
    for (int r = 0; r < 64; ++r)
        for (int c = 0; c < 1024; ++c) {
            float s = 0.f;
            for (int k = 0; k < 64; ++k) s += P[r*64+k] * V[k*1024+c];
            out[r*1024+c] = s;
        }
}

// ── xshard loader ─────────────────────────────────────────────────────────────

struct ShardFile {
    FILE* fp = nullptr; XShardHeader hdr{};
    bool open(const char* p) {
        fp = fopen(p,"rb");
        if (!fp) { fprintf(stderr,"[asx_ram] cannot open %s\n",p); return false; }
        if (fread(&hdr,1,64,fp)!=64 || !xshard_valid_magic(hdr)) { fclose(fp); return false; }
        return true;
    }
    bool read_tile(uint32_t t, float* buf) {
        uint64_t off = xshard_tile_offset(hdr, t);
        if (fseek(fp,(long)off,SEEK_SET)!=0) return false;
        size_t bytes = (size_t)hdr.tile_size * 4;
        return fread(buf,1,bytes,fp)==bytes;
    }
    ~ShardFile() { if (fp) fclose(fp); }
};

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    const char* q_path = argc>1 ? argv[1] : nullptr;
    const char* k_path = argc>2 ? argv[2] : nullptr;
    const char* v_path = argc>3 ? argv[3] : nullptr;
    int passes = argc>4 ? atoi(argv[4]) : 1;
    bool use_prefetch = false;
    for (int i = 1; i < argc; ++i)
        if (strcmp(argv[i], "--prefetch") == 0) { use_prefetch = true; if (passes < 2) passes = 1; }

    if (!q_path || !k_path || !v_path) {
        fprintf(stderr, "Usage: asx_ram_test.exe <q_shard> <k_shard> <v_shard> [passes] [--prefetch]\n");
        return 1;
    }

    printf("[asx_ram] q: %s\n[asx_ram] k: %s\n[asx_ram] v: %s\n", q_path, k_path, v_path);
    printf("[asx_ram] passes  : %d\n", passes);
    printf("[asx_ram] prefetch: %s\n\n", use_prefetch ? "YES (async slot C)" : "no (synchronous)");

    ShardFile q_shard, k_shard, v_shard;
    if (!q_shard.open(q_path)) die("open Q shard");
    if (!k_shard.open(k_path)) die("open K shard");
    if (!v_shard.open(v_path)) die("open V shard");

    // Separate FILE* handles for the prefetch thread (FILE* is not thread-safe)
    ShardFile q_pf, k_pf, v_pf;
    if (use_prefetch) {
        if (!q_pf.open(q_path)) die("open Q shard (prefetch)");
        if (!k_pf.open(k_path)) die("open K shard (prefetch)");
        if (!v_pf.open(v_path)) die("open V shard (prefetch)");
    }

    const uint32_t N_HEADS = q_shard.hdr.tile_count;   // 16
    printf("[asx_ram] heads per layer: %u  tile_size: %u floats (256KB)\n\n", N_HEADS, q_shard.hdr.tile_size);

    // ── D3D11 init ────────────────────────────────────────────────────────────
    D3D11Engine eng;
    if (!eng.init(false, true)) die("D3D11 init");
    ID3D11Device*        dev = eng.rawDevice();
    ID3D11DeviceContext* ctx = eng.rawCtx();
    printf("[asx_ram] adapter: %s\n\n", eng.adapterName().c_str());

    // ── GPU window init ───────────────────────────────────────────────────────
    GPUWindow gpu;
    if (!gpu.init(dev, kSoftmaxHLSL, kVmulHLSL)) die("GPUWindow init");
    printf("[asx_ram] GPU window ready  (Q/K/V=256KB each, P=16KB, ctx=256KB)\n");
    printf("[asx_ram] shaders: softmax(64×1×1 groups) + vmul(8×128×1 groups)\n\n");

    // ── tile CPU slots ────────────────────────────────────────────────────────
    // Sync mode: 3-slot ring  C→B→A
    // Prefetch:  2-slot ping-pong  cur(GPU) / nxt(disk thread)
    TileSlot slots[3];
    for (auto& s : slots) s.alloc();

    // Lambda: load Q/K/V tile into a slot, return actual disk time (ms).
    // Called on main thread (sync) or prefetch thread (async).
    auto load_slot = [](TileSlot& slot, uint32_t head,
                        ShardFile& q, ShardFile& k, ShardFile& v) -> float {
        slot.phase = SlotPhase::Loading;
        slot.key   = { 0u, head };
        auto t0    = Clock::now();
        bool ok    = q.read_tile(head, slot.q.data())
                  && k.read_tile(head, slot.k.data())
                  && v.read_tile(head, slot.v.data());
        float ms   = std::chrono::duration<float,std::milli>(Clock::now()-t0).count();
        slot.phase = ok ? SlotPhase::Ready : SlotPhase::Empty;
        slot.load_ms = ms;
        if (!ok) { fprintf(stderr,"[asx_ram] read tile failed head=%u\n",head); exit(1); }
        return ms;
    };

    // Output buffers
    std::vector<float> ctx_gpu(64 * 1024);
    std::vector<float> p_ref(64 * 64);
    std::vector<float> ctx_ref(64 * 1024);

    // Stats
    int   total_heads = 0, mismatches = 0, device_losses = 0;
    float max_err = 0.f, total_disk_ms = 0.f, total_gpu_ms = 0.f, total_wait_ms = 0.f;
    const float TOL = 1e-3f;

    if (use_prefetch) {
        printf("[asx_ram] ── prefetch pipeline: slot C async, slot A GPU ─────────────\n");
        printf("[asx_ram]  pass  head  disk_ms  wait_ms  gpu_ms  max_err\n");
        printf("[asx_ram] ─────────────────────────────────────────────────────────────\n");
    } else {
        printf("[asx_ram] ── synchronous pass/head table ────────────────────────────────\n");
        printf("[asx_ram]  pass  head  tile  slot:C→B→A  disk_ms  gpu_ms  max_err\n");
        printf("[asx_ram] ─────────────────────────────────────────────────────────────\n");
    }

    for (int pass = 0; pass < passes; ++pass) {

        if (use_prefetch) {
            // ── PREFETCH PIPELINE ─────────────────────────────────────────────
            // Double-buffer: slots[0]=cur (GPU), slots[1]=nxt (disk thread)
            int cur = 0, nxt = 1;
            // Synchronously load head 0 to prime the pipeline
            load_slot(slots[cur], 0, q_shard, k_shard, v_shard);

            for (uint32_t head = 0; head < N_HEADS; ++head) {
                // Fire slot C: async load of head+1 while GPU runs head
                std::future<float> pf_fut;
                if (head + 1 < N_HEADS) {
                    uint32_t next_head = head + 1;
                    pf_fut = std::async(std::launch::async,
                        [&, next_head]() {
                            return load_slot(slots[nxt], next_head, q_pf, k_pf, v_pf);
                        });
                }

                // Slot A: GPU dispatch (concurrent with slot C disk read)
                slots[cur].phase = SlotPhase::Computing;
                auto t_gpu = Clock::now();
                gpu.upload(ctx, slots[cur]);
                gpu.dispatch(ctx);
                bool ok = gpu.readback(ctx, ctx_gpu.data());
                float gpu_ms = elapsed_ms(t_gpu);
                slots[cur].phase = SlotPhase::Empty;

                if (!ok) {
                    fprintf(stderr, "[asx_ram] device lost pass=%d head=%u\n", pass, head);
                    ++device_losses; goto done;
                }

                // Join slot C: measure how long we stalled waiting (ideally 0)
                float disk_ms = 0.f, wait_ms = 0.f;
                if (head + 1 < N_HEADS) {
                    auto t_wait = Clock::now();
                    disk_ms = pf_fut.get();
                    wait_ms = elapsed_ms(t_wait);  // 0 if disk finished before GPU
                }

                // Verify
                cpu_softmax(slots[cur].q.data(), slots[cur].k.data(), p_ref.data());
                cpu_v_mul(p_ref.data(), slots[cur].v.data(), ctx_ref.data());
                float head_err = 0.f; bool mismatch = false;
                for (int e = 0; e < 64*1024; ++e) {
                    float err = fabsf(ctx_gpu[e] - ctx_ref[e]);
                    if (err > head_err) head_err = err;
                    if (err > TOL) { mismatch = true; break; }
                }
                if (head_err > max_err) max_err = head_err;
                if (mismatch) { fprintf(stderr,"[asx_ram] MISMATCH pass=%d head=%u\n",pass,head); ++mismatches; }

                total_disk_ms += disk_ms;
                total_gpu_ms  += gpu_ms;
                total_wait_ms += wait_ms;
                ++total_heads;

                printf("[asx_ram]   %3d   %3u   %6.2f   %6.2f   %6.2f  %.2e%s\n",
                       pass, head, disk_ms, wait_ms, gpu_ms, head_err,
                       mismatch ? "  FAIL" : "");

                // Swap cur ↔ nxt for next iteration
                std::swap(cur, nxt);
            }

        } else {
            // ── SYNCHRONOUS PIPELINE (original) ──────────────────────────────
            for (uint32_t head = 0; head < N_HEADS; ++head) {
                TileSlot& slot_c = slots[head % 3];

                // [C] LOAD — disk → CPU buffer
                slot_c.phase   = SlotPhase::Loading;
                slot_c.key     = { 0u, head };
                auto t_disk    = Clock::now();
                if (!q_shard.read_tile(head, slot_c.q.data())) die("read Q tile");
                if (!k_shard.read_tile(head, slot_c.k.data())) die("read K tile");
                if (!v_shard.read_tile(head, slot_c.v.data())) die("read V tile");
                float disk_ms  = elapsed_ms(t_disk);
                slot_c.phase   = SlotPhase::Ready;
                slot_c.load_ms = disk_ms;

                // [B→A] UPLOAD + COMPUTE
                slot_c.phase = SlotPhase::Computing;
                auto t_gpu   = Clock::now();
                gpu.upload(ctx, slot_c);
                gpu.dispatch(ctx);
                bool ok = gpu.readback(ctx, ctx_gpu.data());
                float gpu_ms = elapsed_ms(t_gpu);
                slot_c.phase = SlotPhase::Empty;

                if (!ok) {
                    fprintf(stderr, "[asx_ram] device lost pass=%d head=%u\n", pass, head);
                    ++device_losses; goto done;
                }

                // VERIFY
                cpu_softmax(slot_c.q.data(), slot_c.k.data(), p_ref.data());
                cpu_v_mul(p_ref.data(), slot_c.v.data(), ctx_ref.data());
                float head_err = 0.f; bool mismatch = false;
                for (int e = 0; e < 64*1024; ++e) {
                    float err = fabsf(ctx_gpu[e] - ctx_ref[e]);
                    if (err > head_err) head_err = err;
                    if (err > TOL) { mismatch = true; break; }
                }
                if (head_err > max_err) max_err = head_err;
                if (mismatch) {
                    fprintf(stderr, "[asx_ram] MISMATCH pass=%d head=%u err=%.2e\n",
                            pass, head, head_err);
                    ++mismatches;
                }

                total_disk_ms += disk_ms;
                total_gpu_ms  += gpu_ms;
                ++total_heads;

                printf("[asx_ram]   %3d   %3u    %2u    C→B→A       %5.2f    %5.2f   %.2e%s\n",
                       pass, head, head, disk_ms, gpu_ms, head_err,
                       mismatch ? "  FAIL" : "");
            }
        }
    }

done:
    printf("[asx_ram] ─────────────────────────────────────────────────────────────\n\n");

    float total_ms   = use_prefetch ? total_gpu_ms  // disk hidden; wall ≈ gpu
                                    : total_disk_ms + total_gpu_ms;
    float throughput = (total_heads > 0) ? (float)total_heads / (total_ms * 0.001f) : 0.f;
    float mb_sec     = throughput * 3.f * 0.25f;   // 3 tiles × 256KB per head
    float hidden_ms  = total_disk_ms - total_wait_ms;  // disk time masked by GPU

    printf("[asx_ram] total heads  : %d\n",   total_heads);
    printf("[asx_ram] disk time    : %.1f ms (%.2f ms/head actual reads)\n",
           total_disk_ms, total_heads ? total_disk_ms/total_heads : 0.f);
    if (use_prefetch) {
        printf("[asx_ram] disk hidden  : %.1f ms  (%.0f%% of reads overlapped with GPU)\n",
               hidden_ms,
               total_disk_ms > 0.f ? 100.f * hidden_ms / total_disk_ms : 100.f);
        printf("[asx_ram] disk stall   : %.1f ms (%.2f ms/head main-thread wait)\n",
               total_wait_ms, total_heads ? total_wait_ms/total_heads : 0.f);
    }
    printf("[asx_ram] gpu  time    : %.1f ms (%.1f ms/head)\n",
           total_gpu_ms,  total_heads ? total_gpu_ms/total_heads  : 0.f);
    printf("[asx_ram] throughput   : %.1f heads/sec  (%.1f MB/sec tile load)\n",
           throughput, mb_sec);
    printf("[asx_ram] max_err      : %.2e  (tol=%.0e)\n", max_err, TOL);
    printf("[asx_ram] mismatches   : %d\n", mismatches);
    printf("[asx_ram] device losses: %d\n\n", device_losses);

    if (mismatches == 0 && device_losses == 0) {
        if (use_prefetch)
            printf("[asx_ram] PASS — all %d head(s) correct, prefetch slot C async, device stable\n",
                   total_heads);
        else
            printf("[asx_ram] PASS — all %d head(s) correct, device stable\n", total_heads);
    } else {
        printf("[asx_ram] FAIL\n");
    }

    return (mismatches == 0 && device_losses == 0) ? 0 : 1;
}
