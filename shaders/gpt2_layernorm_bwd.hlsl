// gpt2_layernorm_bwd.hlsl — LayerNorm backward pass
// y = (x - mean) / sqrt(var + eps) * gamma + beta
// dL/dx, dL/dgamma, dL/dbeta
//
// Two-pass reduction for correctness:
// Pass 1: Compute per-thread partial sums, reduce in shared memory
// Pass 2: Compute dL/dx using reduced sums
//
// Dispatch(seq_len, 1, 1)  numthreads(256, 1, 1)

cbuffer LNBwdParams : register(b0) {
    uint n_embd;
    uint seq_len;
    float eps;
    float inv_n;       // 1.0 / n_embd
    uint pad0;
    uint pad1;
};

StructuredBuffer<float> x      : register(t0);  // input [seq, n_embd]
StructuredBuffer<float> gamma  : register(t1);  // weight [n_embd]
StructuredBuffer<float> y_norm : register(t2);  // normalized activations (x_hat) [seq, n_embd]
StructuredBuffer<float> dout   : register(t3);  // upstream gradient [seq, n_embd]

RWStructuredBuffer<float> dx      : register(u0);
RWStructuredBuffer<float> dgamma  : register(u1);
RWStructuredBuffer<float> dbeta   : register(u2);

groupshared float gs_sum_dout[256];
groupshared float gs_sum_dout_xhat[256];

[numthreads(256, 1, 1)]
void CSMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    const uint seq_idx = gid.x;
    const uint base    = seq_idx * n_embd;
    const uint tid = lid.x;

    // Step 1: Accumulate partial sums and gradient contributions
    float sum_dout = 0.0f;
    float sum_dout_xhat = 0.0f;
    
    for (uint i = tid; i < n_embd; i += 256) {
        const float dy    = dout[base + i];
        const float xhat  = y_norm[base + i];
        const float g     = gamma[i];
        sum_dout         += dy * g;
        sum_dout_xhat    += dy * g * xhat;
        dgamma[i]        += dy * xhat;
        dbeta[i]         += dy;
    }

    // Store partial sums in shared memory
    gs_sum_dout[tid] = sum_dout;
    gs_sum_dout_xhat[tid] = sum_dout_xhat;
    GroupMemoryBarrierWithGroupSync();

    // Step 2: Parallel reduction in shared memory
    if (tid < 128) {
        gs_sum_dout[tid] += gs_sum_dout[tid + 128];
        gs_sum_dout_xhat[tid] += gs_sum_dout_xhat[tid + 128];
    }
    GroupMemoryBarrierWithGroupSync();
    
    if (tid < 64) {
        gs_sum_dout[tid] += gs_sum_dout[tid + 64];
        gs_sum_dout_xhat[tid] += gs_sum_dout_xhat[tid + 64];
    }
    GroupMemoryBarrierWithGroupSync();
    
    if (tid < 32) {
        gs_sum_dout[tid] += gs_sum_dout[tid + 32];
        gs_sum_dout_xhat[tid] += gs_sum_dout_xhat[tid + 32];
    }
    GroupMemoryBarrierWithGroupSync();
    
    if (tid < 16) {
        gs_sum_dout[tid] += gs_sum_dout[tid + 16];
        gs_sum_dout_xhat[tid] += gs_sum_dout_xhat[tid + 16];
    }
    GroupMemoryBarrierWithGroupSync();
    
    if (tid < 8) {
        gs_sum_dout[tid] += gs_sum_dout[tid + 8];
        gs_sum_dout_xhat[tid] += gs_sum_dout_xhat[tid + 8];
    }
    GroupMemoryBarrierWithGroupSync();
    
    if (tid < 4) {
        gs_sum_dout[tid] += gs_sum_dout[tid + 4];
        gs_sum_dout_xhat[tid] += gs_sum_dout_xhat[tid + 4];
    }
    GroupMemoryBarrierWithGroupSync();
    
    if (tid < 2) {
        gs_sum_dout[tid] += gs_sum_dout[tid + 2];
        gs_sum_dout_xhat[tid] += gs_sum_dout_xhat[tid + 2];
    }
    GroupMemoryBarrierWithGroupSync();
    
    if (tid < 1) {
        gs_sum_dout[0] += gs_sum_dout[1];
        gs_sum_dout_xhat[0] += gs_sum_dout_xhat[1];
    }
    GroupMemoryBarrierWithGroupSync();

    // Step 3: Compute dL/dx using reduced sums
    // dx_i = (1/n) * gamma_i * (dout_i - sum_dout/n - x_hat_i * sum_dout_xhat/n)
    const float total_sum_dout = gs_sum_dout[0];
    const float total_sum_dout_xhat = gs_sum_dout_xhat[0];
    
    for (uint i = tid; i < n_embd; i += 256) {
        const float dy    = dout[base + i];
        const float xhat  = y_norm[base + i];
        const float g     = gamma[i];
        
        float dx_val = dy - total_sum_dout * inv_n - xhat * total_sum_dout_xhat * inv_n;
        dx[base + i] += g * dx_val * inv_n;
    }
}
