// scxq2_infer_layer.hlsl — fused INT4-decode + matmul for 1-layer inference
// Matches gpt2_matmul_fwd.hlsl conventions (cs_5_0, tiled 16×16, same registers).
//
// Weight layout: each uint32 encodes 8 × INT4 values.
//   Each tile of W is stored as uint32[TILE][TILE/8].
//   Decode: nibble → signed int [-8,7] → float / scale
//
// Entry points:
//   CSDecodeStore   — decode INT4 weights → float buffer (one-time, streaming)
//   CSQProj         — fused decode+matmul: Y[M,N] = X[M,K] × W_int4[K,N]
//                     (replaces CSMain from gpt2_matmul_fwd for streamed weights)
//
// Dispatch for CSQProj: (ceil(N/16), ceil(M/16), 1)  numthreads(16,16,1)
// Dispatch for CSDecodeStore: (ceil(numPacked/64), 1, 1) numthreads(64,1,1)

// ── Constant buffer ───────────────────────────────────────────────────────────

cbuffer InferParams : register(b0) {
    uint M;           // batch × seq_len (rows of X / Y)
    uint K;           // hidden dim     (cols of X / rows of W)
    uint N;           // out dim        (cols of W / Y)
    uint use_bias;    // 1 = add bias[N] to output
    float w_scale;    // per-tensor INT4 scale  (typically 1.0/8.0)
    float w_zero;     // per-tensor zero-point  (typically 0.0)
    uint  numPacked;  // for CSDecodeStore: number of uint32 elements
    uint  dstOffset;  // for CSDecodeStore: element offset into OUT_decoded
};

// ── Resources ─────────────────────────────────────────────────────────────────

StructuredBuffer<float>   X           : register(t0); // input activations [M,K]
StructuredBuffer<uint>    W_int4      : register(t1); // packed INT4 weights [K, N/8]
StructuredBuffer<float>   bias        : register(t2); // bias [N]  (optional)
RWStructuredBuffer<float> Y           : register(u0); // output [M,N]
RWStructuredBuffer<float> OUT_decoded : register(u1); // decoded weights (CSDecodeStore)

// ── Groupshared tiles ─────────────────────────────────────────────────────────

#define TILE 16

groupshared float Xs[TILE][TILE];       // tile of X (float, loaded directly)
groupshared float Ws[TILE][TILE];       // tile of W (decoded on load, float)

// ── INT4 decode helpers ───────────────────────────────────────────────────────
// W_int4[r, c/8] holds 8 weights for columns c..c+7 of row r.
// Nibble order within uint32: bits[3:0]=w0, bits[7:4]=w1, …, bits[31:28]=w7

float DecodeNibble(uint packed, uint nibbleIdx) {
    int nibble = int((packed >> (nibbleIdx * 4)) & 0xF);
    return float(nibble - 8) * w_scale + w_zero;
}

// Load decoded float for weight[row, col] from packed W_int4
float LoadW(uint row, uint col) {
    uint packedIdx = row * ((N + 7) / 8) + col / 8;
    return DecodeNibble(W_int4[packedIdx], col % 8);
}

// ── CSDecodeStore — stream decode: INT4 → float buffer ───────────────────────
// Use for pre-decoding a tile into OUT_decoded before using it as a plain SRV.
// Dispatch(ceil(numPacked/64), 1, 1)

[numthreads(64, 1, 1)]
void CSDecodeStore(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= numPacked) return;

    uint packed = W_int4[i];

    uint base = dstOffset + i * 8;
    [unroll]
    for (uint j = 0; j < 8; ++j)
        OUT_decoded[base + j] = DecodeNibble(packed, j);
}

// ── CSQProj — fused INT4 decode + tiled matmul ────────────────────────────────
// Y[M,N] += X[M,K] × decode(W_int4[K,N])
// Decode happens inside the tile load for W — no intermediate float buffer.
// Matches gpt2_matmul_fwd.hlsl CSMain interface exactly so it can be swapped in.

[numthreads(TILE, TILE, 1)]
void CSQProj(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    const uint row = gid.y * TILE + lid.y;   // output row  (M)
    const uint col = gid.x * TILE + lid.x;   // output col  (N)

    float acc = 0.f;

    const uint numTiles = (K + TILE - 1) / TILE;

    for (uint t = 0; t < numTiles; ++t) {

        // -- Load X tile (float, straightforward) ----------------------------
        uint xCol = t * TILE + lid.x;
        Xs[lid.y][lid.x] = (row < M && xCol < K) ? X[row * K + xCol] : 0.f;

        // -- Load W tile (INT4 → decode → float) ----------------------------
        // W_int4 is [K rows × N cols packed].  Thread (lid.y, lid.x) loads
        // weight at [t*TILE + lid.y,  col_base + lid.x].
        uint wRow = t * TILE + lid.y;
        uint wCol = gid.x * TILE + lid.x;
        Ws[lid.y][lid.x] = (wRow < K && wCol < N) ? LoadW(wRow, wCol) : 0.f;

        GroupMemoryBarrierWithGroupSync();

        // -- Accumulate dot product ------------------------------------------
        [unroll]
        for (uint k = 0; k < TILE; ++k)
            acc += Xs[lid.y][k] * Ws[k][lid.x];

        GroupMemoryBarrierWithGroupSync();
    }

    if (row < M && col < N)
        Y[row * N + col] = acc + (use_bias ? bias[col] : 0.f);
}

// ── CSAttentionScores — Q·Kᵀ / sqrt(d_k), single head ───────────────────────
// Scores[M, M] = Q[M, d_k] × Kᵀ[d_k, M]
// Both Q and K are already decoded float (output of CSQProj).
// Dispatch(ceil(M/16), ceil(M/16), 1)

cbuffer AttnParams : register(b1) {
    uint A_M;    // sequence length
    uint A_dk;   // head dim (d_k)
    float inv_sqrt_dk;
    uint _pad;
};

StructuredBuffer<float>   Q      : register(t3); // [M, dk]
StructuredBuffer<float>   K_mat  : register(t4); // [M, dk]
RWStructuredBuffer<float> Scores : register(u2); // [M, M]

groupshared float Qs[TILE][TILE];
groupshared float Ks[TILE][TILE];

[numthreads(TILE, TILE, 1)]
void CSAttentionScores(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint row = gid.y * TILE + lid.y;  // query token
    uint col = gid.x * TILE + lid.x;  // key   token
    float acc = 0.f;

    uint numTiles = (A_dk + TILE - 1) / TILE;
    for (uint t = 0; t < numTiles; ++t) {
        uint k = t * TILE + lid.x;
        Qs[lid.y][lid.x] = (row < A_M && k < A_dk) ? Q[row * A_dk + k] : 0.f;

        uint kr = t * TILE + lid.y;
        Ks[lid.y][lid.x] = (col < A_M && kr < A_dk) ? K_mat[col * A_dk + kr] : 0.f;

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint i = 0; i < TILE; ++i)
            acc += Qs[lid.y][i] * Ks[i][lid.x];

        GroupMemoryBarrierWithGroupSync();
    }

    if (row < A_M && col < A_M)
        Scores[row * A_M + col] = acc * inv_sqrt_dk;
}
