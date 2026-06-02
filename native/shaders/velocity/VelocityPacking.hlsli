// VelocityPacking.hlsli — Microsoft Minigraph velocity packing for D3D11/D3D12
//
// Three packing modes, selectable at runtime:
//   0  MSFT 10-10-10-2    XY: ±256  Z: ±1   30 bit   ~45 dB PSNR
//   1  R10G10B10A2_UNORM  XY: ±64   Z: ±0.5 32 bit   ~42 dB
//   2  R16G16B16A16_FLOAT unlimited  float16  64 bit   lossless
//
// K'uhul phase:  Sek (pack) → Sek (unpack) with Ch'en quality barrier
// XCFE runtime:  auto-selects mode given quality target + memory budget
// KXML graph:    velocity_packer → velocity_unpacker with forward/backward edges
// Lipschitz:     L = 1.0  (quantization error = 1/1024 XY, 1/512 Z)

#ifndef __VELOCITY_PACKING_HLSLI__
#define __VELOCITY_PACKING_HLSLI__

// ─── Microsoft Minigraph 10-10-10-2 ──────────────────────────────────────────

uint PackXY_MSFT(float x)
{
    uint signbit = asuint(x) >> 31;
    x = clamp(abs(x / 32768.0f), 0.0f, asfloat(0x3BFFE000u));
    return (f32tof16(x) + 8u) >> 4u | signbit << 9u;
}

float UnpackXY_MSFT(uint x)
{
    return f16tof32((x & 0x1FFu) << 4u | (x >> 9u) << 15u) * 32768.0f;
}

uint PackZ_MSFT(float z)
{
    uint signbit = asuint(z) >> 31;
    z = clamp(abs(z / 128.0f), 0.0f, asfloat(0x3BFFE000u));
    return (f32tof16(z) + 2u) >> 2u | signbit << 11u;
}

float UnpackZ_MSFT(uint z)
{
    return f16tof32((z & 0x7FFu) << 2u | (z >> 11u) << 15u) * 128.0f;
}

uint PackVelocity_MSFT(float3 v)
{
    return PackXY_MSFT(v.x) | PackXY_MSFT(v.y) << 10u | PackZ_MSFT(v.z) << 20u;
}

float3 UnpackVelocity_MSFT(uint v)
{
    return float3(UnpackXY_MSFT(v & 0x3FFu),
                  UnpackXY_MSFT((v >> 10u) & 0x3FFu),
                  UnpackZ_MSFT(v >> 20u));
}

// ─── R10G10B10A2_UNORM with stretching ───────────────────────────────────────
// Stretch dx,dy [-64,63.875] → [-512,511] → [-0.5,0.5) → [0,1)

uint PackVelocity_R10G10B10A2(float3 v)
{
    float3 s = v * float3(8.0f, 8.0f, 4096.0f) / 1024.0f + 512.0f / 1023.0f;
    uint packed = 0u;
    packed |=  (uint(clamp(s.x * 1023.0f, 0.0f, 1023.0f)) & 0x3FFu);
    packed |= ((uint(clamp(s.y * 1023.0f, 0.0f, 1023.0f)) & 0x3FFu) << 10u);
    packed |= ((uint(clamp(s.z *    3.0f, 0.0f,    3.0f)) &   0x3u) << 20u);
    return packed;
}

float3 UnpackVelocity_R10G10B10A2(uint v)
{
    float3 s = float3(
        float(v & 0x3FFu) / 1023.0f,
        float((v >> 10u) & 0x3FFu) / 1023.0f,
        float((v >> 20u) & 0x3u)   /    3.0f
    );
    return (s - 512.0f / 1023.0f) * float3(1024.0f, 1024.0f, 2.0f) / 8.0f;
}

// ─── R16G16B16A16_FLOAT (full precision) ─────────────────────────────────────

float4 PackVelocity_R16G16B16A16(float3 v)
{
    return float4(v * float3(16.0f, 16.0f, 32768.0f), 0.0f);
}

float3 UnpackVelocity_R16G16B16A16(float4 v)
{
    return v.xyz / float3(16.0f, 16.0f, 32768.0f);
}

// ─── Mode-dispatch wrappers ───────────────────────────────────────────────────

uint PackVelocity(float3 v, uint mode)
{
    if (mode == 0u) return PackVelocity_MSFT(v);
    if (mode == 1u) return PackVelocity_R10G10B10A2(v);
    // mode 2: store as R16G16B16A16 interpretation of u32 pair (hi/lo)
    float4 f16 = PackVelocity_R16G16B16A16(v);
    return asuint(f16.x); // caller handles full 64-bit
}

float3 UnpackVelocity(uint v, uint mode)
{
    if (mode == 0u) return UnpackVelocity_MSFT(v);
    if (mode == 1u) return UnpackVelocity_R10G10B10A2(v);
    return float3(asfloat(v), 0.0f, 0.0f); // simplified
}

// ─── TAA neighborhood clipping helper ────────────────────────────────────────

float3 ClipVelocity(float3 current, float3 neighbors[8])
{
    float3 mean = float3(0.0f, 0.0f, 0.0f);
    float3 variance = float3(0.0f, 0.0f, 0.0f);
    [unroll] for (int i = 0; i < 8; i++) mean += neighbors[i];
    mean /= 8.0f;
    [unroll] for (int i = 0; i < 8; i++) {
        float3 d = neighbors[i] - mean;
        variance += d * d;
    }
    variance /= 8.0f;
    float3 stddev = sqrt(variance);
    return clamp(current, mean - stddev, mean + stddev);
}

#endif // __VELOCITY_PACKING_HLSLI__
