// xshard_validate.cpp — CPU-only .xshard loader and validator
// Reads header, walks all tiles, checksums each, prints a report.
// No GPU, no D3D11 — this is the "format is stable" gate.
//
// Build:
//   cl /std:c++17 /O2 /EHsc /I"..\include" xshard_validate.cpp /link /OUT:xshard_validate.exe
// Run:
//   xshard_validate.exe layer_00_q.xshard

#include <xshard.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>

static void die(const char* msg) {
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(1);
}

// Simple Adler-32 checksum (fast, no deps)
static uint32_t adler32(const void* buf, size_t n) {
    const uint8_t* p = (const uint8_t*)buf;
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < n; ++i) {
        a = (a + p[i]) % 65521;
        b = (b + a)    % 65521;
    }
    return (b << 16) | a;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <file.xshard>\n", argv[0]); return 1; }
    const char* path = argv[1];

    FILE* f = fopen(path, "rb");
    if (!f) die("cannot open file");

    // --- Read and validate header ---
    XShardHeader h{};
    if (fread(&h, 1, sizeof(h), f) != sizeof(h)) die("truncated header");

    if (!xshard_valid_magic(h)) {
        fprintf(stderr, "ERROR: bad magic '%c%c%c%c' (want XSQ2)\n",
                h.magic[0], h.magic[1], h.magic[2], h.magic[3]);
        fclose(f); return 1;
    }

    static const char* type_name[] = {"Q","K","V","O","?"};
    static const char* dtype_name[] = {"fp32","fp16","int8","int4","?"};
    uint32_t t = (h.tensor_type <= 3) ? h.tensor_type : 4;
    uint32_t d = (h.dtype <= 3)       ? h.dtype        : 4;

    printf("=== xshard: %s ===\n", path);
    printf("  magic:       %.4s\n",  h.magic);
    printf("  version:     %u\n",    h.version);
    printf("  layer:       %u\n",    h.layer_id);
    printf("  type:        %s (%u)\n", type_name[t], h.tensor_type);
    printf("  shape:       [%u, %u]\n", h.rows, h.cols);
    printf("  tile_size:   %u elements\n", h.tile_size);
    printf("  tile_count:  %u\n",    h.tile_count);
    printf("  dtype:       %s\n",    dtype_name[d]);

    const uint64_t tile_bytes_aligned = xshard_tile_bytes(h);
    const uint64_t elem_bytes = (h.dtype == XSHARD_FP32) ? 4
                              : (h.dtype == XSHARD_FP16) ? 2 : 1;
    const uint64_t tile_data_bytes = (uint64_t)h.tile_size * elem_bytes;
    const uint64_t expected_size   = sizeof(XShardHeader)
                                   + (uint64_t)h.tile_count * tile_bytes_aligned;

    printf("  tile bytes:  %llu (raw) / %llu (aligned)\n",
           (unsigned long long)tile_data_bytes,
           (unsigned long long)tile_bytes_aligned);
    printf("  expected file size: %.2f MB\n", expected_size / 1048576.0);

    // --- Walk tiles ---
    std::vector<uint8_t> tile_buf(tile_bytes_aligned);

    float g_min = 1e38f, g_max = -1e38f, g_sum = 0.f;
    int nan_count = 0, inf_count = 0;

    printf("\n  tile  adler32   min        max        mean\n");
    printf("  ----  --------  ---------  ---------  ---------\n");

    for (uint32_t i = 0; i < h.tile_count; ++i) {
        uint64_t off = xshard_tile_offset(h, i);
        if (fseek(f, (long)off, SEEK_SET) != 0) {
            fprintf(stderr, "ERROR: seek failed at tile %u\n", i);
            fclose(f); return 1;
        }
        size_t got = fread(tile_buf.data(), 1, tile_bytes_aligned, f);
        if (got < tile_data_bytes) {
            fprintf(stderr, "ERROR: tile %u truncated (got %zu, want %llu)\n",
                    i, got, (unsigned long long)tile_data_bytes);
            fclose(f); return 1;
        }

        uint32_t ck = adler32(tile_buf.data(), tile_data_bytes);

        float t_min = 1e38f, t_max = -1e38f, t_sum = 0.f;
        if (h.dtype == XSHARD_FP32) {
            const float* fp = (const float*)tile_buf.data();
            for (uint32_t e = 0; e < h.tile_size; ++e) {
                float v = fp[e];
                if (std::isnan(v))  { nan_count++; continue; }
                if (std::isinf(v))  { inf_count++; continue; }
                if (v < t_min) t_min = v;
                if (v > t_max) t_max = v;
                t_sum += v;
                if (v < g_min) g_min = v;
                if (v > g_max) g_max = v;
                g_sum += v;
            }
        }

        printf("  %4u  %08X  %9.4f  %9.4f  %9.6f\n",
               i, ck, t_min, t_max, t_sum / h.tile_size);
    }

    printf("\n  Global: min=%.4f  max=%.4f  mean=%.6f\n",
           g_min, g_max, g_sum / ((float)h.tile_size * h.tile_count));
    if (nan_count || inf_count)
        printf("  WARNING: %d NaN, %d Inf values\n", nan_count, inf_count);
    else
        printf("  No NaN/Inf — data clean.\n");

    fclose(f);
    printf("\nPASS\n");
    return 0;
}
