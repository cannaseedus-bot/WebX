#include "projection.h"
#include <cmath>

// Low-order SH basis functions (normalized)
float evaluateSH(int index, float theta, float phi)
{
    const float PI = 3.14159265359f;
    const float SQRT_PI = 1.77245385090f;

    // SH basis: Y_l^m(theta, phi)
    // Using normalized spherical harmonics
    // Index mapping: 0=Y00, 1=Y10, 2=Y11, 3=Y1-1, 4=Y20, ...

    float cosTheta = cosf(theta);
    float sinTheta = sinf(theta);
    float cos2Theta = cosf(2 * theta);
    float sin2Theta = sinf(2 * theta);

    switch (index)
    {
        case 0:  // Y00
            return 0.282095f;
        case 1:  // Y10
            return 0.488603f * cosTheta;
        case 2:  // Y11
            return 0.488603f * sinTheta * cosf(phi);
        case 3:  // Y1-1
            return 0.488603f * sinTheta * sinf(phi);
        case 4:  // Y20
            return 1.092548f * (cosTheta * cosTheta - 0.333333f);
        case 5:  // Y21
            return 2.185095f * sinTheta * cosTheta * cosf(phi);
        case 6:  // Y2-1
            return 2.185095f * sinTheta * cosTheta * sinf(phi);
        case 7:  // Y22
            return 1.092548f * sinTheta * sinTheta * cosf(2 * phi);
        case 8:  // Y2-2
            return 1.092548f * sinTheta * sinTheta * sinf(2 * phi);
        default:
            return 0.0f;
    }
}

void projectSVGToSH(const std::vector<float>& svg_samples,
                    std::vector<OpticalNode>& nodes)
{
    // Simple SVG projection: use sample values directly as SH coefficients
    // In a real implementation, would parse SVG and integrate over sphere

    int num_samples = (int)svg_samples.size();
    int nodes_per_sample = (int)nodes.size() / (num_samples > 0 ? num_samples : 1);

    for (int s = 0; s < num_samples && s < (int)nodes.size(); s++)
    {
        float u = (s % 32) / 32.0f;
        float v = (s / 32) / 32.0f;

        float theta = u * 3.14159265359f;
        float phi = v * 6.28318530718f;

        float value = svg_samples[s] * 0.1f;  // Scale down

        for (int i = 0; i < SH_BANDS; i++)
        {
            float Y = evaluateSH(i, theta, phi);
            if (s < nodes.size())
            {
                nodes[s].sh[i].x += Y * value;
            }
        }
    }
}
