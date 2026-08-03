#include "collapse.h"
#include <cmath>

float collapseNode(const OpticalNode& node)
{
    float energy = 0.0f;

    for (int i = 0; i < SH_BANDS; i++)
    {
        float len = sqrtf(node.sh[i].x * node.sh[i].x +
                         node.sh[i].y * node.sh[i].y);
        energy += len;
    }

    return energy / SH_BANDS;
}

float collapseGlobal(const std::vector<OpticalNode>& nodes)
{
    if (nodes.empty()) return 0.0f;

    float total = 0.0f;

    for (const auto& node : nodes)
        total += collapseNode(node);

    return total / nodes.size();
}
