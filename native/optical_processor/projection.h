#pragma once

#include "optical_processor.h"

// Spherical harmonic basis function (normalized, low order)
float evaluateSH(int index, float theta, float phi);

// Project SVG field onto SH basis
void projectSVGToSH(const std::vector<float>& svg_samples,
                    std::vector<OpticalNode>& nodes);
