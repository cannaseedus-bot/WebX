#pragma once

#include "optical_processor.h"

// Compute energy of a single node (coherence metric)
float collapseNode(const OpticalNode& node);

// Compute global coherence (average across all nodes)
float collapseGlobal(const std::vector<OpticalNode>& nodes);
