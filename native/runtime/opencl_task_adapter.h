#pragma once

#include <cstdint>
#include <string>

namespace Kuhul::Runtime {

struct OpenCLProbeResult {
    bool available = false;
    uint32_t platforms = 0;
    uint32_t devices = 0;
    std::string detail;
};

class OpenCLTaskAdapter {
public:
    OpenCLProbeResult probe() const;
};

} // namespace Kuhul::Runtime
