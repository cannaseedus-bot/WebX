#pragma once

#include <string>
#include <vector>

namespace Kuhul::Runtime {

class OpenGLFrameAdapter {
public:
    bool renderSmoke(unsigned frames = 1);
    bool renderObjSmoke(const std::string& path, unsigned frames = 1,
                        bool interactive = false,
                        const std::vector<std::string>& blocks = {});
    const std::string& error() const { return error_; }

private:
    std::string error_;
};

} // namespace Kuhul::Runtime
