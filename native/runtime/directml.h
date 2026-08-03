#pragma once

#include <string>

namespace Kuhul::Runtime {

struct DirectMLStatus {
    bool runtimeLoaded = false;
    bool createDeviceExported = false;
    bool available = false;
    std::string detail;
};

class DirectMLProvider {
public:
    DirectMLStatus probe() const;
};

} // namespace Kuhul::Runtime
