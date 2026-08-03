#define WIN32_LEAN_AND_MEAN
#include "directml.h"

#include <windows.h>

namespace Kuhul::Runtime {

DirectMLStatus DirectMLProvider::probe() const {
    DirectMLStatus status;
    HMODULE runtime = LoadLibraryA("DirectML.dll");
    if (!runtime) {
        status.detail = "directml_runtime_not_loaded";
        return status;
    }
    status.runtimeLoaded = true;
    status.createDeviceExported =
        GetProcAddress(runtime, "DMLCreateDevice") != nullptr;
    status.available = status.createDeviceExported;
    status.detail = status.available
        ? "directml_runtime_and_device_export_ready"
        : "directml_device_export_missing";
    FreeLibrary(runtime);
    return status;
}

} // namespace Kuhul::Runtime
