#include <Windows.h>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using cl_int = int32_t;
using cl_uint = uint32_t;
using cl_platform_id = void*;
using GetPlatformIDs = cl_int (*)(cl_uint, cl_platform_id*, cl_uint*);
using GetDeviceIDs = cl_int (*)(cl_platform_id, uint64_t, cl_uint,
                                void*, cl_uint*);

constexpr cl_int CL_SUCCESS = 0;
constexpr uint64_t CL_DEVICE_TYPE_ALL = 0xFFFFFFFFull;

int probe() {
    HMODULE runtime = nullptr;
    char root[4096] = {};
    const DWORD rootLength = GetEnvironmentVariableA(
        "KUHUL_DRIVER_ROOT", root, static_cast<DWORD>(sizeof(root)));
    if (rootLength > 0) {
        const std::string directory(root, rootLength);
        for (const auto& name : {"IntelOpenCL64.dll", "Intel_OpenCL_ICD64.dll",
                                 "ocl_cpu_IntelOpenCL64.dll",
                                 "ocl_cpu_intelocl64.dll"}) {
            runtime = LoadLibraryA((directory + "\\" + name).c_str());
            if (runtime) break;
        }
    }
    for (const auto& name : {"IntelOpenCL64.dll", "Intel_OpenCL_ICD64.dll",
                             "ocl_cpu_IntelOpenCL64.dll",
                             "ocl_cpu_intelocl64.dll"}) {
        if (runtime) break;
        runtime = LoadLibraryA(name);
    }
    if (!runtime) {
        std::cout << "{\"ok\":false,\"error\":\"runtime_not_loaded\"}\n";
        return 1;
    }

    const auto getPlatforms = reinterpret_cast<GetPlatformIDs>(
        GetProcAddress(runtime, "clGetPlatformIDs"));
    const auto getDevices = reinterpret_cast<GetDeviceIDs>(
        GetProcAddress(runtime, "clGetDeviceIDs"));
    if (!getPlatforms || !getDevices) {
        std::cout << "{\"ok\":false,\"error\":\"exports_missing\"}\n";
        FreeLibrary(runtime);
        return 1;
    }

    cl_uint platformCount = 0;
    if (getPlatforms(0, nullptr, &platformCount) != CL_SUCCESS ||
        platformCount == 0) {
        std::cout << "{\"ok\":false,\"error\":\"no_platforms\"}\n";
        FreeLibrary(runtime);
        return 1;
    }

    std::vector<cl_platform_id> platforms(platformCount);
    if (getPlatforms(platformCount, platforms.data(), nullptr) != CL_SUCCESS) {
        std::cout << "{\"ok\":false,\"error\":\"platform_enumeration_failed\"}\n";
        FreeLibrary(runtime);
        return 1;
    }

    cl_uint deviceCount = 0;
    for (const auto platform : platforms) {
        cl_uint count = 0;
        if (getDevices(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &count) ==
            CL_SUCCESS)
            deviceCount += count;
    }

    std::cout << "{\"ok\":" << (deviceCount > 0 ? "true" : "false")
              << ",\"platforms\":" << platformCount
              << ",\"devices\":" << deviceCount << "}\n";
    FreeLibrary(runtime);
    return deviceCount > 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2 || (std::string(argv[1]) != "--probe" &&
                      std::string(argv[1]) != "--execute")) {
        std::cerr << "Usage: kuhul_opencl_helper.exe --probe|--execute\n";
        return 2;
    }
    if (std::string(argv[1]) == "--execute") {
        std::string request((std::istreambuf_iterator<char>(std::cin)),
                            std::istreambuf_iterator<char>());
        if (request.find("\"action\":\"probe_opencl\"") == std::string::npos) {
            std::cout << "{\"ok\":false,\"error\":\"invalid_opencl_task\"}\n";
            return 1;
        }
    }
    return probe();
}
