#include "opencl_task_adapter.h"

#include <Windows.h>
#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <vector>

namespace Kuhul::Runtime {

namespace {

std::string readPipe(HANDLE pipe) {
    std::string output;
    char buffer[512];
    DWORD bytesRead = 0;
    while (ReadFile(pipe, buffer, sizeof(buffer), &bytesRead, nullptr) &&
           bytesRead != 0)
        output.append(buffer, bytesRead);
    return output;
}

uint32_t jsonUint(const std::string& json, const char* key) {
    const std::string marker = std::string("\"") + key + "\":";
    const size_t start = json.find(marker);
    if (start == std::string::npos) return 0;
    char* end = nullptr;
    const unsigned long value = std::strtoul(
        json.c_str() + start + marker.size(), &end, 10);
    return end == json.c_str() + start + marker.size()
               ? 0
               : static_cast<uint32_t>(value);
}

} // namespace

OpenCLProbeResult OpenCLTaskAdapter::probe() const {
    OpenCLProbeResult result;
    char modulePath[MAX_PATH] = {};
    const DWORD moduleLength =
        GetModuleFileNameA(nullptr, modulePath, sizeof(modulePath));
    std::string helperPath;
    char configuredPath[4096] = {};
    const DWORD configuredLength = GetEnvironmentVariableA(
        "KUHUL_OPENCL_HELPER", configuredPath,
        static_cast<DWORD>(sizeof(configuredPath)));
    if (configuredLength > 0) {
        helperPath.assign(configuredPath, configuredLength);
    } else if (moduleLength > 0) {
        helperPath.assign(modulePath, moduleLength);
        const size_t slash = helperPath.find_last_of("\\/");
        helperPath = helperPath.substr(0, slash + 1) +
                     "kuhul_opencl_helper.exe";
    } else {
        helperPath = "kuhul_opencl_helper.exe";
    }

    std::string command = "\"" + helperPath + "\" --probe";
    std::vector<char> commandLine(command.begin(), command.end());
    commandLine.push_back('\0');
    SECURITY_ATTRIBUTES security{};
    security.nLength = sizeof(security);
    security.bInheritHandle = TRUE;
    HANDLE outputRead = nullptr;
    HANDLE outputWrite = nullptr;
    if (!CreatePipe(&outputRead, &outputWrite, &security, 4096)) {
        result.detail = "helper_pipe_failed";
        return result;
    }
    SetHandleInformation(outputRead, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOA startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdOutput = outputWrite;
    startup.hStdError = outputWrite;
    PROCESS_INFORMATION process{};
    if (!CreateProcessA(nullptr, commandLine.data(), nullptr, nullptr, FALSE,
                        CREATE_NO_WINDOW, nullptr, nullptr, &startup,
                        &process)) {
        CloseHandle(outputRead);
        CloseHandle(outputWrite);
        result.detail = "helper_not_started";
        return result;
    }
    CloseHandle(outputWrite);

    const DWORD waitResult = WaitForSingleObject(process.hProcess, 5000);
    if (waitResult == WAIT_TIMEOUT) {
        TerminateProcess(process.hProcess, 124);
        WaitForSingleObject(process.hProcess, 1000);
        CloseHandle(outputRead);
        CloseHandle(process.hThread);
        CloseHandle(process.hProcess);
        result.detail = "helper_timeout";
        return result;
    }

    DWORD exitCode = 1;
    GetExitCodeProcess(process.hProcess, &exitCode);
    const std::string output = readPipe(outputRead);
    CloseHandle(outputRead);
    result.available = waitResult == WAIT_OBJECT_0 && exitCode == 0;
    result.platforms = jsonUint(output, "platforms");
    result.devices = jsonUint(output, "devices");
    std::ostringstream detail;
    detail << (result.available ? "standard_opencl_helper_ready"
                                : "standard_opencl_helper_failed")
           << ";exit_code=" << exitCode << ";platforms=" << result.platforms
           << ";devices=" << result.devices;
    result.detail = detail.str();
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    return result;
}

} // namespace Kuhul::Runtime
