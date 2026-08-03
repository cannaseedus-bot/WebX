#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::string field(const std::string& json, const char* key) {
    const std::string marker = std::string("\"") + key + "\":\"";
    const size_t start = json.find(marker);
    if (start == std::string::npos) return {};
    const size_t valueStart = start + marker.size();
    const size_t end = json.find('"', valueStart);
    return end == std::string::npos
        ? std::string{}
        : json.substr(valueStart, end - valueStart);
}

bool providerLibraryPresent(const std::string& library) {
    if (library.empty()) return false;
    char root[4096] = {};
    const DWORD length = GetEnvironmentVariableA(
        "KUHUL_DRIVER_ROOT", root, static_cast<DWORD>(sizeof(root)));
    if (length > 0) {
        const std::string path =
            std::string(root, length) + "\\" + library;
        HMODULE module = LoadLibraryA(path.c_str());
        if (module) {
            FreeLibrary(module);
            return true;
        }
    }
    HMODULE module = LoadLibraryA(library.c_str());
    if (!module) return false;
    FreeLibrary(module);
    return true;
}

bool runNativeAction(const std::string& action, std::string& detail) {
    char rootBuffer[4096] = {};
    const DWORD rootLength = GetEnvironmentVariableA(
        "KUHUL_XSHARD_ROOT", rootBuffer, static_cast<DWORD>(sizeof(rootBuffer)));
    if (rootLength == 0) {
        detail = "KUHUL_XSHARD_ROOT_missing";
        return false;
    }

    char modulePath[MAX_PATH] = {};
    const DWORD moduleLength =
        GetModuleFileNameA(nullptr, modulePath, sizeof(modulePath));
    if (moduleLength == 0) {
        detail = "kuhul_engine_path_unavailable";
        return false;
    }
    std::string engine(modulePath, moduleLength);
    engine = engine.substr(0, engine.find_last_of("\\/") + 1) +
             "kuhul_engine.exe";
    const std::string commandName =
        action == "stream_xshard" ? "stream-xshard" : "scx-d3d11-smoke";
    const std::string arguments = action == "stream_xshard"
        ? " \"" + std::string(rootBuffer, rootLength) + "\" 0 4"
        : " \"" + std::string(rootBuffer, rootLength) + "\"";
    std::string command = "\"" + engine + "\" " + commandName + arguments;
    std::vector<char> commandLine(command.begin(), command.end());
    commandLine.push_back('\0');

    SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE nullHandle = CreateFileA(
        "NUL", GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
        &security, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (nullHandle == INVALID_HANDLE_VALUE) {
        detail = "native_action_output_handle_failed";
        return false;
    }
    STARTUPINFOA startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    startup.hStdOutput = nullHandle;
    startup.hStdError = nullHandle;
    PROCESS_INFORMATION process{};
    const BOOL started = CreateProcessA(
        nullptr, commandLine.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW,
        nullptr, nullptr, &startup, &process);
    CloseHandle(nullHandle);
    if (!started) {
        detail = "native_action_start_failed";
        return false;
    }

    const DWORD waitResult = WaitForSingleObject(process.hProcess, 30000);
    DWORD exitCode = 1;
    if (waitResult == WAIT_OBJECT_0)
        GetExitCodeProcess(process.hProcess, &exitCode);
    else {
        TerminateProcess(process.hProcess, 124);
        detail = waitResult == WAIT_TIMEOUT
            ? "native_action_timeout"
            : "native_action_wait_failed";
    }
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    if (waitResult != WAIT_OBJECT_0) return false;
    detail = "native_action_exit=" + std::to_string(exitCode);
    return exitCode == 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2 || std::string(argv[1]) != "--execute") {
        std::cerr << "Usage: kuhul_task_helper.exe --execute\n";
        return 2;
    }

    std::string request((std::istreambuf_iterator<char>(std::cin)),
                        std::istreambuf_iterator<char>());
    const std::string action = field(request, "action");
    const std::string contract = field(request, "contract");
    const std::string requestId = field(request, "request_id");
    const std::string provider = field(request, "provider");
    const std::string library = field(request, "library");
    if (contract != "task-executor.v1" || requestId.empty() ||
        (action != "probe_provider" && action != "probe_opencl" &&
         action != "stream_xshard" && action != "verify_scx") ||
        provider.empty() || library.empty()) {
        std::cout << "{\"ok\":false,\"error\":\"invalid_task_request\"}\n";
        return 1;
    }
    if (action == "probe_opencl") {
        char modulePath[MAX_PATH] = {};
        const DWORD moduleLength =
            GetModuleFileNameA(nullptr, modulePath, sizeof(modulePath));
        std::string helper = "kuhul_opencl_helper.exe";
        if (moduleLength > 0) {
            helper = std::string(modulePath, moduleLength);
            helper = helper.substr(0, helper.find_last_of("\\/") + 1) +
                     "kuhul_opencl_helper.exe";
        }
        std::string command = "\"" + helper + "\" --execute";
        STARTUPINFOA startup{};
        startup.cb = sizeof(startup);
        startup.dwFlags = STARTF_USESTDHANDLES;
        startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
        startup.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
        startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
        PROCESS_INFORMATION process{};
        std::vector<char> commandLine(command.begin(), command.end());
        commandLine.push_back('\0');
        if (!CreateProcessA(nullptr, commandLine.data(), nullptr, nullptr, TRUE,
                            CREATE_NO_WINDOW, nullptr, nullptr, &startup,
                            &process)) {
            std::cout << "{\"ok\":false,\"error\":\"opencl_helper_start_failed\"}\n";
            return 1;
        }
        CloseHandle(process.hThread);
        WaitForSingleObject(process.hProcess, 6000);
        DWORD exitCode = 1;
        GetExitCodeProcess(process.hProcess, &exitCode);
        CloseHandle(process.hProcess);
        return static_cast<int>(exitCode);
    }
    if (action == "stream_xshard" || action == "verify_scx") {
        if (!providerLibraryPresent(library)) {
            std::cout << "{\"ok\":false,\"error\":\"provider_library_missing\"}\n";
            return 1;
        }
        std::string detail;
        const bool success = runNativeAction(action, detail);
        std::cout << "{\"ok\":" << (success ? "true" : "false")
                  << ",\"action\":\"" << action
                  << "\",\"detail\":\"" << detail << "\"}\n";
        return success ? 0 : 1;
    }
    if (!providerLibraryPresent(library)) {
        std::cout << "{\"ok\":false,\"error\":\"provider_library_missing\"}\n";
        return 1;
    }
    std::cout << "{\"ok\":true,\"action\":\"probe_provider\","
                 "\"provider\":\"" << provider << "\"}\n";
    return 0;
}
