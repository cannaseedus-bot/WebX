#define WIN32_LEAN_AND_MEAN
#include "task_executor_abi.h"

#include <windows.h>

#include <filesystem>
#include <string>

namespace Kuhul::Runtime {

namespace {

std::string jsonEscape(const std::string& value) {
    std::string escaped;
    for (const char ch : value) {
        if (ch == '\\' || ch == '"') escaped.push_back('\\');
        escaped.push_back(ch);
    }
    return escaped;
}

std::filesystem::path helperPath() {
    char buffer[MAX_PATH] = {};
    const DWORD length = GetModuleFileNameA(nullptr, buffer, MAX_PATH);
    if (length == 0 || length >= MAX_PATH) return {};
    return std::filesystem::path(buffer).parent_path() / "kuhul_task_helper.exe";
}

} // namespace

bool TaskHelperExecutor::execute(const TaskSpec& task,
                                  const WebX::Provider& provider,
                                  std::string& detail) const {
    detail.clear();
    if (task.action != "probe_provider" &&
        task.action != "probe_opencl" &&
        task.action != "stream_xshard" &&
        task.action != "verify_scx") {
        detail = "helper_action_not_allowlisted:" + task.action;
        return false;
    }

    const auto helper = helperPath();
    if (helper.empty() || !std::filesystem::is_regular_file(helper)) {
        detail = "task_helper_missing:" + helper.string();
        return false;
    }

    SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE inputRead = nullptr;
    HANDLE inputWrite = nullptr;
    HANDLE outputRead = nullptr;
    HANDLE outputWrite = nullptr;
    if (!CreatePipe(&inputRead, &inputWrite, &security, 0) ||
        !CreatePipe(&outputRead, &outputWrite, &security, 0)) {
        if (inputRead) CloseHandle(inputRead);
        if (inputWrite) CloseHandle(inputWrite);
        if (outputRead) CloseHandle(outputRead);
        if (outputWrite) CloseHandle(outputWrite);
        detail = "task_helper_pipe_failed";
        return false;
    }
    SetHandleInformation(inputWrite, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(outputRead, HANDLE_FLAG_INHERIT, 0);

    std::string command = "\"" + helper.string() + "\" --execute";
    std::vector<char> commandLine(command.begin(), command.end());
    commandLine.push_back('\0');
    STARTUPINFOA startup{};
    startup.cb = sizeof(startup);
    startup.hStdInput = inputRead;
    startup.hStdOutput = outputWrite;
    startup.hStdError = outputWrite;
    startup.dwFlags |= STARTF_USESTDHANDLES;
    PROCESS_INFORMATION process{};
    if (!CreateProcessA(nullptr, commandLine.data(), nullptr, nullptr, TRUE,
                        CREATE_NO_WINDOW, nullptr, nullptr, &startup,
                        &process)) {
        CloseHandle(inputRead);
        CloseHandle(inputWrite);
        CloseHandle(outputRead);
        CloseHandle(outputWrite);
        detail = "task_helper_start_failed:" +
                 std::to_string(static_cast<unsigned long>(GetLastError()));
        return false;
    }
    CloseHandle(inputRead);
    CloseHandle(outputWrite);

    const std::string request =
        "{\"contract\":\"task-executor.v1\",\"request_id\":\"" +
        jsonEscape(task.id) + "\",\"action\":\"" + jsonEscape(task.action) +
        "\",\"provider\":\"" + jsonEscape(provider.id) +
        "\",\"library\":\"" + jsonEscape(provider.library) + "\"}\n";
    DWORD written = 0;
    const bool writeOk = WriteFile(inputWrite, request.data(),
                                   static_cast<DWORD>(request.size()),
                                   &written, nullptr);
    CloseHandle(inputWrite);
    if (!writeOk || written != request.size()) {
        TerminateProcess(process.hProcess, 1);
        WaitForSingleObject(process.hProcess, 1000);
        CloseHandle(outputRead);
        CloseHandle(process.hThread);
        CloseHandle(process.hProcess);
        detail = "task_helper_write_failed";
        return false;
    }

    const DWORD helperTimeout =
        (task.action == "stream_xshard" || task.action == "verify_scx")
            ? 35000
            : 5000;
    const DWORD waitResult =
        WaitForSingleObject(process.hProcess, helperTimeout);
    if (waitResult != WAIT_OBJECT_0) {
        TerminateProcess(process.hProcess, 1);
        WaitForSingleObject(process.hProcess, 1000);
        CloseHandle(outputRead);
        CloseHandle(process.hThread);
        CloseHandle(process.hProcess);
        detail = waitResult == WAIT_TIMEOUT
            ? "task_helper_timeout"
            : "task_helper_wait_failed";
        return false;
    }

    std::string response;
    char buffer[512];
    DWORD read = 0;
    while (ReadFile(outputRead, buffer, sizeof(buffer), &read, nullptr) &&
           read > 0)
        response.append(buffer, read);
    CloseHandle(outputRead);
    DWORD exitCode = 1;
    GetExitCodeProcess(process.hProcess, &exitCode);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);

    const bool ok = response.find("\"ok\":true") != std::string::npos &&
                    exitCode == 0;
    detail = response.empty() ? "task_helper_empty_response" : response;
    return ok;
}

} // namespace Kuhul::Runtime
