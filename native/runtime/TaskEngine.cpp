#include "task_engine.h"
#include "opencl_task_adapter.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <map>
#include <set>
#include <sstream>

namespace Kuhul::Runtime {

namespace {

std::string trim(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::vector<std::string> quotedValues(const std::string& line) {
    std::vector<std::string> values;
    size_t cursor = 0;
    while ((cursor = line.find('"', cursor)) != std::string::npos) {
        const size_t end = line.find('"', cursor + 1);
        if (end == std::string::npos) break;
        values.push_back(line.substr(cursor + 1, end - cursor - 1));
        cursor = end + 1;
    }
    return values;
}

std::string keyedValue(const std::string& line, const char* key) {
    const size_t marker = line.find(key);
    if (marker == std::string::npos) return {};
    const auto values = quotedValues(line.substr(marker + std::strlen(key)));
    return values.empty() ? std::string() : values.front();
}

std::string jsonString(const std::string& object, const char* key) {
    const std::string marker = std::string("\"") + key + "\"";
    const size_t markerPos = object.find(marker);
    if (markerPos == std::string::npos) return {};
    const size_t colon = object.find(':', markerPos + marker.size());
    const size_t quote = object.find('"', colon == std::string::npos ? colon : colon + 1);
    if (colon == std::string::npos || quote == std::string::npos) return {};
    const size_t end = object.find('"', quote + 1);
    return end == std::string::npos
        ? std::string{}
        : object.substr(quote + 1, end - quote - 1);
}

std::vector<std::string> jsonStringArray(const std::string& object,
                                         const char* key) {
    const std::string marker = std::string("\"") + key + "\"";
    const size_t markerPos = object.find(marker);
    if (markerPos == std::string::npos) return {};
    const size_t open = object.find('[', markerPos + marker.size());
    const size_t close = object.find(']', open == std::string::npos ? open : open + 1);
    if (open == std::string::npos || close == std::string::npos) return {};
    return quotedValues(object.substr(open, close - open + 1));
}

bool loadJsonTasks(const std::string& source, std::vector<TaskSpec>& tasks,
                   std::string& error) {
    const size_t tasksKey = source.find("\"tasks\"");
    if (tasksKey == std::string::npos) {
        error = "json_tasks_array_missing";
        return false;
    }
    const size_t arrayStart = source.find('[', tasksKey);
    if (arrayStart == std::string::npos) {
        error = "json_tasks_array_invalid";
        return false;
    }
    size_t arrayEnd = std::string::npos;
    size_t depth = 0;
    bool quoted = false;
    for (size_t i = arrayStart; i < source.size(); ++i) {
        if (source[i] == '"' && (i == 0 || source[i - 1] != '\\'))
            quoted = !quoted;
        if (quoted) continue;
        if (source[i] == '[') ++depth;
        if (source[i] == ']' && --depth == 0) {
            arrayEnd = i;
            break;
        }
    }
    if (arrayEnd == std::string::npos) {
        error = "json_tasks_array_invalid";
        return false;
    }

    size_t cursor = arrayStart + 1;
    while (cursor < arrayEnd) {
        const size_t open = source.find('{', cursor);
        if (open == std::string::npos || open >= arrayEnd) break;
        const size_t close = source.find('}', open + 1);
        if (close == std::string::npos || close > arrayEnd) {
            error = "json_task_object_invalid";
            return false;
        }
        const std::string object = source.substr(open, close - open + 1);
        TaskSpec task;
        task.id = jsonString(object, "id");
        task.action = jsonString(object, "action");
        task.description = jsonString(object, "description");
        task.provider = jsonString(object, "provider");
        task.dependsOn = jsonStringArray(object, "depends_on");
        if (task.dependsOn.empty())
            task.dependsOn = jsonStringArray(object, "depends");
        if (task.id.empty()) {
            error = "json_task_missing_id";
            return false;
        }
        tasks.push_back(std::move(task));
        cursor = close + 1;
    }
    return true;
}

std::string executorAbiDetail(const WebX::Provider& provider) {
    if (provider.id != "opencl_cpu_executor")
        if (provider.id == "opencl") {
            const auto probe = OpenCLTaskAdapter{}.probe();
            return probe.detail + ";platforms=" +
                   std::to_string(probe.platforms) + ";devices=" +
                   std::to_string(probe.devices);
        } else {
            return "provider_available";
        }

    char rootBuffer[4096] = {};
    const DWORD rootLength = GetEnvironmentVariableA(
        "KUHUL_DRIVER_ROOT", rootBuffer,
        static_cast<DWORD>(sizeof(rootBuffer)));
    HMODULE handle = nullptr;
    if (rootLength > 0) {
        const std::string path =
            std::string(rootBuffer, rootLength) + "\\" + provider.library;
        handle = LoadLibraryA(path.c_str());
    }
    if (!handle) handle = LoadLibraryA(provider.library.c_str());
    if (!handle) return "provider_available;export_probe_failed";

    constexpr const char* getTaskExecutor =
        "?GetTaskExecutor@TaskExecutor@OpenCL@Intel@@YAPEAVITaskExecutor@123@XZ";
    const bool exported = GetProcAddress(handle, getTaskExecutor) != nullptr;
    FreeLibrary(handle);
    return exported
        ? "provider_available;private_cpp_itaskexecutor_export"
        : "provider_available;task_executor_export_missing";
}

} // namespace

TaskEngine::TaskEngine(WebX::ProviderManager& providers)
    : providers_(providers) {}

bool TaskEngine::load(const std::string& path, std::string& error) {
    tasks_.clear();
    std::filesystem::path inputPath(path);
    if (inputPath.is_relative() && !std::filesystem::exists(inputPath)) {
        auto directory = std::filesystem::current_path();
        while (true) {
            const auto candidate = directory / inputPath;
            if (std::filesystem::exists(candidate)) {
                inputPath = candidate;
                break;
            }
            const auto parent = directory.parent_path();
            if (parent == directory) break;
            directory = parent;
        }
    }
    path_ = inputPath.string();

    std::ifstream input(inputPath);
    if (!input) {
        error = "cannot_open_task_list";
        return false;
    }

    if (path.size() >= 5 &&
        path.compare(path.size() - 5, 5, ".json") == 0) {
        std::ostringstream contents;
        contents << input.rdbuf();
        if (!loadJsonTasks(contents.str(), tasks_, error)) return false;
        return validate(error);
    }

    TaskSpec* current = nullptr;
    std::string line;
    while (std::getline(input, line)) {
        const std::string clean = trim(line);
        if (clean.empty()) continue;

        if (clean.rfind("task ", 0) == 0) {
            const auto values = quotedValues(clean);
            if (values.empty()) {
                error = "task_missing_id";
                return false;
            }
            tasks_.push_back(TaskSpec{});
            current = &tasks_.back();
            current->id = values.front();
            continue;
        }
        if (!current) continue;

        if (clean.find("action") != std::string::npos)
            current->action = keyedValue(clean, "action");
        else if (clean.find("description") != std::string::npos)
            current->description = keyedValue(clean, "description");
        else if (clean.find("provider") != std::string::npos)
            current->provider = keyedValue(clean, "provider");
        else if (clean.find("depends") != std::string::npos ||
                 clean.find("depends_on") != std::string::npos)
            current->dependsOn = quotedValues(clean);

        if (clean == "}" || clean == "};") current = nullptr;
    }
    return validate(error);
}

const WebX::Provider* TaskEngine::findProvider(const std::string& id) const {
    for (const auto& provider : providers_.getProviders()) {
        if (provider.id == id) return &provider;
    }
    return nullptr;
}

bool TaskEngine::validate(std::string& error) const {
    std::set<std::string> ids;
    for (const auto& task : tasks_) {
        if (task.id.empty() || !ids.insert(task.id).second) {
            error = "task_ids_must_be_unique";
            return false;
        }
        for (const auto& dependency : task.dependsOn) {
            if (ids.find(dependency) == ids.end()) {
                const auto found = std::find_if(
                    tasks_.begin(), tasks_.end(),
                    [&dependency](const TaskSpec& candidate) {
                        return candidate.id == dependency;
                    });
                if (found == tasks_.end()) {
                    error = "missing_task_dependency:" + dependency;
                    return false;
                }
            }
        }
    }
    if (tasks_.empty()) {
        error = "task_list_is_empty";
        return false;
    }
    return true;
}

std::vector<TaskResult> TaskEngine::plan() const {
    std::vector<TaskResult> results;
    std::set<std::string> completed;
    std::set<std::string> failed;
    std::set<std::string> remaining;
    for (const auto& task : tasks_) remaining.insert(task.id);

    while (!remaining.empty()) {
        bool progressed = false;
        for (const auto& task : tasks_) {
            if (!remaining.count(task.id)) continue;
            if (std::any_of(task.dependsOn.begin(), task.dependsOn.end(),
                            [&failed](const std::string& id) {
                                return failed.count(id) != 0;
                            })) {
                results.push_back({task.id, task.provider, "blocked",
                                   "dependency_failed"});
                failed.insert(task.id);
                completed.insert(task.id);
                remaining.erase(task.id);
                progressed = true;
                continue;
            }
            if (!std::all_of(task.dependsOn.begin(), task.dependsOn.end(),
                             [&completed](const std::string& id) {
                                 return completed.count(id) != 0;
                             }))
                continue;

            const WebX::Provider* provider = findProvider(task.provider);
            if (!provider) {
                results.push_back({task.id, task.provider, "blocked",
                                   "provider_not_registered"});
                failed.insert(task.id);
            } else if (!provider->available) {
                results.push_back({task.id, provider->id, "blocked",
                                   "provider_not_available"});
                failed.insert(task.id);
            } else {
                const std::string detail = executorAbiDetail(*provider);
                const bool helperReady =
                    provider->id != "opencl" ||
                    detail.find("standard_opencl_helper_ready") !=
                        std::string::npos;
                results.push_back({task.id, provider->id,
                                   helperReady ? "admitted" : "blocked",
                                   detail});
                if (!helperReady) failed.insert(task.id);
            }
            completed.insert(task.id);
            remaining.erase(task.id);
            progressed = true;
        }
        if (!progressed) {
            for (const auto& id : remaining)
                results.push_back({id, {}, "blocked", "dependency_cycle"});
            break;
        }
    }
    return results;
}

std::vector<TaskResult> TaskEngine::run(const TaskExecutor& executor) const {
    const auto scheduled = plan();
    if (!executor) return scheduled;

    std::vector<TaskResult> results;
    for (const auto& result : scheduled) {
        if (result.status != "admitted") {
            results.push_back(result);
            continue;
        }
        const auto task = std::find_if(
            tasks_.begin(), tasks_.end(),
            [&result](const TaskSpec& candidate) {
                return candidate.id == result.id;
            });
        const WebX::Provider* provider = findProvider(result.provider);
        std::string detail;
        const bool success = task != tasks_.end() && provider &&
                             executor(*task, *provider, detail);
        results.push_back({result.id, result.provider,
                           success ? "completed" : "failed",
                           detail.empty() ? (success ? "executor_completed"
                                                     : "executor_failed")
                                          : detail});
    }
    return results;
}

} // namespace Kuhul::Runtime
