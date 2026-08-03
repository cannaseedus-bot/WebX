#pragma once

#include "../webx_compute.h"

#include <functional>
#include <string>
#include <vector>

namespace Kuhul::Runtime {

struct TaskSpec {
    std::string id;
    std::string action;
    std::string description;
    std::string provider;
    std::vector<std::string> dependsOn;
};

struct TaskResult {
    std::string id;
    std::string provider;
    std::string status;
    std::string detail;
};

using TaskExecutor = std::function<bool(
    const TaskSpec&, const WebX::Provider&, std::string& detail)>;

class TaskEngine {
public:
    explicit TaskEngine(WebX::ProviderManager& providers);

    bool load(const std::string& path, std::string& error);
    bool validate(std::string& error) const;
    std::vector<TaskResult> plan() const;
    std::vector<TaskResult> run(const TaskExecutor& executor) const;
    const std::vector<TaskSpec>& tasks() const { return tasks_; }

private:
    WebX::ProviderManager& providers_;
    std::vector<TaskSpec> tasks_;
    std::string path_;

    const WebX::Provider* findProvider(const std::string& id) const;
};

} // namespace Kuhul::Runtime
