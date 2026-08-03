#pragma once

#include "task_engine.h"

namespace Kuhul::Runtime {

class TaskHelperExecutor {
public:
    bool execute(const TaskSpec& task,
                 const WebX::Provider& provider,
                 std::string& detail) const;
};

} // namespace Kuhul::Runtime
