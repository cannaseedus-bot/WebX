#pragma once

#include <cstdint>
#include <string>

class XShardAttentionEngine {
public:
    bool run(const std::string& q_path,
             const std::string& k_path,
             const std::string& v_path,
             uint32_t passes = 1);

    const std::string& error() const { return error_; }

private:
    std::string error_;
};
