#pragma once

#include <cstdint>
#include <string>

class LlamaRuntime {
public:
    ~LlamaRuntime();

    bool load(const std::string& runtime_root);
    bool generate(const std::string& model_path,
                  const std::string& prompt,
                  uint32_t max_tokens,
                  std::string& output);

    const std::string& error() const { return error_; }
    bool loaded() const { return llama_ != nullptr; }

private:
    void unload();
    void* llama_ = nullptr;
    void* ggml_ = nullptr;
    std::string error_;
};
