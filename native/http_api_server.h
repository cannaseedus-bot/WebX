#pragma once

#include "llama_runtime.h"

#include <cstdint>
#include <string>

class HttpApiServer {
public:
    explicit HttpApiServer(uint16_t port);
    ~HttpApiServer();

    HttpApiServer(const HttpApiServer&) = delete;
    HttpApiServer& operator=(const HttpApiServer&) = delete;

    bool start();
    void run();
    void stop();

    const std::string& error() const { return error_; }
    uint16_t port() const { return port_; }

private:
    struct Impl;
    Impl* impl_ = nullptr;
    uint16_t port_;
    std::string error_;
    LlamaRuntime llama_;
    bool llama_loaded_ = false;
};
