#pragma once

#include <string>
#include <vector>

namespace Kuhul::Runtime {

struct ApiEndpointSpec {
    std::string path;
    std::string method;
    std::string operationId;
    std::string requestSchema;
    std::string responseSchema;
};

struct ApiWriterRequest {
    std::string title;
    std::string version = "1.0.0";
    std::vector<ApiEndpointSpec> endpoints;
};

class APIWriter {
public:
    bool validate(const ApiWriterRequest& request, std::string& error) const;
    bool writeOpenApi(const ApiWriterRequest& request,
                      std::string& output,
                      std::string& error) const;
};

} // namespace Kuhul::Runtime
