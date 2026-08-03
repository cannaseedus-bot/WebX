#include "api_writer.h"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace Kuhul::Runtime {

namespace {

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return value;
}

std::string json(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size() + 2);
    for (const char c : value) {
        if (c == '"' || c == '\\') escaped.push_back('\\');
        escaped.push_back(c);
    }
    return "\"" + escaped + "\"";
}

} // namespace

bool APIWriter::validate(const ApiWriterRequest& request,
                         std::string& error) const {
    if (request.title.empty() || request.version.empty()) {
        error = "api_metadata_missing";
        return false;
    }
    if (request.endpoints.empty()) {
        error = "api_endpoints_empty";
        return false;
    }
    for (const auto& endpoint : request.endpoints) {
        const std::string method = lower(endpoint.method);
        if (endpoint.path.empty() || endpoint.path.front() != '/' ||
            endpoint.operationId.empty()) {
            error = "api_endpoint_identity_invalid";
            return false;
        }
        if (method != "get" && method != "post" && method != "put" &&
            method != "patch" && method != "delete") {
            error = "api_method_invalid:" + endpoint.method;
            return false;
        }
        if (endpoint.responseSchema.empty()) {
            error = "api_response_schema_missing:" + endpoint.operationId;
            return false;
        }
    }
    return true;
}

bool APIWriter::writeOpenApi(const ApiWriterRequest& request,
                             std::string& output,
                             std::string& error) const {
    if (!validate(request, error)) return false;

    std::ostringstream document;
    document << "{\n"
             << "  \"openapi\": \"3.0.3\",\n"
             << "  \"info\": {\"title\": " << json(request.title)
             << ", \"version\": " << json(request.version) << "},\n"
             << "  \"paths\": {";
    for (std::size_t i = 0; i < request.endpoints.size(); ++i) {
        const auto& endpoint = request.endpoints[i];
        if (i != 0) document << ",";
        document << "\n    " << json(endpoint.path) << ": {"
                 << json(lower(endpoint.method)) << ": {"
                 << "\"operationId\": " << json(endpoint.operationId)
                 << ", \"responses\": {\"200\": {\"description\": \"OK\", "
                    "\"x-kuhul-response-schema\": "
                 << json(endpoint.responseSchema) << "}}";
        if (!endpoint.requestSchema.empty()) {
            document << ", \"x-kuhul-request-schema\": "
                     << json(endpoint.requestSchema);
        }
        document << "}}";
    }
    document << "\n  }\n}\n";
    output = document.str();
    return true;
}

} // namespace Kuhul::Runtime
