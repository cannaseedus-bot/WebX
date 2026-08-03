#define WIN32_LEAN_AND_MEAN
#include "http_api_server.h"
#include "runtime/KuhulAppCreator.h"

#include <windows.h>
#include <http.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <sstream>
#include <vector>

#pragma comment(lib, "httpapi.lib")

struct HttpApiServer::Impl {
    HTTP_SERVER_SESSION_ID session = HTTP_NULL_ID;
    HTTP_URL_GROUP_ID url_group = HTTP_NULL_ID;
    HANDLE request_queue = nullptr;
    std::wstring url;
    bool initialized = false;
};

namespace {

std::string jsonEscape(const std::string& value) {
    std::ostringstream out;
    for (unsigned char ch : value) {
        switch (ch) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\b': out << "\\b"; break;
        case '\f': out << "\\f"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (ch < 0x20) {
                out << "\\u00";
                const char* hex = "0123456789abcdef";
                out << hex[ch >> 4] << hex[ch & 0x0f];
            } else {
                out << static_cast<char>(ch);
            }
        }
    }
    return out.str();
}

class JsonRequest {
public:
    explicit JsonRequest(const std::string& text) : text_(text) {}

    bool stringField(const char* name, std::string& value) const {
        const size_t position = findField(name);
        if (position == std::string::npos) return false;
        size_t cursor = position;
        skipWhitespace(cursor);
        if (cursor >= text_.size() || text_[cursor] != '"') return false;
        ++cursor;
        value.clear();
        while (cursor < text_.size()) {
            const char ch = text_[cursor++];
            if (ch == '"') return true;
            if (ch != '\\') {
                value += ch;
                continue;
            }
            if (cursor >= text_.size()) return false;
            const char escaped = text_[cursor++];
            switch (escaped) {
            case '"': value += '"'; break;
            case '\\': value += '\\'; break;
            case '/': value += '/'; break;
            case 'b': value += '\b'; break;
            case 'f': value += '\f'; break;
            case 'n': value += '\n'; break;
            case 'r': value += '\r'; break;
            case 't': value += '\t'; break;
            default: return false;
            }
        }
        return false;
    }

    bool uintField(const char* name, uint32_t& value) const {
        const size_t position = findField(name);
        if (position == std::string::npos) return false;
        size_t cursor = position;
        skipWhitespace(cursor);
        if (cursor >= text_.size() || !std::isdigit(
                static_cast<unsigned char>(text_[cursor]))) return false;
        uint64_t parsed = 0;
        while (cursor < text_.size() &&
               std::isdigit(static_cast<unsigned char>(text_[cursor]))) {
            parsed = parsed * 10 + static_cast<unsigned>(text_[cursor++] - '0');
            if (parsed > std::numeric_limits<uint32_t>::max()) return false;
        }
        value = static_cast<uint32_t>(parsed);
        return true;
    }

private:
    size_t findField(const char* name) const {
        const std::string needle = std::string("\"") + name + "\"";
        const size_t key = text_.find(needle);
        if (key == std::string::npos) return key;
        size_t cursor = key + needle.size();
        skipWhitespace(cursor);
        if (cursor >= text_.size() || text_[cursor] != ':') return std::string::npos;
        ++cursor;
        skipWhitespace(cursor);
        return cursor;
    }

    void skipWhitespace(size_t& cursor) const {
        while (cursor < text_.size() &&
               std::isspace(static_cast<unsigned char>(text_[cursor]))) ++cursor;
    }

    const std::string& text_;
};

std::string statusJson(const std::string& status, const std::string& message = {}) {
    std::string json = "{\"status\":\"" + jsonEscape(status) + "\"";
    if (!message.empty()) json += ",\"error\":\"" + jsonEscape(message) + "\"";
    return json + "}";
}

std::string requestBody(const HTTP_REQUEST* request,
                        HANDLE queue,
                        HTTP_REQUEST_ID requestId) {
    std::string body;
    for (ULONG i = 0; i < request->EntityChunkCount; ++i) {
        const HTTP_DATA_CHUNK& chunk = request->pEntityChunks[i];
        if (chunk.DataChunkType == HttpDataChunkFromMemory &&
            chunk.FromMemory.pBuffer) {
            body.append(static_cast<const char*>(chunk.FromMemory.pBuffer),
                        chunk.FromMemory.BufferLength);
        }
    }
    if ((request->Flags & HTTP_REQUEST_FLAG_MORE_ENTITY_BODY_EXISTS) == 0)
        return body;

    std::vector<BYTE> buffer(16 * 1024);
    ULONG received = 0;
    ULONG result = ERROR_SUCCESS;
    do {
        result = HttpReceiveRequestEntityBody(
            queue, requestId, 0, buffer.data(),
            static_cast<ULONG>(buffer.size()), &received, nullptr);
        if (result == ERROR_SUCCESS || result == ERROR_MORE_DATA)
            body.append(reinterpret_cast<const char*>(buffer.data()), received);
    } while (result == ERROR_MORE_DATA);
    return body;
}

} // namespace

HttpApiServer::HttpApiServer(uint16_t port) : port_(port), impl_(new Impl()) {}

HttpApiServer::~HttpApiServer() {
    stop();
    delete impl_;
}

bool HttpApiServer::start() {
    error_.clear();
    if (port_ == 0) {
        error_ = "port must be between 1 and 65535";
        return false;
    }

    impl_->url = L"http://127.0.0.1:" + std::to_wstring(port_) + L"/";
    HTTPAPI_VERSION version{2, 0};
    ULONG result = HttpInitialize(version, HTTP_INITIALIZE_SERVER, nullptr);
    if (result != ERROR_SUCCESS) {
        error_ = "HttpInitialize failed: " + std::to_string(result);
        return false;
    }
    impl_->initialized = true;

    result = HttpCreateHttpHandle(&impl_->request_queue, 0);
    if (result == ERROR_SUCCESS)
        result = HttpAddUrl(impl_->request_queue, impl_->url.c_str(), nullptr);
    if (result != ERROR_SUCCESS) {
        error_ = "HTTP server setup failed: " + std::to_string(result);
        stop();
        return false;
    }
    llama_loaded_ = llama_.load("");
    return true;
}

void HttpApiServer::run() {
    if (!impl_->request_queue) return;

    std::vector<BYTE> buffer(64 * 1024);
    while (impl_->request_queue) {
        ULONG received = 0;
        const ULONG result = HttpReceiveHttpRequest(
            impl_->request_queue, HTTP_NULL_ID, HTTP_RECEIVE_REQUEST_FLAG_COPY_BODY,
            reinterpret_cast<HTTP_REQUEST*>(buffer.data()),
            static_cast<ULONG>(buffer.size()), &received, nullptr);
        if (result == ERROR_OPERATION_ABORTED || !impl_->request_queue) break;
        if (result != ERROR_SUCCESS) {
            error_ = "HttpReceiveHttpRequest failed: " + std::to_string(result);
            break;
        }

        const auto* request = reinterpret_cast<const HTTP_REQUEST*>(buffer.data());
        const std::wstring path(request->CookedUrl.pAbsPath
                                    ? request->CookedUrl.pAbsPath
                                    : L"",
                                request->CookedUrl.AbsPathLength / sizeof(wchar_t));
        std::string response;
        USHORT status = 404;
        if (request->Verb == HttpVerbGET && path == L"/health") {
            status = 200;
            response = "{\"status\":\"ok\"}";
        } else if (request->Verb == HttpVerbGET &&
                   (path == L"/v1/wwa/templates" ||
                    path == L"/v1/wwa/kits" ||
                    path == L"/v1/wwa/demos" ||
                    path == L"/v1/wwa/components")) {
            using Creator = Kuhul::Runtime::KuhulAppCreator;
            response = "{\"items\":[";
            if (path == L"/v1/wwa/templates") {
                for (size_t i = 0; i < Creator::templates().size(); ++i) {
                    const auto& item = Creator::templates()[i];
                    response += "{\"id\":\"" + jsonEscape(item.id) +
                                "\",\"description\":\"" + jsonEscape(item.description) + "\"}";
                    if (i + 1 < Creator::templates().size()) response += ",";
                }
            } else if (path == L"/v1/wwa/kits") {
                for (size_t i = 0; i < Creator::designKits().size(); ++i) {
                    const auto& item = Creator::designKits()[i];
                    response += "{\"id\":\"" + jsonEscape(item.id) +
                                "\",\"description\":\"" + jsonEscape(item.description) + "\"}";
                    if (i + 1 < Creator::designKits().size()) response += ",";
                }
            } else if (path == L"/v1/wwa/demos") {
                for (size_t i = 0; i < Creator::demos().size(); ++i) {
                    const auto& item = Creator::demos()[i];
                    response += "{\"id\":\"" + jsonEscape(item.id) +
                                "\",\"description\":\"" + jsonEscape(item.description) + "\"}";
                    if (i + 1 < Creator::demos().size()) response += ",";
                }
            } else {
                for (size_t i = 0; i < Creator::components().size(); ++i) {
                    const auto& item = Creator::components()[i];
                    response += "{\"id\":\"" + jsonEscape(item.id) +
                                "\",\"category\":\"" + jsonEscape(item.category) +
                                "\",\"description\":\"" + jsonEscape(item.description) + "\"}";
                    if (i + 1 < Creator::components().size()) response += ",";
                }
            }
            response += "]}";
            status = 200;
        } else if (request->Verb == HttpVerbGET &&
                   path == L"/v1/wwa/status") {
            const Kuhul::Runtime::KuhulAppCreator creator;
            const auto wwa = creator.inspect();
            status = wwa.ready() ? 200 : 503;
            response = "{\"status\":\"" +
                       std::string(wwa.ready() ? "ready" : "unavailable") +
                       "\",\"wwa_ext\":" +
                       std::string(wwa.extension_present ? "true" : "false") +
                       ",\"wwa_api\":" +
                       std::string(wwa.api_present ? "true" : "false") +
                       ",\"wwa_host\":" +
                       std::string(wwa.host_present ? "true" : "false");
            if (!wwa.error.empty())
                response += ",\"error\":\"" + jsonEscape(wwa.error) + "\"";
            response += "}";
        } else if (request->Verb == HttpVerbPOST &&
                   path == L"/v1/wwa/apps") {
            const std::string body = requestBody(
                request, impl_->request_queue, request->RequestId);
            JsonRequest json(body);
            std::string parent;
            std::string name;
            std::string templateId = "basic";
            std::string designKit = "default";
            std::string serverHost = "127.0.0.1";
            uint32_t serverPort = 7431;
            std::string modelProvider;
            std::string deploymentEnv;
            std::string endpointEnv;
            std::string apiKeyEnv;
            if (!json.stringField("parent", parent) ||
                !json.stringField("name", name)) {
                status = 400;
                response = statusJson("error", "parent and name are required");
            } else {
                json.stringField("template", templateId);
                json.stringField("design_kit", designKit);
                json.stringField("server_host", serverHost);
                json.uintField("server_port", serverPort);
                json.stringField("model_provider", modelProvider);
                json.stringField("deployment_env", deploymentEnv);
                json.stringField("endpoint_env", endpointEnv);
                json.stringField("api_key_env", apiKeyEnv);
                const Kuhul::Runtime::KuhulAppCreator creator;
                std::filesystem::path appRoot;
                std::string error;
                if (serverPort > 65535 ||
                    !creator.create(parent, name, appRoot, error, templateId,
                                    designKit, serverHost,
                                    static_cast<uint16_t>(serverPort),
                                    modelProvider, deploymentEnv,
                                    endpointEnv, apiKeyEnv)) {
                    status = 400;
                    response = statusJson("error", error);
                } else {
                    status = 201;
                    response = "{\"status\":\"created\",\"path\":\"" +
                               jsonEscape(appRoot.string()) + "\"}";
                }
            }
        } else if (request->Verb == HttpVerbGET && path == L"/v1/providers") {
            status = 200;
            response = "{\"providers\":[{\"id\":\"llama\",\"available\":" +
                       std::string(llama_loaded_ ? "true" : "false") + "}]}";
        } else if (request->Verb == HttpVerbPOST &&
                   (path == L"/v1/infer/llama" ||
                    path == L"/v1/chat/completions")) {
            const std::string body = requestBody(
                request, impl_->request_queue, request->RequestId);
            JsonRequest json(body);
            std::string model;
            std::string prompt;
            uint32_t tokens = 32;
            if (!json.stringField("model", model) ||
                !json.stringField("prompt", prompt)) {
                status = 400;
                response = statusJson("error", "model and prompt are required");
            } else if (json.uintField("tokens", tokens) && tokens == 0) {
                status = 400;
                response = statusJson("error", "tokens must be greater than zero");
            } else if (!llama_loaded_ && !(llama_loaded_ = llama_.load(""))) {
                status = 503;
                response = statusJson("error", llama_.error());
            } else {
                std::string output;
                if (!llama_.generate(model, prompt, tokens, output)) {
                    status = 500;
                    response = statusJson("error", llama_.error());
                } else {
                    status = 200;
                    if (path == L"/v1/chat/completions") {
                        response = "{\"model\":\"" + jsonEscape(model) +
                                   "\",\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"" +
                                   jsonEscape(output) + "\"}}]}";
                    } else {
                        response = "{\"model\":\"" + jsonEscape(model) +
                                   "\",\"output\":\"" + jsonEscape(output) + "\"}";
                    }
                }
            }
        } else {
            response = statusJson("error", "route not found");
        }

        HTTP_RESPONSE httpResponse{};
        httpResponse.StatusCode = status;
        httpResponse.pReason = status == 200 ? "OK" :
                               status == 201 ? "Created" :
                               status == 400 ? "Bad Request" :
                               status == 404 ? "Not Found" :
                               status == 503 ? "Service Unavailable" :
                               "Internal Server Error";
        httpResponse.ReasonLength = static_cast<USHORT>(
            std::strlen(httpResponse.pReason));
        httpResponse.pEntityChunks = nullptr;
        HTTP_DATA_CHUNK chunk{};
        chunk.DataChunkType = HttpDataChunkFromMemory;
        chunk.FromMemory.pBuffer = const_cast<char*>(response.data());
        chunk.FromMemory.BufferLength = static_cast<ULONG>(response.size());
        httpResponse.EntityChunkCount = 1;
        httpResponse.pEntityChunks = &chunk;
        HTTP_UNKNOWN_HEADER contentType{
            static_cast<USHORT>(sizeof("Content-Type") - 1),
            static_cast<USHORT>(sizeof("application/json") - 1),
            "Content-Type",
            "application/json"
        };
        httpResponse.Headers.pUnknownHeaders = &contentType;
        httpResponse.Headers.UnknownHeaderCount = 1;
        const ULONG sendResult = HttpSendHttpResponse(
            impl_->request_queue, request->RequestId, 0, &httpResponse,
            nullptr, nullptr, nullptr, 0, nullptr, nullptr);
        if (sendResult != ERROR_SUCCESS && sendResult != ERROR_OPERATION_ABORTED) {
            error_ = "HttpSendHttpResponse failed: " + std::to_string(sendResult);
        }
    }
}

void HttpApiServer::stop() {
    if (!impl_) return;
    if (impl_->request_queue) {
        HttpShutdownRequestQueue(impl_->request_queue);
        CloseHandle(impl_->request_queue);
        impl_->request_queue = nullptr;
    }
    if (impl_->initialized) {
        HttpTerminate(HTTP_INITIALIZE_SERVER, nullptr);
        impl_->initialized = false;
    }
}
