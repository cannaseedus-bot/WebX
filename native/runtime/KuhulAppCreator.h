#pragma once

#include <filesystem>
#include <string>
#include <cstdint>
#include <vector>

namespace Kuhul::Runtime {

struct WwaRuntimePaths {
    std::filesystem::path wwaExt =
        LR"(C:\Windows\System32\WwaExt.dll)";
    std::filesystem::path wwaApi =
        LR"(C:\Windows\System32\WwaApi.dll)";
    std::filesystem::path host =
        LR"(C:\Users\canna\.NNC-K\bin\v3.5.0-WebX\bin\WWAHost.exe)";
};

struct WwaRuntimeStatus {
    bool extension_present = false;
    bool api_present = false;
    bool host_present = false;
    std::string error;

    bool ready() const {
        return extension_present && api_present && host_present;
    }
};

struct WwaTemplate {
    std::string id;
    std::string description;
};

struct WwaDesignKit {
    std::string id;
    std::string description;
    std::string css;
};

struct WwaDemo {
    std::string id;
    std::string description;
};

struct WwaComponent {
    std::string id;
    std::string description;
    std::string category;
};

class KuhulAppCreator {
public:
    explicit KuhulAppCreator(WwaRuntimePaths paths = {});

    WwaRuntimeStatus inspect() const;
    static const std::vector<WwaTemplate>& templates();
    static const std::vector<WwaDesignKit>& designKits();
    static const std::vector<WwaDemo>& demos();
    static const std::vector<WwaComponent>& components();
    bool create(const std::filesystem::path& parent,
                const std::string& name,
                std::filesystem::path& appRoot,
                std::string& error,
                const std::string& templateId = "basic",
                const std::string& designKitId = "default",
                const std::string& serverHost = "127.0.0.1",
                uint16_t serverPort = 7431,
                const std::string& modelProvider = {},
                const std::string& deploymentEnv = {},
                const std::string& endpointEnv = {},
                const std::string& apiKeyEnv = {}) const;
    bool launch(const std::filesystem::path& appRoot, std::string& error) const;
    const WwaRuntimePaths& paths() const { return paths_; }

private:
    WwaRuntimePaths paths_;
};

} // namespace Kuhul::Runtime
