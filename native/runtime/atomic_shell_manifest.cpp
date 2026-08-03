#include "atomic_shell_manifest.h"

#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <sstream>

namespace Kuhul::Runtime {

namespace {

std::string jsonString(const std::string& json, const char* key) {
    const std::string marker = std::string("\"") + key + "\"";
    const size_t keyPos = json.find(marker);
    if (keyPos == std::string::npos) return {};
    const size_t colon = json.find(':', keyPos + marker.size());
    if (colon == std::string::npos) return {};
    const size_t quote = json.find('"', colon + 1);
    if (quote == std::string::npos) return {};
    const size_t end = json.find('"', quote + 1);
    return end == std::string::npos
        ? std::string{}
        : json.substr(quote + 1, end - quote - 1);
}

bool jsonBool(const std::string& json, const char* key, bool& value) {
    const std::string marker = std::string("\"") + key + "\"";
    const size_t keyPos = json.find(marker);
    if (keyPos == std::string::npos) return false;
    const size_t colon = json.find(':', keyPos + marker.size());
    if (colon == std::string::npos) return false;
    const size_t truePos = json.find("true", colon + 1);
    const size_t falsePos = json.find("false", colon + 1);
    if (truePos == std::string::npos &&
        falsePos == std::string::npos) return false;
    if (truePos != std::string::npos &&
        (falsePos == std::string::npos || truePos < falsePos)) {
        value = true;
    } else {
        value = false;
    }
    return true;
}

bool validBlock(const std::string& block) {
    return block == "HEADER" || block == "MENU" || block == "BODY" ||
           block == "FOOTER" || block == "GRID" || block == "FEED" ||
           block == "FRAME" || block == "BUTTON" || block == "IMAGE" ||
           block == "VIDEO" || block == "TEXT" || block == "INPUT" ||
           block == "CARD" || block == "PANEL" || block == "GAME";
}

std::vector<std::string> jsonStringArray(
    const std::string& json, const char* key) {
    std::vector<std::string> values;
    const std::string marker = std::string("\"") + key + "\"";
    const size_t keyPos = json.find(marker);
    if (keyPos == std::string::npos) return values;
    const size_t open = json.find('[', keyPos + marker.size());
    const size_t close = json.find(']', open == std::string::npos ? open : open + 1);
    if (open == std::string::npos || close == std::string::npos) return values;
    size_t cursor = open;
    while ((cursor = json.find('"', cursor + 1)) < close) {
        const size_t end = json.find('"', cursor + 1);
        if (end == std::string::npos || end > close) break;
        values.push_back(json.substr(cursor + 1, end - cursor - 1));
        cursor = end;
    }
    return values;
}

bool validSchemaRoute(const std::string& schema) {
    return schema.rfind("atomics://", 0) == 0 ||
           schema.rfind("hash://", 0) == 0 ||
           schema.rfind("cache://", 0) == 0;
}

std::filesystem::path feedArtifactPath(const std::string& manifestPath) {
    if (const char* configured = std::getenv("KUHUL_XCFE_WASM")) {
        if (*configured != '\0' &&
            std::filesystem::is_regular_file(configured)) {
            return configured;
        }
        return {};
    }
    const std::filesystem::path relative =
        "tree-sitter-xcfe/tree-sitter-xcfe.wasm";
    if (std::filesystem::exists(relative)) return relative;
    std::filesystem::path root = std::filesystem::path(manifestPath).parent_path();
    for (int i = 0; i < 4 && !root.empty(); ++i) {
        const auto candidate = root / relative;
        if (std::filesystem::exists(candidate)) return candidate;
        root = root.parent_path();
    }
    return {};
}

bool validWasmArtifact(const std::filesystem::path& path) {
    if (!std::filesystem::is_regular_file(path)) return false;
    std::ifstream file(path, std::ios::binary);
    unsigned char header[8] = {};
    file.read(reinterpret_cast<char*>(header), sizeof(header));
    return file.gcount() == static_cast<std::streamsize>(sizeof(header)) &&
           header[0] == 0x00 && header[1] == 0x61 &&
           header[2] == 0x73 && header[3] == 0x6d &&
           header[4] == 0x01;
}

} // namespace

bool AtomicShellManifestLoader::load(
    const std::string& path, AtomicShellManifest& manifest) {
    error_.clear();
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        error_ = "atomic_manifest_unreadable";
        return false;
    }
    std::ostringstream contents;
    contents << file.rdbuf();
    const std::string json = contents.str();
    manifest.id = jsonString(json, "id");
    manifest.block = jsonString(json, "block");
    manifest.version = jsonString(json, "version");
    manifest.schema = jsonString(json, "$schema");
    const std::string backend = jsonString(json, "backend");
    if (!backend.empty()) manifest.backend = backend;
    manifest.feedParser = jsonString(json, "feed_parser");
    manifest.assetFormat = jsonString(json, "asset_format");
    manifest.assetUri = jsonString(json, "asset_uri");
    manifest.assetUri = jsonString(json, "asset_uri");
    manifest.blocks = jsonStringArray(json, "blocks");
    if (manifest.id.empty() || manifest.block.empty() ||
        manifest.version.empty() || manifest.schema.empty()) {
        error_ = "atomic_manifest_required_field_missing";
        return false;
    }
    if (!validBlock(manifest.block)) {
        error_ = "atomic_manifest_invalid_block";
        return false;
    }
    if (manifest.blocks.empty())
        manifest.blocks.push_back(manifest.block);
    for (const auto& block : manifest.blocks) {
        if (!validBlock(block)) {
            error_ = "atomic_manifest_invalid_composed_block";
            return false;
        }
    }
    const bool hasFeed = std::find(
        manifest.blocks.begin(), manifest.blocks.end(), "FEED") !=
        manifest.blocks.end();
    if (hasFeed && manifest.feedParser != "xcfe://tree-sitter-wasm") {
        error_ = "atomic_manifest_feed_parser_required";
        return false;
    }
    if (hasFeed) {
        const auto artifact = feedArtifactPath(path);
        if (artifact.empty()) {
            error_ = "atomic_manifest_feed_parser_artifact_missing";
            return false;
        }
        if (!validWasmArtifact(artifact)) {
            error_ = "atomic_manifest_feed_parser_artifact_invalid";
            return false;
        }
        manifest.feedParserArtifact = artifact.string();
    }
    if (!validSchemaRoute(manifest.schema)) {
        error_ = "atomic_manifest_external_schema_route";
        return false;
    }
    if (manifest.backend != "terminal" && manifest.backend != "opengl") {
        error_ = "atomic_manifest_invalid_backend";
        return false;
    }
    if (!manifest.assetFormat.empty() &&
        manifest.assetFormat != "gltf" && manifest.assetFormat != "glb" &&
        manifest.assetFormat != "obj" && manifest.assetFormat != "stl") {
        error_ = "atomic_manifest_invalid_asset_format";
        return false;
    }
    if (!manifest.assetFormat.empty() && manifest.backend != "opengl") {
        error_ = "atomic_manifest_3d_asset_requires_opengl";
        return false;
    }
    if (!jsonBool(json, "execution_gated", manifest.executionGated)) {
        error_ = "atomic_manifest_execution_gate_missing";
        return false;
    }
    if (!manifest.executionGated) {
        error_ = "atomic_manifest_execution_not_gated";
        return false;
    }
    return true;
}

} // namespace Kuhul::Runtime
