/**
 * WebX Compute Engine — Simplified GPU Compute with Semantic Field Graph
 * 
 * This is a STREAMLINED version focused purely on GPU compute operations
 * using semantic field graphs. NOT the full NNC-K system.
 * 
 * Features:
 * - GPU compute shader dispatch
 * - Semantic field graph operations
 * - Pressure-based scheduling (simplified)
 * - Basic Micronaut hosting
 * - BOSS layer orchestration (1-3 layers)
 * 
 * @version 1.0.0
 * @author WebX Team
 * @license MIT
 */

#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <chrono>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cctype>
#include <sstream>
#include <cstring>
#include <windows.h>

// ─────────────────────────────────────────────────────────────
// Field Graph Structures (Simplified)
// ──────────────────────────────────────────────────────────────

namespace WebX {

/**
 * Field Confidence Profile.
 * Tracks how confidence evolves through research, discussion, and validation.
 */
struct FieldConfidenceProfile {
    float semantic_correctness = 0.0f;     // SM/VM validation score
    float domain_alignment = 0.0f;         // GM domain match
    float constraint_satisfaction = 0.0f;  // VM constraint score
    float token_coherence = 0.0f;          // MM working-set coherence
    float aggregate = 0.0f;                // weighted final confidence
    uint32_t research_cycles = 0;          // number of research/discussion passes
    uint32_t validation_cycles = 0;        // number of cross-expert validations
    
    static constexpr float SEMANTIC_WEIGHT = 0.35f;
    static constexpr float DOMAIN_WEIGHT = 0.25f;
    static constexpr float CONSTRAINT_WEIGHT = 0.25f;
    static constexpr float COHERENCE_WEIGHT = 0.15f;
};

/**
 * Token — Atomic execution element
 * Simplified from full NNC-K Token
 */
struct Token {
    int position = 0;
    float activation = 0.0f;
    float confidence = 1.0f;
    float pressure = 0.5f;
};

/**
 * Semantic Field — Core execution unit
 * Simplified from full NNC-K SCX Field
 */
struct Field {
    std::string id;
    std::string domain;          // "Compute", "Attention", "Memory"
    
    // Pressure-based scheduling
    float pressure = 0.5f;       // 0.0 - 1.0
    float coherence = 1.0f;      // 0.0 - 1.0
    
    // GPU execution
    int card_count = 0;
    int token_count = 0;
    void* gpu_data = nullptr;
    
    // Working set tokens for semantic validation (MM Micronaut)
    std::vector<Token> tokens;
    
    // Confidence profile from Micronaut expert validation
    FieldConfidenceProfile confidence_profile;
    
    // Metrics
    double latency_ms = 0.0;
    uint64_t execution_count = 0;
};

/**
 * Card — Schedulable partition within a Field
 * Simplified from full NNC-K Card
 */
struct Card {
    std::string id;
    Field* field = nullptr;
    
    int token_start = 0;
    int token_count = 0;
    
    float pressure = 0.5f;
    void* gpu_buffer = nullptr;
};

// ─────────────────────────────────────────────────────────────
// Semantic Validation (Micronaut Expert Shader Layer)
// ──────────────────────────────────────────────────────────────

/**
 * Semantic validation factors
 * Each Micronaut expert contributes one factor to confidence.
 */
enum class SemanticFactor {
    SemanticCorrectness,   // SM / VM: does the operation make semantic sense?
    DomainAlignment,       // GM: does the field belong to the right domain?
    ConstraintSatisfaction,// VM: are constraints satisfied?
    TokenCoherence         // MM: is the working set coherent?
};

/**
 * Result of semantic validation before GPU dispatch.
 * Produced by Micronaut expert shaders.
 */
struct SemanticValidationResult {
    bool passed = true;
    float confidence = 1.0f;                 // [0.0, 1.0]
    std::string domain;                      // validated domain
    std::string shader_name;                 // selected shader
    std::vector<std::string> diagnostics;    // warnings/errors
    std::unordered_map<SemanticFactor, float> factor_scores;

    SemanticValidationResult() {
        factor_scores[SemanticFactor::SemanticCorrectness] = 1.0f;
        factor_scores[SemanticFactor::DomainAlignment] = 1.0f;
        factor_scores[SemanticFactor::ConstraintSatisfaction] = 1.0f;
        factor_scores[SemanticFactor::TokenCoherence] = 1.0f;
    }

    float getFactorScore(SemanticFactor factor) const {
        auto it = factor_scores.find(factor);
        return it != factor_scores.end() ? it->second : 0.0f;
    }
};

/**
 * Geometry → Shader → Execution Class alignment.
 * Mirrors native/GEOMETRY_SHADER_ALIGNMENT.md.
 */
struct GeometryShaderMapping {
    std::string geometry;
    std::string topology;
    std::string shader;
    std::string exec_class;
    std::string domain;
};

class GeometryShaderAlignment {
public:
    static const GeometryShaderMapping* lookup(const std::string& geometry) {
        static const GeometryShaderMapping table[] = {
            { "Box",           "FieldTopology::Box",           "neural_layer.klsl",          "Attention",  "Compute" },
            { "Cone",          "FieldTopology::Cone",          "xvm_fused_qkv_attention.hlsl", "Attention",  "Compute" },
            { "Cylinder",      "FieldTopology::Cylinder",      "orchestrate.hlsl",           "Fiber",      "Execution" },
            { "Torus",         "FieldTopology::Torus",         "pi_field.wgsl",              "Fiber",      "Memory" },
            { "GeoSphere",     "FieldTopology::GeoSphere",     "xvm_compute.hlsl",           "Wave",       "Geometry" },
            { "Icosahedron",   "FieldTopology::Icosahedron",   "FibonacciCS.hlsl",           "Scalar",     "Geometry" },
            { "Dodecahedron",  "FieldTopology::Dodecahedron",  "scxq2_infer_layer.hlsl",     "Scalar",     "Geometry" },
            { "Tetrahedron",   "FieldTopology::Tetrahedron",   "kuhul_fold_compute.hlsl",    "Fiber",      "Execution" },
            { "Octahedron",    "FieldTopology::Octahedron",    "int4_matmul.hlsl",           "SIMD",       "Geometry" },
            { "Teapot",        "FieldTopology::Teapot",        "evolution.hlsl",             "Evolution",  "Evolution" },
            { "OpticalField",  "FieldTopology::OpticalField",  "optical_wave.klsl",          "Optical",    "Vision" }
        };
        static constexpr size_t count = sizeof(table) / sizeof(table[0]);

        for (size_t i = 0; i < count; ++i) {
            if (table[i].geometry == geometry) {
                return &table[i];
            }
        }
        return nullptr;
    }

    static std::string resolveShaderForDomain(const std::string& domain) {
        if (domain == "Compute" || domain == "Attention") return "neural_layer.klsl";
        if (domain == "Memory") return "pi_field.wgsl";
        if (domain == "Geometry") return "xvm_compute.hlsl";
        if (domain == "Execution") return "kuhul_fold_compute.hlsl";
        if (domain == "Evolution") return "evolution.hlsl";
        if (domain == "Vision") return "optical_wave.klsl";
        return "webx_compute";
    }
};

/**
 * Semantic Validator — Micronaut expert shader proxy.
 * Validates semantics before GPU dispatch.
 */
class SemanticValidator {
public:
    static SemanticValidationResult validate(
        const Field& field,
        const std::vector<Token>& tokens,
        const std::string& requested_shader = ""
    ) {
        SemanticValidationResult result;
        result.domain = field.domain;

        // SM/VM: Semantic correctness — does the field have content?
        if (field.token_count == 0) {
            result.factor_scores[SemanticFactor::SemanticCorrectness] = 0.0f;
            result.diagnostics.push_back("SM: field has zero tokens");
        } else if (field.pressure < 0.05f) {
            result.factor_scores[SemanticFactor::SemanticCorrectness] = 0.3f;
            result.diagnostics.push_back("SM: field pressure too low to execute");
        } else {
            result.factor_scores[SemanticFactor::SemanticCorrectness] = 0.95f;
        }

        // GM: Domain alignment — map domain to geometry/shader alignment
        std::string geometryHint = inferGeometryFromDomain(field.domain);
        auto mapping = GeometryShaderAlignment::lookup(geometryHint);
        if (mapping && mapping->domain == field.domain) {
            result.factor_scores[SemanticFactor::DomainAlignment] = 0.95f;
            result.shader_name = mapping->shader;
        } else {
            result.factor_scores[SemanticFactor::DomainAlignment] = 0.7f;
            result.diagnostics.push_back("GM: domain " + field.domain + " has no canonical geometry mapping");
            result.shader_name = GeometryShaderAlignment::resolveShaderForDomain(field.domain);
        }

        // VM: Constraint satisfaction — simple token-level check
        float token_confidence_sum = 0.0f;
        for (const auto& token : tokens) {
            token_confidence_sum += token.confidence;
        }
        float avg_confidence = tokens.empty() ? 1.0f : token_confidence_sum / tokens.size();
        result.factor_scores[SemanticFactor::ConstraintSatisfaction] =
            std::clamp(avg_confidence, 0.0f, 1.0f);

        // MM: Token coherence — inverse pressure variance
        float mean_pressure = tokens.empty() ? 0.5f :
            std::accumulate(tokens.begin(), tokens.end(), 0.0f,
                [](float s, const Token& t) { return s + t.pressure; }) / tokens.size();
        float variance = 0.0f;
        for (const auto& token : tokens) {
            float d = token.pressure - mean_pressure;
            variance += d * d;
        }
        variance = tokens.empty() ? 0.0f : variance / tokens.size();
        result.factor_scores[SemanticFactor::TokenCoherence] =
            std::clamp(1.0f - variance * 5.0f, 0.0f, 1.0f);

        // Override shader if requested explicitly
        if (!requested_shader.empty()) {
            result.shader_name = requested_shader;
        }

        // Compute aggregate confidence
        float total = 0.0f;
        for (const auto& [factor, score] : result.factor_scores) {
            total += score;
        }
        result.confidence = total / static_cast<float>(result.factor_scores.size());
        result.passed = result.confidence >= 0.4f;

        return result;
    }

private:
    static std::string inferGeometryFromDomain(const std::string& domain) {
        if (domain == "Compute" || domain == "Attention") return "Box";
        if (domain == "Memory") return "Torus";
        if (domain == "Geometry") return "GeoSphere";
        if (domain == "Execution") return "Tetrahedron";
        if (domain == "Evolution") return "Teapot";
        if (domain == "Vision") return "OpticalField";
        return "Box";
    }
};

// ─────────────────────────────────────────────────────────────
// GPU Compute Interface (Abstracted)
// ──────────────────────────────────────────────────────────────

enum class GPUBackend {
    DirectML,      // Cross-vendor (Intel, AMD, NVIDIA)
    CUDA,          // NVIDIA only
    CPU            // Fallback
};

struct GPUConfig {
    GPUBackend backend = GPUBackend::DirectML;
    int device_id = 0;
    size_t memory_limit_mb = 4096;
    bool use_fp16 = true;
};

class GPUComputeEngine {
private:
    GPUConfig config_;
    bool initialized_ = false;
    
    // Device resources (simplified)
    void* device_ = nullptr;
    void* context_ = nullptr;
    
    // Last semantic validation result (from Micronaut experts)
    SemanticValidationResult last_validation_;
    
public:
    GPUComputeEngine() = default;
    ~GPUComputeEngine() { shutdown(); }
    
    bool initialize(const GPUConfig& config) {
        config_ = config;
        
        // Initialize GPU backend (simplified)
        switch (config.backend) {
            case GPUBackend::DirectML:
                return initDirectML();
            case GPUBackend::CUDA:
                return initCUDA();
            case GPUBackend::CPU:
                return initCPU();
            default:
                return false;
        }
    }
    
    void shutdown() {
        if (device_) {
            releaseDevice(device_);
            device_ = nullptr;
        }
        initialized_ = false;
    }
    
    bool isInitialized() const { return initialized_; }
    
    // Execute compute shader on field
    // tokens: working set for semantic validation (Micronaut layer)
    bool execute(Field& field, const std::string& shader_name, const std::vector<Token>& tokens = {}) {
        if (!initialized_) {
            std::cerr << "GPU not initialized" << std::endl;
            return false;
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // ─────────────────────────────────────────────────────
        // Pop/Wo: Semantic validation before GPU dispatch
        // Micronaut expert shaders validate semantics, domain, constraints, coherence
        // ─────────────────────────────────────────────────────
        const std::vector<Token>& validation_tokens = tokens.empty() ? field.tokens : tokens;
        last_validation_ = SemanticValidator::validate(field, validation_tokens, shader_name);
        
        if (!last_validation_.passed) {
            std::cerr << "Semantic validation failed for field " << field.id 
                      << " (confidence: " << last_validation_.confidence << ")" << std::endl;
            for (const auto& diagnostic : last_validation_.diagnostics) {
                std::cerr << "  - " << diagnostic << std::endl;
            }
            field.pressure = std::max(0.0f, field.pressure - 0.1f);
            return false;
        }
        
        // Use Micronaut-selected shader if no explicit shader requested
        const std::string& dispatch_shader = shader_name.empty()
            ? last_validation_.shader_name
            : shader_name;
        
        // Dispatch compute shader
        bool success = dispatchShader(field, dispatch_shader);
        
        auto end = std::chrono::high_resolution_clock::now();
        field.latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
        field.execution_count++;
        
        // Update pressure based on execution and semantic confidence
        if (success) {
            float pressure_delta = 0.01f * last_validation_.confidence;
            field.pressure = std::min(1.0f, field.pressure + pressure_delta);
        } else {
            field.pressure = std::max(0.0f, field.pressure - 0.05f);
        }
        
        return success;
    }
    
    // Get last semantic validation result
    const SemanticValidationResult& getLastValidation() const {
        return last_validation_;
    }
    
    // Get GPU memory usage
    size_t getMemoryUsage() const {
        // Placeholder - would query actual GPU memory
        return 0;
    }
    
private:
    bool initDirectML() {
        // Initialize DirectML (simplified)
        // In production: Create DML device, command queue, etc.
        std::cout << "Initializing DirectML..." << std::endl;
        device_ = reinterpret_cast<void*>(0x1);  // Placeholder
        initialized_ = true;
        return true;
    }
    
    bool initCUDA() {
        // Initialize CUDA (simplified)
        // In production: cudaSetDevice, create streams, etc.
        std::cout << "Initializing CUDA..." << std::endl;
        device_ = reinterpret_cast<void*>(0x2);  // Placeholder
        initialized_ = true;
        return true;
    }
    
    bool initCPU() {
        // CPU fallback
        std::cout << "Using CPU fallback..." << std::endl;
        device_ = reinterpret_cast<void*>(0x3);  // Placeholder
        initialized_ = true;
        return true;
    }
    
    void releaseDevice(void* device) {
        // Release GPU resources
        device = nullptr;
    }
    
    bool dispatchShader(Field& field, const std::string& shader_name) {
        // Dispatch compute shader (simplified)
        // In production: Bind resources, dispatch threads, read back
        
        std::cout << "Dispatching shader: " << shader_name 
                  << " on field " << field.id 
                  << " (" << field.card_count << " cards, " 
                  << field.token_count << " tokens)" << std::endl;
        
        // Simulate execution
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        return true;
    }
};

// ─────────────────────────────────────────────────────────────
// Native Provider Abstraction
// Providers are platform-specific libraries used by the Micronaut Forge.
// They are NOT Micronauts and they are NOT folds.
// ──────────────────────────────────────────────────────────────

enum class ProviderCapability {
    ShaderCompiler,
    TensorCompute,
    KernelCompute,
    NativeCompiler,
    GraphicsAPI,
    SoftwareRasterizer,
    JITBackend,
    Cache,
    Debug
};

struct Provider {
    std::string id;
    ProviderCapability capability;
    std::string library;
    std::vector<std::string> outputs;
    int priority = 1;
    bool available = false;
};

class ProviderManager {
private:
    std::vector<Provider> providers_;
    std::vector<HMODULE> resident_handles_;

public:
    ProviderManager() {
        registerDefaults();
    }

    void registerDefaults() {
        providers_.push_back({"d3dcompiler", ProviderCapability::ShaderCompiler,
                              "D3DCompiler_47.dll", {"hlsl_bytecode", "cso"}, 1, false});
        providers_.push_back({"shader_cache", ProviderCapability::Cache,
                              "D3DSCache.dll", {"compiled_shaders", "artifacts"}, 1, false});
        providers_.push_back({"directml", ProviderCapability::TensorCompute,
                              "DirectML.dll", {"tensor_graph", "inference_kernel"}, 1, false});
        providers_.push_back({"xcfe_directml", ProviderCapability::TensorCompute,
                              "ggml-xcfe.dll", {"ggml_backend", "mul_mat", "directml_gemm"}, 1, false});
        providers_.push_back({"directml_debug", ProviderCapability::Debug,
                              "DirectML.Debug.dll", {"diagnostics", "validation"}, 1, false});
        providers_.push_back({"opencl", ProviderCapability::KernelCompute,
                              "IntelOpenCL64.dll", {"opencl_kernel", "buffer"}, 2, false});
        providers_.push_back({"intel_opencl_icd", ProviderCapability::KernelCompute,
                              "Intel_OpenCL_ICD64.dll", {"opencl_icd"}, 2, false});
        providers_.push_back({"opencl_cpu_device", ProviderCapability::KernelCompute,
                              "ocl_cpu_cpu_device64.dll", {"cpu_opencl_device"}, 3, false});
        providers_.push_back({"opencl_cpu_runtime_impl", ProviderCapability::KernelCompute,
                              "ocl_cpu_IntelOpenCL64.dll", {"cpu_opencl_runtime"}, 3, false});
        providers_.push_back({"opencl_cpu_runtime", ProviderCapability::KernelCompute,
                              "ocl_cpu_intelocl64.dll", {"cpu_opencl_kernel", "buffer"}, 3, false});
        providers_.push_back({"opencl_cpu_backend", ProviderCapability::KernelCompute,
                              "ocl_cpu_OclCpuBackend64.dll", {"cpu_task_backend"}, 4, false});
        providers_.push_back({"opencl_cpu_executor", ProviderCapability::KernelCompute,
                              "ocl_cpu_task_executor64.dll", {"cpu_task_executor"}, 4, false});
        providers_.push_back({"opencl_cpu_tbb", ProviderCapability::KernelCompute,
                              "ocl_cpu_tbb64.dll", {"cpu_threading"}, 5, false});
        providers_.push_back({"opencl_cpu_clang", ProviderCapability::JITBackend,
                              "ocl_cpu_clang_compiler64.dll", {"opencl_cpu_ir"}, 3, false});
        providers_.push_back({"clang", ProviderCapability::NativeCompiler,
                              "common_clang64.dll", {"exe", "dll", "obj", "wasm", "llvm_ir"}, 1, false});
        providers_.push_back({"d3d12", ProviderCapability::GraphicsAPI,
                              "D3D12.dll", {"command_list", "resource", "pso"}, 1, false});
        providers_.push_back({"d3d11", ProviderCapability::GraphicsAPI,
                              "d3d11.dll", {"device", "context", "shader"}, 2, false});
        providers_.push_back({"intel_opengl_icd", ProviderCapability::GraphicsAPI,
                              "ig8icd64.dll", {"opengl_context", "shader"}, 3, false});
        providers_.push_back({"intel_opengl_legacy", ProviderCapability::GraphicsAPI,
                              "ig9icd64.dll", {"opengl_context", "shader"}, 4, false});
        providers_.push_back({"intel_opengl_legacy_75", ProviderCapability::GraphicsAPI,
                              "ig75icd64.dll", {"opengl_context", "shader"}, 5, false});
        providers_.push_back({"intel_d3d10_user", ProviderCapability::GraphicsAPI,
                              "igd10iumd64.dll", {"d3d10_device", "shader"}, 3, false});
        providers_.push_back({"intel_d3d11_video", ProviderCapability::GraphicsAPI,
                              "igd11dxva64.dll", {"video_decode"}, 3, false});
        providers_.push_back({"intel_d3d12_user", ProviderCapability::GraphicsAPI,
                              "igd12umd64.dll", {"d3d12_device", "shader"}, 3, false});
        providers_.push_back({"intel_graphics_jit", ProviderCapability::JITBackend,
                              "igc64.dll", {"jitted_kernels", "shader_ir"}, 2, false});
        providers_.push_back({"d3d10warp", ProviderCapability::SoftwareRasterizer,
                              "d3d10warp.dll", {"cpu_rasterized_surface"}, 4, false});
        detect();
    }

    void detect() {
        char driverRoot[4096] = {};
        const DWORD rootLength = GetEnvironmentVariableA(
            "KUHUL_DRIVER_ROOT", driverRoot, static_cast<DWORD>(sizeof(driverRoot)));
        const std::string root = rootLength > 0
            ? std::string(driverRoot, rootLength)
            : std::string();
        for (auto& provider : providers_) {
            HMODULE handle = LoadLibraryA(provider.library.c_str());
            if (!handle && !root.empty()) {
                const std::string path = root + "\\" + provider.library;
                handle = LoadLibraryA(path.c_str());
            }
            if (handle) {
                provider.available = true;
                // Intel user-mode graphics drivers can execute teardown code
                // that assumes the process remains graphics-initialized.
                if (provider.id.rfind("intel_", 0) == 0 ||
                    provider.id.rfind("opencl_cpu_", 0) == 0) {
                    resident_handles_.push_back(handle);
                } else {
                    FreeLibrary(handle);
                }
            }
        }
    }

    Provider* selectBest(ProviderCapability capability) {
        Provider* best = nullptr;
        for (auto& provider : providers_) {
            if (provider.capability == capability && provider.available) {
                if (!best || provider.priority < best->priority) {
                    best = &provider;
                }
            }
        }
        return best;
    }

    const std::vector<Provider>& getProviders() const { return providers_; }

    void printStatus() const {
        std::cout << "Provider Manager status:" << std::endl;
        for (const auto& provider : providers_) {
            std::cout << "  " << provider.id << " (" << provider.library << ") "
                      << (provider.available ? "available" : "not found") << std::endl;
        }
    }
};

// ─────────────────────────────────────────────────────────────
// Field Graph Manager
// ──────────────────────────────────────────────────────────────

class FieldGraph {
private:
    std::unordered_map<std::string, std::unique_ptr<Field>> fields_;
    std::unordered_map<std::string, std::unique_ptr<Card>> cards_;
    
    mutable std::mutex mutex_;
    
public:
    // Create field
    Field* createField(const std::string& id, const std::string& domain) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto field = std::make_unique<Field>();
        field->id = id;
        field->domain = domain;
        
        Field* ptr = field.get();
        fields_[id] = std::move(field);
        
        return ptr;
    }
    
    // Get field by ID
    Field* getField(const std::string& id) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = fields_.find(id);
        return it != fields_.end() ? it->second.get() : nullptr;
    }
    
    // Create card in field
    Card* createCard(Field* field, int token_start, int token_count) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto card = std::make_unique<Card>();
        card->id = field->id + "_card_" + std::to_string(field->card_count);
        card->field = field;
        card->token_start = token_start;
        card->token_count = token_count;
        
        field->card_count++;
        field->token_count += token_count;
        
        // Populate working set tokens for this card
        int base_position = static_cast<int>(field->tokens.size());
        for (int i = 0; i < token_count; ++i) {
            Token token;
            token.position = base_position + i;
            token.activation = 0.5f;
            token.confidence = 1.0f;
            token.pressure = field->pressure;
            field->tokens.push_back(token);
        }
        
        Card* ptr = card.get();
        cards_[card->id] = std::move(card);
        
        return ptr;
    }
    
    // Get high-pressure fields (for scheduling)
    std::vector<Field*> getHighPressureFields(float threshold = 0.7f) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::vector<Field*> result;
        for (auto& [id, field] : fields_) {
            if (field->pressure >= threshold) {
                result.push_back(field.get());
            }
        }
        
        // Sort by pressure (descending)
        std::sort(result.begin(), result.end(),
            [](const Field* a, const Field* b) {
                return a->pressure > b->pressure;
            });
        
        return result;
    }
    
    // Update pressure from execution results
    void updatePressure(const std::string& field_id, double reward) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = fields_.find(field_id);
        if (it != fields_.end()) {
            Field* field = it->second.get();
            
            // Simplified pressure update
            field->pressure += 0.1f * static_cast<float>(reward - field->pressure);
            field->pressure = std::clamp(field->pressure, 0.0f, 1.0f);
        }
    }
    
    // Get all fields
    const auto& getAllFields() const { return fields_; }
    
    // Get field count
    size_t getFieldCount() const { return fields_.size(); }
};

// ─────────────────────────────────────────────────────────────
// Micronaut Expert Shader Registry
// Each K'UHUL Micronaut type maps to exactly one expert shader.
// ──────────────────────────────────────────────────────────────

enum class MicronautType {
    // K'UHUL canonical native types
    MM,  // Memory Micronaut     → pi_field.wgsl
    RM,  // Reason Micronaut     → kuhul_fold_compute.hlsl
    PM,  // Parser Micronaut     → orchestrate.hlsl
    CM,  // Compiler Micronaut   → kuhul_fold_storage.hlsl
    SM,  // Shader Micronaut     → neural_layer.klsl
    XM,  // Execution Micronaut  → kuhul_fold_compute.hlsl
    FM,  // File Micronaut       → scxq2_infer_layer.hlsl
    DM,  // Data Micronaut       → xvm_compute.hlsl
    VM,  // Validation Micronaut → StabilizeCS.hlsl
    OM,  // Optical Micronaut    → optical_wave.klsl
    GM,  // Grammar Micronaut    → kuhul_fold_meta.hlsl
    TM,  // Training Micronaut   → gpt2_adam.hlsl
    NM,  // Network Micronaut    → neural_layer.klsl
    
    // WebX simplified aliases (kept for compatibility)
    Compute = SM,
    Attention = SM,
    MatrixMath = DM,
    Optical = OM
};

/**
 * Maps every Micronaut type to its canonical expert shader.
 * This registry is the source of truth for Micronaut → shader dispatch.
 */
class MicronautShaderRegistry {
public:
    static std::string getShader(MicronautType type) {
        switch (type) {
            case MicronautType::MM: return "pi_field.wgsl";
            case MicronautType::RM: return "kuhul_fold_compute.hlsl";
            case MicronautType::PM: return "orchestrate.hlsl";
            case MicronautType::CM: return "kuhul_fold_storage.hlsl";
            case MicronautType::SM: return "neural_layer.klsl";
            case MicronautType::XM: return "kuhul_fold_compute.hlsl";
            case MicronautType::FM: return "scxq2_infer_layer.hlsl";
            case MicronautType::DM: return "xvm_compute.hlsl";
            case MicronautType::VM: return "StabilizeCS.hlsl";
            case MicronautType::OM: return "optical_wave.klsl";
            case MicronautType::GM: return "kuhul_fold_meta.hlsl";
            case MicronautType::TM: return "gpt2_adam.hlsl";
            case MicronautType::NM: return "neural_layer.klsl";
            default: return "webx_compute";
        }
    }

    static std::string getTypeName(MicronautType type) {
        switch (type) {
            case MicronautType::MM: return "MM-Memory";
            case MicronautType::RM: return "RM-Reason";
            case MicronautType::PM: return "PM-Parser";
            case MicronautType::CM: return "CM-Compiler";
            case MicronautType::SM: return "SM-Shader";
            case MicronautType::XM: return "XM-Execution";
            case MicronautType::FM: return "FM-File";
            case MicronautType::DM: return "DM-Data";
            case MicronautType::VM: return "VM-Validation";
            case MicronautType::OM: return "OM-Optical";
            case MicronautType::GM: return "GM-Grammar";
            case MicronautType::TM: return "TM-Training";
            case MicronautType::NM: return "NM-Network";
            default: return "Unknown";
        }
    }
};

struct MicronautConfig {
    std::string id;
    MicronautType type;
    GPUBackend backend;
    std::string shader_path;
    std::vector<std::string> field_specs;
    std::vector<std::string> node_ids;
    float pressure = 0.5f;
    float confidence = 0.0f;
    std::string residency = "STREAMING";
    std::string provider = "directml";
    std::vector<std::string> history;
    int thread_groups_x = 16;
    int thread_groups_y = 16;
    int thread_groups_z = 1;
};

class Micronaut {
private:
    MicronautConfig config_;
    GPUComputeEngine* gpu_ = nullptr;
    FieldGraph* graph_ = nullptr;
    
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> execution_count_{0};
    
public:
    Micronaut(const MicronautConfig& config, 
              GPUComputeEngine* gpu,
              FieldGraph* graph)
        : config_(config), gpu_(gpu), graph_(graph) {}
    
    bool initialize() {
        running_ = true;
        if (config_.node_ids.empty()) config_.node_ids.push_back(config_.id);
        config_.history.push_back("initialized");
        std::cout << "Micronaut " << config_.id 
                  << " [" << MicronautShaderRegistry::getTypeName(config_.type) << "]"
                  << " initialized" << std::endl;
        return true;
    }
    
    void shutdown() {
        running_ = false;
        config_.history.push_back("shutdown");
    }
    
    // Execute on field using the Micronaut's expert shader
    bool execute(Field* field) {
        if (!running_ || !field) {
            return false;
        }
        
        // Resolve canonical expert shader for this Micronaut type
        std::string shader_name = !config_.shader_path.empty()
            ? config_.shader_path
            : MicronautShaderRegistry::getShader(config_.type);
        
        // Pass field working set tokens to GPU for semantic validation
        bool success = gpu_->execute(*field, shader_name, field->tokens);

        if (success) {
            execution_count_++;
            config_.pressure = field->pressure;
            config_.confidence = gpu_->getLastValidation().confidence;
            config_.residency = "HOT";
            config_.history.push_back("executed:" + field->id);
        } else {
            config_.history.push_back("failed:" + field->id);
        }
        
        return success;
    }
    
    // Get last semantic validation result from underlying GPU execution
    const SemanticValidationResult& getLastValidation() const {
        return gpu_->getLastValidation();
    }
    
    // Get metrics
    uint64_t getExecutionCount() const { return execution_count_; }
    const std::string& getId() const { return config_.id; }
    MicronautType getType() const { return config_.type; }
    const std::vector<std::string>& getFieldSpecs() const {
        return config_.field_specs;
    }
    const std::vector<std::string>& getNodeIds() const {
        return config_.node_ids;
    }
    float getPressure() const { return config_.pressure; }
    float getConfidence() const { return config_.confidence; }
    const std::string& getResidency() const { return config_.residency; }
    const std::string& getProvider() const { return config_.provider; }
    const std::vector<std::string>& getHistory() const {
        return config_.history;
    }
};

// ─────────────────────────────────────────────────────────────
// BOSS Layer Orchestration (Simplified 1-3 layers)
// ──────────────────────────────────────────────────────────────

class BOSSOrchestrator {
private:
    int layer_count_ = 1;  // 1-3 for WebX (simplified from NNC-K's 5)
    
    GPUComputeEngine* gpu_ = nullptr;
    FieldGraph* graph_ = nullptr;
    std::vector<std::unique_ptr<Micronaut>> micronauts_;
    
public:
    BOSSOrchestrator(int layers = 1) 
        : layer_count_(std::clamp(layers, 1, 3)) {}
    
    bool initialize(GPUComputeEngine* gpu, FieldGraph* graph) {
        gpu_ = gpu;
        graph_ = graph;
        
        std::cout << "BOSS Orchestrator initialized with " 
                  << layer_count_ << " layers" << std::endl;
        
        // Create default Micronauts based on layer count
        createMicronauts();
        
        return true;
    }
    
    // Execute field through BOSS layers
    bool executeField(Field* field) {
        if (!field) return false;
        
        std::cout << "BOSS: Executing field " << field->id 
                  << " (pressure: " << field->pressure << ")" << std::endl;
        
        // Layer 1: Model Execution (always active)
        if (!executeLayer1(field)) {
            return false;
        }
        
        // Layer 2: Routing & Load Balancing (if enabled)
        if (layer_count_ >= 2) {
            if (!executeLayer2(field)) {
                return false;
            }
        }
        
        // Layer 3: Caching & Optimization (if enabled)
        if (layer_count_ >= 3) {
            if (!executeLayer3(field)) {
                return false;
            }
        }
        
        return true;
    }
    
private:
    void createMicronauts() {
        // Create compute Micronaut
        MicronautConfig compute_config;
        compute_config.id = "webx_compute_001";
        compute_config.type = MicronautType::Compute;
        compute_config.backend = GPUBackend::DirectML;
        
        auto compute = std::make_unique<Micronaut>(compute_config, gpu_, graph_);
        compute->initialize();
        micronauts_.push_back(std::move(compute));
        
        // Create attention Micronaut
        MicronautConfig attention_config;
        attention_config.id = "webx_attention_001";
        attention_config.type = MicronautType::Attention;
        attention_config.backend = GPUBackend::DirectML;
        
        auto attention = std::make_unique<Micronaut>(attention_config, gpu_, graph_);
        attention->initialize();
        micronauts_.push_back(std::move(attention));
    }
    
    bool executeLayer1(Field* field) {
        // Layer 1: Model Execution
        // Find appropriate Micronaut and execute
        for (auto& micronaut : micronauts_) {
            if (micronaut->execute(field)) {
                // Log semantic validation confidence from Micronaut expert shaders
                const auto& validation = micronaut->getLastValidation();
                std::cout << "BOSS Layer 1: field " << field->id
                          << " semantic confidence: " << validation.confidence
                          << " shader: " << validation.shader_name << std::endl;
                return true;
            }
        }
        return false;
    }
    
    bool executeLayer2(Field* field) {
        // Layer 2: Routing & Load Balancing
        // Simplified: just log for now
        std::cout << "BOSS Layer 2: Routing field " << field->id << std::endl;
        return true;
    }
    
    bool executeLayer3(Field* field) {
        // Layer 3: Caching & Optimization
        // Simplified: update pressure based on performance
        if (field->latency_ms < 50.0) {
            graph_->updatePressure(field->id, 0.9);
        } else {
            graph_->updatePressure(field->id, 0.5);
        }
        return true;
    }
};

// ─────────────────────────────────────────────────────────────
// Confidence Engine — Computes and improves field confidence
// Used by FieldOperations and MicronautFactory.
// ──────────────────────────────────────────────────────────────

class ConfidenceEngine {
public:
    static FieldConfidenceProfile compute(const Field& field,
                                          const std::vector<Token>& tokens) {
        FieldConfidenceProfile profile;
        
        // SM/VM: Semantic correctness
        if (field.token_count == 0) {
            profile.semantic_correctness = 0.0f;
        } else if (field.pressure < 0.05f) {
            profile.semantic_correctness = 0.25f;
        } else {
            profile.semantic_correctness = 0.90f;
        }
        
        // GM: Domain alignment
        static const std::unordered_set<std::string> known_domains = {
            "Compute", "Attention", "Memory", "Geometry",
            "Execution", "Evolution", "Vision"
        };
        profile.domain_alignment = known_domains.count(field.domain) ? 0.95f : 0.60f;
        
        // VM: Constraint satisfaction
        float confidence_sum = 0.0f;
        for (const auto& token : tokens) {
            confidence_sum += token.confidence;
        }
        profile.constraint_satisfaction = tokens.empty()
            ? 0.0f
            : std::clamp(confidence_sum / tokens.size(), 0.0f, 1.0f);
        
        // MM: Token coherence
        profile.token_coherence = computeCoherence(tokens);
        
        // Weighted aggregate
        profile.aggregate =
            profile.semantic_correctness * FieldConfidenceProfile::SEMANTIC_WEIGHT +
            profile.domain_alignment * FieldConfidenceProfile::DOMAIN_WEIGHT +
            profile.constraint_satisfaction * FieldConfidenceProfile::CONSTRAINT_WEIGHT +
            profile.token_coherence * FieldConfidenceProfile::COHERENCE_WEIGHT;
        
        return profile;
    }
    
    static float computeCoherence(const std::vector<Token>& tokens) {
        if (tokens.size() < 2) return 1.0f;
        
        float mean = std::accumulate(tokens.begin(), tokens.end(), 0.0f,
            [](float sum, const Token& t) { return sum + (t.activation * t.confidence); })
            / tokens.size();
        
        float variance = 0.0f;
        for (const auto& token : tokens) {
            float pressure = token.activation * token.confidence;
            float diff = pressure - mean;
            variance += diff * diff;
        }
        variance /= tokens.size();
        
        return std::clamp(1.0f - variance * 5.0f, 0.0f, 1.0f);
    }
    
    static void improve(Field& field,
                        std::vector<Token>& tokens,
                        uint32_t research_cycles = 1) {
        if (research_cycles == 0) return;
        
        for (uint32_t i = 0; i < research_cycles; ++i) {
            for (auto& token : tokens) {
                float coherence_bonus = token.activation * 0.05f;
                token.confidence = std::clamp(token.confidence + coherence_bonus, 0.0f, 1.0f);
            }
            
            auto profile = compute(field, tokens);
            float agreement_floor = profile.aggregate * 0.1f;
            for (auto& token : tokens) {
                token.confidence = std::clamp(
                    std::max(token.confidence, agreement_floor), 0.0f, 1.0f);
            }
            
            field.confidence_profile.research_cycles++;
            field.confidence_profile.validation_cycles++;
        }
        
        field.confidence_profile = compute(field, tokens);
    }
};

// ─────────────────────────────────────────────────────────────
// Micronaut Factory — Builds semantic shader networks on the fly
// ──────────────────────────────────────────────────────────────

struct SemanticPrompt {
    std::string text;
    std::string domain_hint;   // optional: "math", "vision", "code", etc.
    float urgency = 0.5f;      // 0.0 - 1.0
};

struct MicronautNetwork {
    std::string id;
    std::string topic;
    std::vector<std::string> field_specs;
    std::vector<std::unique_ptr<Micronaut>> micronauts;
    std::vector<std::string> replay_log;
    uint64_t invocation_count = 0;
    
    void log(const std::string& message) {
        replay_log.push_back(message);
    }
};

class MicronautFactory {
private:
    GPUComputeEngine* gpu_ = nullptr;
    FieldGraph* graph_ = nullptr;
    std::unordered_map<std::string, std::unique_ptr<MicronautNetwork>> networks_;
    mutable std::mutex mutex_;
    uint64_t next_network_id_ = 1;
    
public:
    MicronautFactory(GPUComputeEngine* gpu, FieldGraph* graph)
        : gpu_(gpu), graph_(graph) {}
    
    /**
     * Build a network of Micronauts from a user prompt/reply.
     * The factory infers which expert shaders are needed, creates Micronauts,
     * and compiles their shaders on the fly (via D3DCompiler_47 conceptually).
     */
    MicronautNetwork* buildNetwork(const SemanticPrompt& prompt) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::string network_id = "network_" + std::to_string(next_network_id_++);
        auto network = std::make_unique<MicronautNetwork>();
        network->id = network_id;
        network->topic = prompt.domain_hint.empty() ? inferTopic(prompt.text) : prompt.domain_hint;
        
        std::cout << "MicronautFactory: building network " << network_id
                  << " for topic '" << network->topic << "'" << std::endl;
        
        // Infer required Micronaut expert types from prompt semantics
        auto types = inferMicronautTypes(prompt, network->topic);
        network->field_specs = inferFieldSpecs(prompt, network->topic);

        for (MicronautType type : types) {
            MicronautConfig config;
            config.id = network_id + "_" + MicronautShaderRegistry::getTypeName(type);
            config.type = type;
            config.backend = GPUBackend::DirectML;
            config.shader_path = compileShaderOnTheFly(type, network->topic);
            config.field_specs = network->field_specs;
            config.node_ids.push_back(config.id);
            config.provider = "directml";
            
            auto micronaut = std::make_unique<Micronaut>(config, gpu_, graph_);
            micronaut->initialize();
            network->log("Created " + MicronautShaderRegistry::getTypeName(type)
                         + " with shader " + config.shader_path);
            network->micronauts.push_back(std::move(micronaut));
        }
        
        network->log("Network built with " + std::to_string(network->micronauts.size()) + " Micronauts");
        network->log("Network declares " +
                     std::to_string(network->field_specs.size()) + " field specs");
        
        MicronautNetwork* ptr = network.get();
        networks_[network_id] = std::move(network);
        return ptr;
    }
    
    /**
     * Replay a network against a field to simulate memory, data discovery,
     * or improve recent topics/domains/events through semantic association.
     */
    bool replayNetwork(const std::string& network_id, Field* field) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = networks_.find(network_id);
        if (it == networks_.end() || !field) {
            return false;
        }
        
        MicronautNetwork* network = it->second.get();
        network->invocation_count++;
        network->log("Replay #" + std::to_string(network->invocation_count) + " on field " + field->id);
        
        std::cout << "MicronautFactory: replaying network " << network_id
                  << " on field " << field->id << std::endl;
        
        bool any_success = false;
        for (auto& micronaut : network->micronauts) {
            if (micronaut->execute(field)) {
                any_success = true;
                const auto& validation = micronaut->getLastValidation();
                network->log("  " + MicronautShaderRegistry::getTypeName(micronaut->getType())
                             + " confidence=" + std::to_string(validation.confidence));
                
                // Improve field confidence through replay (discussion/memory)
                ConfidenceEngine::improve(*field, field->tokens, 1);
            }
        }
        
        return any_success;
    }
    
    /**
     * List all active networks and their topics.
     */
    std::vector<std::pair<std::string, std::string>> listNetworks() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::pair<std::string, std::string>> result;
        for (const auto& [id, network] : networks_) {
            result.push_back({id, network->topic});
        }
        return result;
    }
    
    /**
     * Get a network by ID.
     */
    MicronautNetwork* getNetwork(const std::string& id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = networks_.find(id);
        return it != networks_.end() ? it->second.get() : nullptr;
    }
    
private:
    std::string inferTopic(const std::string& text) {
        // Simplified topic inference via keyword matching
        std::string lower;
        lower.reserve(text.size());
        for (char c : text) lower.push_back(static_cast<char>(std::tolower(c)));
        
        if (lower.find("image") != std::string::npos ||
            lower.find("see") != std::string::npos ||
            lower.find("vision") != std::string::npos) {
            return "vision";
        }
        if (lower.find("math") != std::string::npos ||
            lower.find("number") != std::string::npos ||
            lower.find("calculate") != std::string::npos) {
            return "math";
        }
        if (lower.find("code") != std::string::npos ||
            lower.find("program") != std::string::npos ||
            lower.find("function") != std::string::npos) {
            return "code";
        }
        if (lower.find("memory") != std::string::npos ||
            lower.find("remember") != std::string::npos ||
            lower.find("past") != std::string::npos) {
            return "memory";
        }
        return "general";
    }
    
    std::vector<std::string> inferFieldSpecs(
        const SemanticPrompt& prompt, const std::string& topic) const {
        std::string text = prompt.text + " " + topic;
        std::transform(text.begin(), text.end(), text.begin(),
                       [](unsigned char c) {
                           return static_cast<char>(std::tolower(c));
                       });

        std::vector<std::string> specs;
        if (text.find("attraction") != std::string::npos ||
            text.find("repulsion") != std::string::npos ||
            text.find("force field") != std::string::npos ||
            text.find("physics") != std::string::npos) {
            specs.push_back(
                "native/win2d/field_system/specs/attraction_well_spec.json");
        }
        if (text.find("navigation") != std::string::npos ||
            text.find("navigate") != std::string::npos) {
            specs.push_back(
                "native/win2d/field_system/specs/navigation_force_spec.json");
        }
        return specs;
    }

    std::vector<MicronautType> inferMicronautTypes(const SemanticPrompt& prompt,
                                                   const std::string& topic) {
        std::vector<MicronautType> types;
        
        // Every network gets a parser and validator
        types.push_back(MicronautType::PM);
        types.push_back(MicronautType::VM);
        
        if (topic == "vision") {
            types.push_back(MicronautType::OM);
            types.push_back(MicronautType::DM);
            types.push_back(MicronautType::RM);
        } else if (topic == "math") {
            types.push_back(MicronautType::DM);
            types.push_back(MicronautType::RM);
            types.push_back(MicronautType::GM);
        } else if (topic == "code") {
            types.push_back(MicronautType::CM);
            types.push_back(MicronautType::DM);
            types.push_back(MicronautType::RM);
        } else if (topic == "memory") {
            types.push_back(MicronautType::MM);
            types.push_back(MicronautType::RM);
            types.push_back(MicronautType::FM);
        } else {
            // General reasoning: memory + reason + shader
            types.push_back(MicronautType::MM);
            types.push_back(MicronautType::RM);
            types.push_back(MicronautType::SM);
        }
        
        // High urgency adds training/execution Micronauts
        if (prompt.urgency > 0.7f) {
            types.push_back(MicronautType::XM);
            types.push_back(MicronautType::TM);
        }
        
        return types;
    }
    
    std::string compileShaderOnTheFly(MicronautType type, const std::string& topic) {
        // In production: call D3DCompileFromFile / D3DCompiler_47.dll here.
        // For this simplified runtime, we select the canonical expert shader and
        // append a topic-specific suffix so each network has a distinct shader identity.
        std::string base = MicronautShaderRegistry::getShader(type);
        
        // Replace extension with topic-specific compiled name
        size_t dot = base.find_last_of('.');
        std::string name = (dot == std::string::npos) ? base : base.substr(0, dot);
        return name + "_" + topic + ".cso";
    }
};

// ─────────────────────────────────────────────────────────────
// Artifact Emitter — Manufactures runtime artifacts from semantic intent.
// The Forge uses emitters to produce HLSL, WGSL, LLVM, OpenCL, DirectML
// graphs, SafeTensors, LoRA, SVG3D tensors, SCX state, and CSO binaries.
// ──────────────────────────────────────────────────────────────

enum class ForgeTarget {
    HLSL,
    WGSL,
    DXIL,
    LLVM,
    OpenCL,
    DirectML,
    SafeTensors,
    LoRA,
    SVG3D,
    SCX,
    CSO,
    FieldGraph
};

struct Artifact {
    ForgeTarget target;
    std::string name;
    std::string micronaut_id;
    std::vector<uint8_t> data;
    std::string provider_id;
    float confidence = 1.0f;
};

class ArtifactEmitter {
public:
    static Artifact emit(ForgeTarget target,
                         const Field& field,
                         const Provider* provider) {
        Artifact artifact;
        artifact.target = target;
        artifact.name = field.id + "." + targetName(target);
        if (target == ForgeTarget::CSO) {
            artifact.micronaut_id = "S7";
        }
        artifact.provider_id = provider ? provider->id : "cpu";

        // Simplified emitter: produce a small placeholder payload whose size
        // encodes the semantic topology and target format.
        const char* header = "KUHUL_ARTIFACT:";
        artifact.data.insert(artifact.data.end(), header, header + std::strlen(header));
        artifact.data.push_back(static_cast<uint8_t>(target));

        // Encode field pressure and confidence as a deterministic payload.
        uint32_t payload = static_cast<uint32_t>((field.pressure + field.confidence_profile.aggregate) * 1000.0f);
        artifact.data.push_back(static_cast<uint8_t>((payload >> 0) & 0xFF));
        artifact.data.push_back(static_cast<uint8_t>((payload >> 8) & 0xFF));

        return artifact;
    }

    static std::string targetName(ForgeTarget target) {
        switch (target) {
            case ForgeTarget::HLSL: return "hlsl";
            case ForgeTarget::WGSL: return "wgsl";
            case ForgeTarget::DXIL: return "dxil";
            case ForgeTarget::LLVM: return "llvm";
            case ForgeTarget::OpenCL: return "opencl";
            case ForgeTarget::DirectML: return "directml";
            case ForgeTarget::SafeTensors: return "safetensors";
            case ForgeTarget::LoRA: return "lora";
            case ForgeTarget::SVG3D: return "svg3d";
            case ForgeTarget::SCX: return "scx";
            case ForgeTarget::CSO: return "cso";
            case ForgeTarget::FieldGraph: return "fieldgraph";
        }
        return "unknown";
    }
};

// ─────────────────────────────────────────────────────────────
// Tensor Contract — Neutral tensor definition shared by all emitters.
// SafeTensors and SVG3D tensors are peers derived from the same contract.
// ──────────────────────────────────────────────────────────────

struct TensorContract {
    std::vector<int> shape;
    std::string layout = "NCHW";
    std::string precision = "float16";
    std::unordered_map<std::string, std::string> metadata;
    std::unordered_map<std::string, std::string> semantics;
    std::vector<uint8_t> storage;

    static TensorContract fromField(const Field& field) {
        TensorContract tensor;
        tensor.shape = {field.card_count, field.token_count};
        tensor.metadata["domain"] = field.domain;
        tensor.metadata["pressure"] = std::to_string(field.pressure);
        tensor.semantics["topology"] = "canonical";
        tensor.semantics["source"] = "FieldGraph";
        return tensor;
    }

    size_t elementCount() const {
        size_t count = 1;
        for (int dim : shape) count *= static_cast<size_t>(dim);
        return count;
    }
};

class TensorBuilder {
public:
    static TensorContract build(const Field& field, const std::string& format_hint = "") {
        TensorContract tensor = TensorContract::fromField(field);
        if (format_hint == "safetensors") {
            tensor.metadata["format"] = "safetensors";
        } else if (format_hint == "svg3d") {
            tensor.metadata["format"] = "svg3d";
        } else if (format_hint == "scx") {
            tensor.metadata["format"] = "scx";
        } else {
            tensor.metadata["format"] = "native";
        }
        return tensor;
    }

    static Artifact emitLoRA(const TensorContract& tensor, const Provider* provider) {
        Artifact artifact;
        artifact.target = ForgeTarget::LoRA;
        artifact.name = "lora_adapter.bin";
        artifact.provider_id = provider ? provider->id : "directml";
        artifact.data.insert(artifact.data.end(), tensor.storage.begin(), tensor.storage.end());
        return artifact;
    }
};

// ─────────────────────────────────────────────────────────────
// Micronaut Forge — Manufactures Micronauts and artifacts from semantic intent.
// The Forge uses native providers to compile and cache runtime objects.
// ──────────────────────────────────────────────────────────────

struct ForgeRequest {
    std::string prompt;
    std::string domain_hint;
    std::vector<ForgeTarget> targets;
    float urgency = 0.5f;
};

struct ForgeResult {
    std::string network_id;
    std::vector<std::string> field_specs;
    std::vector<std::unique_ptr<Micronaut>> micronauts;
    std::vector<Artifact> artifacts;
    float confidence = 0.0f;
};

class MicronautForge {
private:
    GPUComputeEngine* gpu_ = nullptr;
    FieldGraph* graph_ = nullptr;
    ProviderManager providers_;
    std::unordered_map<std::string, std::unique_ptr<ForgeResult>> cache_;
    mutable std::mutex mutex_;
    uint64_t next_id_ = 1;

public:
    MicronautForge(GPUComputeEngine* gpu, FieldGraph* graph)
        : gpu_(gpu), graph_(graph) {}

    ForgeResult* forge(const ForgeRequest& request) {
        std::lock_guard<std::mutex> lock(mutex_);

        std::string topic = request.domain_hint.empty()
            ? inferTopic(request.prompt)
            : request.domain_hint;

        // Reuse cached result if an identical prompt was recently forged.
        auto cache_key = request.prompt + "|" + topic;
        auto cached = cache_.find(cache_key);
        if (cached != cache_.end()) {
            cached->second->confidence = std::min(1.0f, cached->second->confidence + 0.01f);
            return cached->second.get();
        }

        std::string id = "forge_" + std::to_string(next_id_++);
        auto result = std::make_unique<ForgeResult>();
        result->network_id = id;

        std::cout << "MicronautForge: forging " << id << " for topic '" << topic << "'" << std::endl;

        // Select providers for required capabilities.
        Provider* compiler = providers_.selectBest(ProviderCapability::ShaderCompiler);
        Provider* compute = providers_.selectBest(ProviderCapability::TensorCompute);
        Provider* cache_provider = providers_.selectBest(ProviderCapability::Cache);

        if (compiler) std::cout << "  compiler: " << compiler->id << std::endl;
        if (compute) std::cout << "  compute: " << compute->id << std::endl;
        if (cache_provider) std::cout << "  cache: " << cache_provider->id << std::endl;

        // Manufacture Micronauts from semantic intent.
        MicronautFactory factory(gpu_, graph_);
        SemanticPrompt sp{request.prompt, topic, request.urgency};
        MicronautNetwork* network = factory.buildNetwork(sp);

        if (network) {
            result->field_specs = network->field_specs;
            for (auto& micronaut : network->micronauts) {
                result->micronauts.push_back(std::move(micronaut));
            }
        }

        // Emit requested artifacts.
        Field* target = graph_->createField(id + ".target", topic);
        if (target) {
            target->pressure = 0.9f;
            TensorContract tensor = TensorBuilder::build(*target, "svg3d");
            for (ForgeTarget target_type : request.targets) {
                result->artifacts.push_back(ArtifactEmitter::emit(target_type, *target, compute));
            }
            if (std::find(request.targets.begin(), request.targets.end(), ForgeTarget::LoRA) != request.targets.end()) {
                result->artifacts.push_back(TensorBuilder::emitLoRA(tensor, compute));
            }
        }

        result->confidence = 0.85f;
        ForgeResult* ptr = result.get();
        cache_[cache_key] = std::move(result);
        return ptr;
    }

    ProviderManager& getProviderManager() { return providers_; }

private:
    std::string inferTopic(const std::string& text) {
        std::string lower;
        lower.reserve(text.size());
        for (char c : text) lower.push_back(static_cast<char>(std::tolower(c)));

        if (lower.find("image") != std::string::npos ||
            lower.find("see") != std::string::npos ||
            lower.find("vision") != std::string::npos) {
            return "vision";
        }
        if (lower.find("math") != std::string::npos ||
            lower.find("number") != std::string::npos ||
            lower.find("calculate") != std::string::npos) {
            return "math";
        }
        if (lower.find("code") != std::string::npos ||
            lower.find("program") != std::string::npos ||
            lower.find("function") != std::string::npos) {
            return "code";
        }
        if (lower.find("memory") != std::string::npos ||
            lower.find("remember") != std::string::npos ||
            lower.find("past") != std::string::npos) {
            return "memory";
        }
        if (lower.find("train") != std::string::npos ||
            lower.find("lora") != std::string::npos ||
            lower.find("learn") != std::string::npos) {
            return "training";
        }
        return "general";
    }
};

// ─────────────────────────────────────────────────────────────
// WebX Main Runtime (Simplified NNC-K)
// ──────────────────────────────────────────────────────────────

class WebXRuntime {
private:
    GPUComputeEngine gpu_;
    FieldGraph graph_;
    std::unique_ptr<BOSSOrchestrator> boss_;
    
    bool running_ = false;
    
public:
    WebXRuntime() = default;
    
    bool initialize(const GPUConfig& gpu_config, int boss_layers = 1) {
        std::cout << "+----------------------------------------------------+" << std::endl;
        std::cout << "|  WebX Compute Engine v3.5.0                        |" << std::endl;
        std::cout << "+----------------------------------------------------+" << std::endl;
        std::cout << "|  Simplified GPU Compute with Semantic Field Graph  |" << std::endl;
        std::cout << "+----------------------------------------------------+" << std::endl;
        std::cout << std::endl;
        
        // Initialize GPU
        if (!gpu_.initialize(gpu_config)) {
            std::cerr << "Failed to initialize GPU" << std::endl;
            return false;
        }
        
        // Initialize BOSS orchestrator
        boss_ = std::make_unique<BOSSOrchestrator>(boss_layers);
        if (!boss_->initialize(&gpu_, &graph_)) {
            std::cerr << "Failed to initialize BOSS" << std::endl;
            return false;
        }
        
        running_ = true;
        
        std::cout << "[OK] WebX Runtime initialized" << std::endl;
        std::cout << "   GPU Backend: " << (gpu_config.backend == GPUBackend::DirectML ? "DirectML" : 
                                            gpu_config.backend == GPUBackend::CUDA ? "CUDA" : "CPU") << std::endl;
        std::cout << "   BOSS Layers: " << boss_layers << std::endl;
        std::cout << std::endl;
        
        return true;
    }
    
    void shutdown() {
        running_ = false;
        gpu_.shutdown();
        std::cout << "WebX Runtime shutdown" << std::endl;
    }
    
    // Create field for execution
    Field* createField(const std::string& id, const std::string& domain) {
        return graph_.createField(id, domain);
    }

    // Access GPU and graph for advanced usage (Forge, diagnostics, bindings)
    GPUComputeEngine& getGPU() { return gpu_; }
    FieldGraph& getGraph() { return graph_; }

    // Execute field through BOSS
    bool executeField(const std::string& field_id) {
        Field* field = graph_.getField(field_id);
        if (!field) {
            std::cerr << "Field not found: " << field_id << std::endl;
            return false;
        }
        
        return boss_->executeField(field);
    }
    
    // Get high-pressure fields
    std::vector<Field*> getReadyFields(float threshold = 0.7f) {
        return graph_.getHighPressureFields(threshold);
    }
    
    // Get statistics
    struct Stats {
        size_t field_count;
        size_t total_cards;
        size_t total_tokens;
        double avg_pressure;
    };
    
    Stats getStats() const {
        Stats stats = {0, 0, 0, 0.0};
        
        const auto& fields = graph_.getAllFields();
        stats.field_count = fields.size();
        
        double total_pressure = 0.0;
        for (const auto& [id, field] : fields) {
            stats.total_cards += field->card_count;
            stats.total_tokens += field->token_count;
            total_pressure += field->pressure;
        }
        
        if (!fields.empty()) {
            stats.avg_pressure = total_pressure / fields.size();
        }
        
        return stats;
    }
    
    bool isRunning() const { return running_; }
};

}  // namespace WebX

// ─────────────────────────────────────────────────────────────
// C API (for Python/other language bindings)
// ──────────────────────────────────────────────────────────────

extern "C" {

typedef void* WebXRuntimeHandle;

inline __declspec(dllexport) WebXRuntimeHandle webx_create() {
    return new WebX::WebXRuntime();
}

inline __declspec(dllexport) void webx_destroy(WebXRuntimeHandle handle) {
    delete static_cast<WebX::WebXRuntime*>(handle);
}

inline __declspec(dllexport) bool webx_initialize(
    WebXRuntimeHandle handle,
    int gpu_backend,  // 0=DirectML, 1=CUDA, 2=CPU
    int boss_layers
) {
    WebX::GPUConfig config;
    config.backend = static_cast<WebX::GPUBackend>(gpu_backend);
    return static_cast<WebX::WebXRuntime*>(handle)->initialize(config, boss_layers);
}

inline __declspec(dllexport) void webx_shutdown(WebXRuntimeHandle handle) {
    static_cast<WebX::WebXRuntime*>(handle)->shutdown();
}

inline __declspec(dllexport) void* webx_create_field(
    WebXRuntimeHandle handle,
    const char* id,
    const char* domain
) {
    return static_cast<WebX::WebXRuntime*>(handle)->createField(id, domain);
}

inline __declspec(dllexport) bool webx_execute_field(
    WebXRuntimeHandle handle,
    const char* field_id
) {
    return static_cast<WebX::WebXRuntime*>(handle)->executeField(field_id);
}

inline __declspec(dllexport) bool webx_get_stats(
    WebXRuntimeHandle handle,
    size_t* field_count,
    size_t* total_cards,
    size_t* total_tokens,
    double* avg_pressure
) {
    auto stats = static_cast<WebX::WebXRuntime*>(handle)->getStats();
    *field_count = stats.field_count;
    *total_cards = stats.total_cards;
    *total_tokens = stats.total_tokens;
    *avg_pressure = stats.avg_pressure;
    return true;
}

}  // extern "C"
