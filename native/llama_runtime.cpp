#define WIN32_LEAN_AND_MEAN
#include "llama_runtime.h"

#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <thread>
#include <vector>

#if defined(KUHUL_HAS_LLAMA_CPP)
#include <llama.h>
#endif

namespace {

#if defined(KUHUL_HAS_LLAMA_CPP)
template <typename T>
T symbol(HMODULE module, const char* name) {
    return reinterpret_cast<T>(GetProcAddress(module, name));
}

struct LlamaApi {
    using backend_init_fn = void (*)();
    using backend_free_fn = void (*)();
    using model_default_fn = llama_model_params (*)();
    using context_default_fn = llama_context_params (*)();
    using model_load_fn = llama_model* (*)(const char*, llama_model_params);
    using model_free_fn = void (*)(llama_model*);
    using init_context_fn = llama_context* (*)(llama_model*, llama_context_params);
    using free_context_fn = void (*)(llama_context*);
    using model_vocab_fn = const llama_vocab* (*)(const llama_model*);
    using vocab_count_fn = int32_t (*)(const llama_vocab*);
    using tokenize_fn = int32_t (*)(const llama_vocab*, const char*, int32_t,
                                    llama_token*, int32_t, bool, bool);
    using batch_one_fn = llama_batch (*)(llama_token*, int32_t);
    using decode_fn = int32_t (*)(llama_context*, llama_batch);
    using logits_fn = float* (*)(llama_context*);
    using vocab_text_fn = const char* (*)(const llama_vocab*, llama_token);
    using vocab_eog_fn = bool (*)(const llama_vocab*, llama_token);

    backend_init_fn backend_init = nullptr;
    backend_free_fn backend_free = nullptr;
    model_default_fn model_default = nullptr;
    context_default_fn context_default = nullptr;
    model_load_fn model_load = nullptr;
    model_free_fn model_free = nullptr;
    init_context_fn init_context = nullptr;
    free_context_fn free_context = nullptr;
    model_vocab_fn model_vocab = nullptr;
    vocab_count_fn vocab_count = nullptr;
    tokenize_fn tokenize = nullptr;
    batch_one_fn batch_one = nullptr;
    decode_fn decode = nullptr;
    logits_fn logits = nullptr;
    vocab_text_fn vocab_text = nullptr;
    vocab_eog_fn vocab_eog = nullptr;

    bool bind(HMODULE module) {
#define BIND(field, api_name) \
        field = symbol<decltype(field)>(module, "llama_" api_name)
        BIND(backend_init, "backend_init");
        BIND(backend_free, "backend_free");
        BIND(model_default, "model_default_params");
        BIND(context_default, "context_default_params");
        BIND(model_load, "model_load_from_file");
        BIND(model_free, "model_free");
        BIND(init_context, "init_from_model");
        BIND(free_context, "free");
        BIND(model_vocab, "model_get_vocab");
        BIND(vocab_count, "vocab_n_tokens");
        BIND(tokenize, "tokenize");
        BIND(batch_one, "batch_get_one");
        BIND(decode, "decode");
        BIND(logits, "get_logits");
        BIND(vocab_text, "vocab_get_text");
        BIND(vocab_eog, "vocab_is_eog");
#undef BIND
        return backend_init && backend_free && model_default &&
               context_default && model_load && model_free &&
               init_context && free_context && model_vocab && vocab_count &&
               tokenize && batch_one && decode && logits &&
               vocab_text && vocab_eog;
    }
};
#endif

std::filesystem::path rootFromEnv() {
    char value[4096] = {};
    DWORD length = GetEnvironmentVariableA("KUHUL_LLAMA_ROOT", value,
                                           static_cast<DWORD>(sizeof(value)));
    if (length > 0) return std::filesystem::path(value);
#if defined(KUHUL_LLAMA_DEFAULT_ROOT)
    return std::filesystem::path(KUHUL_LLAMA_DEFAULT_ROOT);
#else
    return {};
#endif
}

} // namespace

LlamaRuntime::~LlamaRuntime() {
    unload();
}

bool LlamaRuntime::load(const std::string& runtime_root) {
    unload();
    error_.clear();

#if !defined(KUHUL_HAS_LLAMA_CPP)
    (void)runtime_root;
    error_ = "llama.cpp headers were not configured";
    return false;
#else
    std::filesystem::path root = runtime_root.empty()
        ? rootFromEnv()
        : std::filesystem::path(runtime_root);
    if (root.empty()) {
        error_ = "set KUHUL_LLAMA_ROOT or pass a llama.cpp runtime directory";
        return false;
    }

    const std::vector<std::filesystem::path> dependencies = {
        root / "ggml-base.dll",
        root / "ggml-cpu-haswell.dll",
        root / "ggml-cpu-x64.dll",
        root / "ggml.dll"
    };
    for (const auto& dependency : dependencies) {
        if (std::filesystem::exists(dependency))
            LoadLibraryW(dependency.c_str());
    }

    const auto llamaPath = root / "llama.dll";
    HMODULE llama = LoadLibraryW(llamaPath.c_str());
    if (!llama) {
        error_ = "cannot load llama.dll: " + llamaPath.string();
        return false;
    }

    static LlamaApi api;
    if (!api.bind(llama)) {
        FreeLibrary(llama);
        error_ = "llama.dll is missing required C API exports";
        return false;
    }

    llama_ = llama;
    api.backend_init();
    return true;
#endif
}

bool LlamaRuntime::generate(const std::string& model_path,
                            const std::string& prompt,
                            uint32_t max_tokens,
                            std::string& output) {
    output.clear();
    error_.clear();
#if !defined(KUHUL_HAS_LLAMA_CPP)
    (void)model_path;
    (void)prompt;
    (void)max_tokens;
    error_ = "llama.cpp headers were not configured";
    return false;
#else
    if (!llama_) {
        error_ = "llama runtime is not loaded";
        return false;
    }
    static LlamaApi api;
    HMODULE module = static_cast<HMODULE>(llama_);
    if (!api.bind(module)) {
        error_ = "llama API binding is unavailable";
        return false;
    }

    llama_model_params modelParams = api.model_default();
    modelParams.n_gpu_layers = 0;
    llama_model* model = api.model_load(model_path.c_str(), modelParams);
    if (!model) {
        error_ = "llama model load failed: " + model_path;
        return false;
    }

    const llama_vocab* vocab = api.model_vocab(model);
    const int32_t tokenCapacity = std::max<int32_t>(
        512, static_cast<int32_t>(prompt.size() * 4 + 32));
    std::vector<llama_token> tokens(static_cast<size_t>(tokenCapacity));
    int32_t tokenCount = api.tokenize(vocab, prompt.c_str(),
                                       static_cast<int32_t>(prompt.size()),
                                       tokens.data(), tokenCapacity, true, false);
    if (tokenCount < 0) {
        tokens.resize(static_cast<size_t>(-tokenCount));
        tokenCount = api.tokenize(vocab, prompt.c_str(),
                                  static_cast<int32_t>(prompt.size()),
                                  tokens.data(), -tokenCount, true, false);
    }
    if (tokenCount <= 0) {
        api.model_free(model);
        error_ = "llama tokenization failed";
        return false;
    }
    tokens.resize(static_cast<size_t>(tokenCount));

    llama_context_params contextParams = api.context_default();
    contextParams.n_ctx = std::max<uint32_t>(
        2048, static_cast<uint32_t>(tokens.size()) + max_tokens + 8);
    contextParams.n_batch = std::min<uint32_t>(512, contextParams.n_ctx);
    contextParams.n_ubatch = contextParams.n_batch;
    contextParams.n_threads = static_cast<int32_t>(
        std::max(1u, std::thread::hardware_concurrency()));
    contextParams.n_threads_batch = contextParams.n_threads;
    llama_context* context = api.init_context(model, contextParams);
    if (!context) {
        api.model_free(model);
        error_ = "llama context creation failed";
        return false;
    }

    auto finish = [&]() {
        api.free_context(context);
        api.model_free(model);
    };

    if (api.decode(context, api.batch_one(tokens.data(), tokenCount)) != 0) {
        finish();
        error_ = "llama prompt decode failed";
        return false;
    }

    const int32_t vocabCount = api.vocab_count(vocab);
    for (uint32_t step = 0; step < max_tokens; ++step) {
        float* logits = api.logits(context);
        if (!logits || vocabCount <= 0) {
            finish();
            error_ = "llama logits unavailable";
            return false;
        }
        llama_token next = 0;
        for (int32_t token = 1; token < vocabCount; ++token) {
            if (logits[token] > logits[next]) next = token;
        }
        if (api.vocab_eog(vocab, next)) break;

        const char* piece = api.vocab_text(vocab, next);
        if (piece) output += piece;
        if (api.decode(context, api.batch_one(&next, 1)) != 0) {
            finish();
            error_ = "llama generation decode failed";
            return false;
        }
    }

    finish();
    return true;
#endif
}

void LlamaRuntime::unload() {
#if defined(KUHUL_HAS_LLAMA_CPP)
    if (llama_) {
        using backend_free_fn = void (*)();
        auto backendFree = reinterpret_cast<backend_free_fn>(
            GetProcAddress(static_cast<HMODULE>(llama_), "llama_backend_free"));
        if (backendFree) backendFree();
        FreeLibrary(static_cast<HMODULE>(llama_));
    }
#endif
    llama_ = nullptr;
    ggml_ = nullptr;
}
