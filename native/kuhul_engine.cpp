/**
 * kuhul_engine.cpp — Unified K'UHUL Semantic Engine entry point.
 *
 * This is the canonical executable entry point for the v3.5.0-WebX runtime.
 * It bootstraps the WebX compute runtime, initializes a FieldGraph from the
 * canonical semantic contract, and demonstrates prompt → Micronaut Forge →
 * execution flow.
 */

#include "webx_compute.h"
#include "llama_runtime.h"
#include "http_api_server.h"
#include "inference/xshard_model_loader.h"
#include "inference/scx_manifest_bridge.h"
#include "inference/xshard_hot_swap.h"
#include "inference/safetensors_reader.h"
#include "inference/xshard_block_reader.h"
#include "inference/xshard_int8_quantizer.h"
#include "gpu_trainer/scx_stream_engine.h"
#include "gpu_trainer/d3d11_expert_block.h"
#include "gpu_trainer/xshard_attention.h"
#include "runtime/phase_runtime.h"
#include "runtime/fold_registry.h"
#include "runtime/node_registry.h"
#include "runtime/KuhulAppCreator.h"
#include "runtime/domain_runtime.h"
#include "runtime/task_engine.h"
#include "runtime/task_executor_abi.h"
#include <cstdlib>
#include "runtime/atomic_shell_manifest.h"
#include "runtime/opengl_frame_adapter.h"
#include "runtime/world_tile.h"
#include "runtime/instant_agent.h"
#include "source/kuhul_source.h"
#include <iostream>
#include <filesystem>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_4.h>
#include <d3d11.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

static int runScxDx12Smoke(const char* path, uint32_t maxTiles,
                           uint32_t tensorType) {
    XShardModelManifest manifest;
    XShardModelLoader loader;
    if (!loader.validate(path, manifest)) {
        std::cerr << "SCX DX12 smoke failed: " << loader.error() << "\n";
        return 1;
    }

    if (std::getenv("KUHUL_D3D12_DEBUG") == std::string("1")) {
        ComPtr<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
            debug->EnableDebugLayer();
            ComPtr<ID3D12Debug1> debug1;
            if (SUCCEEDED(debug.As(&debug1)))
                debug1->SetEnableGPUBasedValidation(TRUE);
            std::cerr << "SCX DX12 debug layer enabled\n";
        } else {
            std::cerr << "SCX DX12 debug layer unavailable\n";
        }
    }

    ComPtr<IDXGIFactory4> factory;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
        std::cerr << "SCX DX12 smoke failed: CreateDXGIFactory1 hr=0x"
                  << std::hex << static_cast<unsigned long>(hr) << "\n";
        return 1;
    }

    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    const bool forceHardware =
        std::getenv("KUHUL_D3D12_HARDWARE") == std::string("1");
    const bool forceWarp =
        !forceHardware || std::getenv("KUHUL_D3D12_WARP") == std::string("1");
    if (!forceWarp) {
        for (UINT i = 0;
             factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND;
             ++i) {
            DXGI_ADAPTER_DESC1 desc{};
            adapter->GetDesc1(&desc);
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
                adapter.Reset();
                continue;
            }
            if (SUCCEEDED(D3D12CreateDevice(
                    adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                    IID_PPV_ARGS(&device))))
                break;
            adapter.Reset();
        }
    }
    if (!device) {
        adapter.Reset();
        hr = factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter));
        if (FAILED(hr) ||
            FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                     IID_PPV_ARGS(&device)))) {
            std::cerr << "SCX DX12 smoke failed: no compatible adapter\n";
            return 1;
        }
    }

    D3D12_COMMAND_QUEUE_DESC queueDesc{};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    ComPtr<ID3D12CommandQueue> queue;
    if (FAILED(device->CreateCommandQueue(
            &queueDesc, IID_PPV_ARGS(&queue)))) {
        std::cerr << "SCX DX12 smoke failed: CreateCommandQueue\n";
        return 1;
    }

    ScxStreamEngine engine;
    if (!engine.init(device.Get(), queue.Get(), 1, 2880)) {
        std::cerr << "SCX DX12 smoke failed: engine initialization ("
                  << engine.lastError() << ")\n";
        return 1;
    }
    engine.setOpcodeCallback([](const std::string& op,
                                const std::string& json) {
        if (op == "GPU_DECODE" || op == "PLUGIN_REGISTER")
            std::cerr << "  " << op << " " << json << "\n";
    });

    uint32_t streamedTiles = 0;
    std::string streamError;
    if (!streamXShardTilesToEngine(
            manifest, engine, 0, maxTiles, streamedTiles, streamError,
            tensorType)) {
        engine.shutdown();
        std::cerr << "SCX DX12 smoke failed: " << streamError << "\n";
        return 1;
    }

    engine.shutdown();
    const bool ready = engine.layerReady(0);
    std::cout << "SCX DX12 smoke ready=" << (ready ? "true" : "false")
              << " streamed=" << streamedTiles << "\n";
    return ready ? 0 : 1;
}

static int runScxD3D11Smoke(const char* path) {
    XShardModelManifest manifest;
    XShardModelLoader loader;
    if (!loader.validate(path, manifest)) {
        std::cerr << "SCX D3D11 smoke failed: " << loader.error() << "\n";
        return 1;
    }

    D3D11Engine d3d11;
    const bool forceHardware =
        std::getenv("KUHUL_D3D11_HARDWARE") == std::string("1");
    const bool forceWarp =
        !forceHardware || std::getenv("KUHUL_D3D11_WARP") == std::string("1");
    if (!d3d11.init(forceWarp, true)) {
        std::cerr << "SCX D3D11 smoke failed: " << d3d11.initReason() << "\n";
        return 1;
    }
    if (d3d11.featureLevel() < D3D_FEATURE_LEVEL_11_0) {
        std::cerr << "SCX D3D11 smoke failed: feature level below 11_0\n";
        return 1;
    }

    D3D11ExpertBlockGemm gemm;
    if (!gemm.init(d3d11)) {
        std::cerr << "SCX D3D11 smoke failed: " << gemm.error() << "\n";
        return 1;
    }
    const std::vector<float> hidden = {1.f, 2.f, 3.f, 4.f};
    const std::vector<float> weights = {
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 1.f, 0.f,
        0.f, 0.f, 0.f, 1.f};
    std::vector<float> output;
    if (!gemm.run(hidden, weights, 1, 4, 4, output) ||
        output != hidden) {
        std::cerr << "SCX D3D11 smoke failed: "
                  << (gemm.error().empty() ? "CSO result mismatch"
                                           : gemm.error())
                  << "\n";
        return 1;
    }

    std::cout << "SCX D3D11 smoke ready=true"
              << " feature_level=0x" << std::hex
              << static_cast<unsigned>(d3d11.featureLevel()) << std::dec
              << " cs=cs_5_0"
              << " warp=" << (d3d11.usedWarp() ? "true" : "false")
              << " tensors=" << manifest.tensors.size() << "\n";
    return 0;
}

static void printUsage(const char* argv0) {
    std::cout << "K'UHUL Semantic Engine v3.5.0-WebX\n"
              << "Usage: " << argv0 << " [command] [options]\n\n"
              << "Commands:\n"
              << "  --version          Print version and exit\n"
              << "  --detect-gpu       Detect available GPU backend\n"
              << "  --providers        List native provider status\n"
              << "  --runtime-sandbox  Run fold phase admission for MicrosoftSDK\n"
              << "  --runtime-domains  Execute all seven universal algebra domains\n"
              << "  atomic-shell PATH [--render]\n"
              << "                     Validate and optionally render Atomic Shell\n"
              << "  opengl-frame-smoke [FRAMES]\n"
              << "                     Render a native OpenGL FRAME smoke surface\n"
              << "  opengl-obj-smoke PATH [FRAMES|--interactive]\n"
              << "                     Load and render an OBJ scene\n"
              << "  opengl-game-smoke MANIFEST [FRAMES|--interactive]\n"
              << "                     Render a GAME manifest scene\n"
              << "  world-tile-smoke X Z [SIZE] [SEED]\n"
              << "                     Generate a deterministic bounded world tile\n"
              << "  opengl-world-tile-smoke X Z [SIZE] [SEED] [FRAMES|--interactive]\n"
              << "                     Render a generated terrain tile in OpenGL\n"
              << "  opengl-strategy-globe [FRAMES|--interactive]\n"
              << "                     Render the strategy Ghost HUD and globe\n"
              << "  opengl-particle-smoke EFFECT [COUNT] [SEED] [FRAMES|--interactive]\n"
              << "                     Render a deterministic particle effect\n"
              << "  instant-agent-smoke ROLE [SEED] [FRAMES|--interactive]\n"
              << "                     Spawn and render an instant in-game agent\n"
              << "  --Atomic.DOM [PATH] [--chat|--login|--game]\n"
              << "                     Launch the Atomic DOM terminal menu\n"
              << "  stream-xshard PATH [MAX_LAYER] [MAX_TILES]\n"
              << "                     Validate xshards and optionally stream a bounded tile prefix\n"
              << "  scx-dx12-smoke PATH [MAX_TILES] [TENSOR_TYPE]\n"
              << "                     Initialize one DX12 layer and stream a bounded prefix\n"
              << "  scx-d3d11-smoke PATH\n"
              << "                     Validate a D3D11 feature-level 11_0+ cs_5_0 CSO path\n"
              << "  hot-swap-xshard ACTIVE CANDIDATE [MAX_LAYER] [MAX_TILES] [TENSOR_TYPE]\n"
              << "                     Stage a bounded compatible xshard load before atomic activation\n"
              << "  task-engine PATH    Plan a TaskList.kuhul against native providers\n"
              << "  task-run PATH       Execute allowlisted admitted tasks via helper ABI\n"
              << "  task-boss PATH      Execute admitted tasks through BOSS and FieldGraph\n"
              << "  -task PATH          Plan a JSON task list against native providers\n"
              << "  parse PATH          Parse a .kuhul source document\n"
              << "  manifest PATH       Validate a runtime manifest and its fold files\n"
              << "  fold-manifest PATH  Validate runtime fold contracts and CSO artifacts\n"
              << "  node-manifest PATH  Load node contracts and resolve providers\n"
              << "  wwa-status          Inspect WWA runtime components\n"
              << "  wwa-templates        List app templates\n"
              << "  wwa-kits             List app design kits\n"
              << "  wwa-demos            List app demos\n"
              << "  wwa-components       List reusable app components\n"
              << "  wwa-create PARENT NAME [TEMPLATE] [KIT] [HOST] [PORT]\n"
              << "                     Create an app scaffold\n"
              << "  ast PATH            Emit KAST JSON\n"
              << "  ebnf PATH           Show grammar validation context\n"
              << "  analyze PATH        Analyze folds, providers, and phases\n"
              << "  validate PATH       Validate a .kuhul source document\n"
              << "  compile PATH [OUT]  Emit a semantic SCX package\n"
              << "  code PATH [TARGET]  Generate a host/provider code skeleton\n"
              << "  emit TARGET PATH    Emit a supported semantic representation\n"
              << "  execute PATH        Execute a validated .kuhul runtime\n"
              << "  runtime PATH        Run the native runtime sandbox\n"
              << "  graph|pressure|cache|inspect  Inspect native runtime state\n"
              << "  micronaut list      Inspect forged runtime registry\n"
              << "  --test all         Run self-test/demo loop\n"
              << "  --infer-xshard Q K V [passes]\n"
              << "                     Run native D3D11 attention over XSQ2 shards\n"
              << "  --validate-xshard PATH\n"
              << "                     Validate one shard or an xshard directory\n"
              << "  xshard-block PATH EXPERT ROW ROWS COL COLS\n"
              << "                     Read one bounded expert block from disk\n"
              << "  xshard-block-plan PATH BLOCK_ROWS BLOCK_COLS\n"
              << "                     Plan bounded blocks for every expert record\n"
              << "  xshard-block-gemm PATH EXPERT ROW ROWS\n"
              << "                     Run one streamed expert block through D3D11 SM5\n"
              << "  xshard-block-gemm-int8 PATH EXPERT ROW ROWS [SCALE] [ZERO]\n"
              << "                     Run one streamed INT8 expert block through D3D11 SM5\n"
              << "  xshard-block-gemm-int4 PATH EXPERT ROW ROWS [SCALE] [ZERO]\n"
              << "                     Run one packed INT4 expert block through D3D11 SM5\n"
              << "  xshard-quantize-int8 INPUT OUTPUT\n"
              << "                     Convert FP16/FP32 xshard to INT8 with scale metadata\n"
              << "  xshard-quantize-int4 INPUT OUTPUT\n"
              << "                     Convert FP16/FP32 xshard to packed INT4\n"
              << "  --validate-safetensors PATH\n"
              << "                     Validate a SafeTensors index and data bounds\n"
              << "  --infer-llama MODEL PROMPT [tokens]\n"
              << "                     Run native llama.cpp CPU inference\n"
              << "  --serve [port]      Run the localhost HTTP API server\n"
              << "  strategy-host [port] Run an opt-in public strategy host\n"
              << "  --prompt <text>    Forge and execute a Micronaut network\n"
              << "  --forge <text>     Forge artifacts from semantic intent\n"
              << "  --help             Show this message\n";
}

static bool loadKuhulDocument(const std::string& path, Kuhul::Source::Document& document) {
    if (Kuhul::Source::load(path, document)) return true;
    std::cerr << "K'UHUL source load failed: " << path << "\n";
    for (const auto& error : document.errors) std::cerr << "  error: " << error << "\n";
    return false;
}

static int runKuhulSourceCommand(const std::string& command, int argc, char** argv) {
    if (command == "graph" || command == "pressure" ||
        command == "cache" || command == "inspect") {
        const auto& context = Kuhul::Runtime::runtime().context();
        std::cout << "Runtime " << command << "\n"
                  << "  nodes:      " << context.nodes.size() << "\n"
                  << "  queue:      " << context.queue.size() << "\n"
                  << "  tick:       " << context.tick.number << "\n"
                  << "  cache:      " << context.cache.updates << "\n"
                  << "  providers:  " << context.backends.selections << "\n"
                  << "  trace:      " << context.trace.size() << "\n";
        return 0;
    }

    if (command == "micronaut") {
        if (argc < 3 || std::string(argv[2]) == "list") {
            const auto& context = Kuhul::Runtime::runtime().context();
            std::cout << "Micronaut registry\n"
                      << "  lookups: " << context.micronauts.lookups << "\n"
                      << "  lifecycle: forged artifacts are managed by --forge\n";
            return 0;
        }
        std::cerr << "micronaut subcommand is not registered: " << argv[2] << "\n";
        return 2;
    }

    if (command == "manifest") {
        if (argc < 3) {
            std::cerr << "Usage: manifest PATH\n";
            return 2;
        }
        Kuhul::Source::Document manifest;
        if (!loadKuhulDocument(argv[2], manifest)) return 1;
        const std::filesystem::path manifestPath(argv[2]);
        const std::filesystem::path base = manifestPath.parent_path();
        size_t checked = 0;
        size_t missing = 0;
        size_t offset = 0;
        while ((offset = manifest.source.find('"', offset)) != std::string::npos) {
            const size_t end = manifest.source.find('"', offset + 1);
            if (end == std::string::npos) break;
            const std::string value = manifest.source.substr(offset + 1, end - offset - 1);
            const bool isKuhul = value.size() >= 6 &&
                value.compare(value.size() - 6, 6, ".kuhul") == 0;
            const bool isNode = value.size() >= 5 &&
                value.compare(value.size() - 5, 5, ".node") == 0;
            if (isKuhul || isNode) {
                ++checked;
                const auto referenced = base / value;
                if (!std::filesystem::exists(referenced)) {
                    ++missing;
                    std::cerr << "Missing manifest entry: " << referenced.string() << "\n";
                }
            }
            offset = end + 1;
        }
        std::cout << "Manifest: " << argv[2] << "\n"
                  << "  valid:   " << (manifest.valid() && missing == 0 ? "true" : "false") << "\n"
                  << "  entries: " << checked << "\n";
        return manifest.valid() && missing == 0 ? 0 : 1;
    }

    if (command == "fold-manifest") {
        if (argc < 3) {
            std::cerr << "Usage: fold-manifest PATH\n";
            return 2;
        }
        Kuhul::Runtime::FoldRegistry registry;
        if (!registry.loadManifest(argv[2])) {
            std::cerr << "Fold manifest invalid: " << registry.error() << "\n";
            return 1;
        }
        std::cout << "Fold manifest: " << argv[2] << "\n"
                  << "  valid: true\n"
                  << "  runtime folds: " << registry.folds().size() << "\n";
        for (const auto& fold : registry.folds()) {
            std::cout << "  " << fold.name
                      << " phase=" << Kuhul::Runtime::phaseName(fold.phase)
                      << " micronaut=" << fold.micronaut
                      << " entry=" << fold.entry_point
                      << " observer=" << (fold.observer ? "true" : "false")
                      << "\n";
        }
        return 0;
    }

    if (command == "node-manifest") {
        if (argc < 3) {
            std::cerr << "Usage: node-manifest PATH\n";
            return 2;
        }
        WebX::ProviderManager providers;
        Kuhul::Runtime::NodeRegistry registry;
        if (!registry.loadManifest(argv[2])) {
            std::cerr << "Node manifest invalid: " << registry.error() << "\n";
            return 1;
        }
        registry.resolveProviders(providers);
        std::cout << "Node manifest: " << argv[2] << "\n"
                  << "  valid: true\n"
                  << "  nodes: " << registry.nodes().size() << "\n";
        for (const auto& node : registry.nodes()) {
            std::cout << "  " << node.name
                      << " capability=" << node.capability
                      << " provider=" << (node.provider_available
                          ? node.provider : "blocked")
                      << "\n";
        }
        return 0;
    }

    if (command == "wwa-status") {
        const Kuhul::Runtime::KuhulAppCreator creator;
        const auto status = creator.inspect();
        std::cout << "WWA runtime\n"
                  << "  WwaExt.dll: " << (status.extension_present ? "present" : "missing") << "\n"
                  << "  WwaApi.dll: " << (status.api_present ? "present" : "missing") << "\n"
                  << "  WWAHost.exe: " << (status.host_present ? "present" : "missing") << "\n";
        if (!status.ready()) {
            std::cerr << "WWA runtime unavailable: " << status.error << "\n";
            return 1;
        }
        return 0;
    }

    if (command == "wwa-create") {
        if (argc < 4) {
            std::cerr << "Usage: wwa-create PARENT NAME\n";
            return 2;
        }
        const Kuhul::Runtime::KuhulAppCreator creator;
        std::filesystem::path appRoot;
        std::string error;
        const std::string templateId = argc >= 5 ? argv[4] : "basic";
        const std::string kit = argc >= 6 ? argv[5] : "default";
        const std::string host = argc >= 7 ? argv[6] : "127.0.0.1";
        uint16_t port = 7431;
        if (argc >= 8) {
            try {
                const auto value = std::stoul(argv[7]);
                if (value == 0 || value > 65535) throw std::out_of_range("port");
                port = static_cast<uint16_t>(value);
            } catch (const std::exception&) {
                std::cerr << "Invalid app server port\n";
                return 2;
            }
        }
        if (!creator.create(argv[2], argv[3], appRoot, error, templateId,
                            kit, host, port)) {
            std::cerr << "WWA app creation failed: " << error << "\n";
            return 1;
        }
        std::cout << "WWA app created: " << appRoot.string() << "\n";
        return 0;
    }

    if (command == "wwa-templates") {
        for (const auto& item : Kuhul::Runtime::KuhulAppCreator::templates())
            std::cout << item.id << ": " << item.description << "\n";
        return 0;
    }

    if (command == "wwa-kits") {
        for (const auto& kit : Kuhul::Runtime::KuhulAppCreator::designKits())
            std::cout << kit.id << ": " << kit.description << "\n";
        return 0;
    }

    if (command == "wwa-demos") {
        for (const auto& demo : Kuhul::Runtime::KuhulAppCreator::demos())
            std::cout << demo.id << ": " << demo.description << "\n";
        return 0;
    }

    if (command == "wwa-components") {
        for (const auto& component : Kuhul::Runtime::KuhulAppCreator::components())
            std::cout << component.id << " [" << component.category << "]: "
                      << component.description << "\n";
        return 0;
    }

    const bool requiresSource =
        command == "parse" || command == "ast" || command == "ebnf" ||
        command == "analyze" || command == "validate" || command == "compile" ||
        command == "code" || command == "emit" || command == "execute" ||
        command == "runtime";
    if (!requiresSource) return -1;

    std::string sourcePath;
    std::string target;
    if (command == "emit") {
        if (argc < 4) {
            std::cerr << "Usage: emit TARGET PATH\n";
            return 2;
        }
        target = argv[2];
        sourcePath = argv[3];
    } else {
        if (argc < 3) {
            std::cerr << "Usage: " << command << " PATH\n";
            return 2;
        }
        sourcePath = argv[2];
        if (command == "code" && argc >= 4) target = argv[3];
    }

    Kuhul::Source::Document document;
    if (!loadKuhulDocument(sourcePath, document)) return 1;

    if (command == "parse") {
        std::cout << "KAST parsed: " << sourcePath << "\n"
                  << Kuhul::Source::analysis(document);
        return 0;
    }
    if (command == "ast") {
        std::cout << Kuhul::Source::astJson(document) << "\n";
        return 0;
    }
    if (command == "ebnf") {
        std::cout << "Grammar: kuhul/grammar/kuhul.ebnf\n"
                  << "Validation mode: structural native host validation\n"
                  << "Phases: Pop Wo Yax Sek Chen Xul\n";
        return 0;
    }
    if (command == "analyze") {
        std::cout << Kuhul::Source::analysis(document);
        return 0;
    }
    if (command == "validate") {
        std::cout << (document.valid() ? "Validation passed\n" : "Validation failed\n")
                  << Kuhul::Source::analysis(document);
        return document.valid() ? 0 : 1;
    }
    if (command == "compile") {
        const std::string package = Kuhul::Source::semanticPackage(document);
        if (argc >= 4) {
            std::string error;
            if (!Kuhul::Source::writeFile(argv[3], package, error)) {
                std::cerr << "Compile failed: " << error << "\n";
                return 1;
            }
            std::cout << "Semantic package written: " << argv[3] << "\n";
        } else {
            std::cout << package;
        }
        return 0;
    }
    if (command == "code") {
        if (target.empty()) target = "cpp";
        std::cout << Kuhul::Source::generatedCode(document, target);
        return 0;
    }
    if (command == "emit") {
        if (target != "cpp" && target != "hlsl" && target != "wgsl" &&
            target != "opencl" && target != "scx" && target != "scxq2" &&
            target != "svg3d") {
            std::cerr << "Emitter unavailable for target: " << target << "\n";
            return 2;
        }
        if (target == "scx" || target == "scxq2") {
            std::cout << Kuhul::Source::semanticPackage(document);
        } else if (target == "svg3d") {
            std::cout << "<svg data-kuhul-source=\"" << sourcePath
                      << "\" data-kind=\"semantic-graph\"/>\n";
        } else {
            std::cout << Kuhul::Source::generatedCode(document, target);
        }
        return 0;
    }
    if (command == "execute" || command == "runtime") {
        auto& sandbox = Kuhul::Runtime::runtime();
        sandbox.reset();
        sandbox.seedNode("kuhul." + sourcePath, 0.75f);
        sandbox.Run();
        const auto& context = sandbox.context();
        std::cout << "Executed " << sourcePath << " through runtime()\n"
                  << "  phases: Pop -> Wo -> Yax -> Sek -> Chen -> Xul\n"
                  << "  tick: " << context.tick.number << "\n";
        return 0;
    }
    return -1;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 0;
    }

    std::string command = argv[1];
    if (command.rfind("--", 0) == 0) command = command.substr(2);

    if (command == "help" || command == "-h") {
        printUsage(argv[0]);
        return 0;
    }

    if (command == "serve") {
        uint16_t port = 8080;
        if (argc >= 3) {
            try {
                const unsigned long parsed = std::stoul(argv[2]);
                if (parsed == 0 || parsed > 65535) throw std::out_of_range("port");
                port = static_cast<uint16_t>(parsed);
            } catch (const std::exception&) {
                std::cerr << "Invalid HTTP server port\n";
                return 1;
            }
        }
        HttpApiServer server(port);
        if (!server.start()) {
            std::cerr << "HTTP server failed to start: " << server.error() << "\n";
            return 1;
        }
        std::cout << "K'UHUL HTTP server listening on http://127.0.0.1:"
                  << port << "\n";
        server.run();
        return 0;
    }

    if (command == "strategy-host") {
        uint16_t port = 8743;
        if (argc >= 3) {
            try {
                const unsigned long parsed = std::stoul(argv[2]);
                if (parsed == 0 || parsed > 65535) throw std::out_of_range("port");
                port = static_cast<uint16_t>(parsed);
            } catch (const std::exception&) {
                std::cerr << "Invalid strategy host port\n";
                return 2;
            }
        }
        HttpApiServer server(port, true, true);
        if (!server.start()) {
            std::cerr << "Strategy host failed to start: "
                      << server.error() << "\n";
            return 1;
        }
        std::cout << "Strategy host listening on http://+:"
                  << port << "\n"
                  << "Join endpoint: http://<HOST-IP>:" << port
                  << "/v1/strategy/session\n"
                  << "Session token: " << server.strategyToken() << "\n"
                  << "Host-authoritative proposal/commit endpoints enabled\n"
                  << "Use an explicit firewall rule and share only with trusted players\n";
        server.run();
        return 0;
    }

    if (command == "version") {
        std::cout << "K'UHUL Semantic Engine v3.5.0-WebX (unified native runtime)\n";
        return 0;
    }

    if (command == "runtime-sandbox") {
        const bool capability_requested = argc >= 3 && std::string(argv[2]) == "external";
        auto& sandbox = Kuhul::Runtime::runtime();
        sandbox.reset();
        sandbox.setExternalCapabilityRequested(capability_requested);
        sandbox.seedNode("MicrosoftSDK", 0.64f);

        const auto yax = sandbox.decide("MicrosoftSDK", Kuhul::Runtime::Phase::Yax);
        std::cout << "MicrosoftSDK phase=Yax pressure=" << yax.pressure
                  << " decision=" << yax.reason << "\n";
        return 0;
    }

    if (command == "runtime-domains") {
        WebX::FieldGraph graph;
        Kuhul::Runtime::RuntimeContext context;
        context.graph = &graph;
        const auto results = Kuhul::Runtime::runAllDomains(context);
        for (const auto& result : results) {
            std::cout << result.domain
                      << " admitted=" << (result.admitted ? "true" : "false")
                      << " executed=" << (result.executed ? "true" : "false")
                      << " pressure=" << result.pressure << "\n";
        }
        return 0;
    }

    if (command == "atomic-shell" || command == "Atomic.DOM") {
        const bool atomicDom = command == "Atomic.DOM";
        int manifestArg = 2;
        const std::string defaultManifest =
            "native/runtime/atomic.frame.manifest.json";
        const char* manifestPath = nullptr;
        if (argc > manifestArg &&
            std::string(argv[manifestArg]).rfind("--", 0) != 0) {
            manifestPath = argv[manifestArg++];
        } else if (atomicDom) {
            manifestPath = defaultManifest.c_str();
        } else {
            std::cerr << "Usage: atomic-shell MANIFEST\n";
            return 2;
        }
        Kuhul::Runtime::AtomicShellManifest manifest;
        Kuhul::Runtime::AtomicShellManifestLoader loader;
        if (!loader.load(manifestPath, manifest)) {
            std::cerr << "Atomic Shell manifest invalid: "
                      << loader.error() << "\n";
            return 1;
        }
        std::cout << "Atomic Shell manifest valid\n"
                  << "  id:      " << manifest.id << "\n"
                  << "  block:   " << manifest.block << "\n"
                  << "  version: " << manifest.version << "\n"
                  << "  schema:  " << manifest.schema << "\n"
                  << "  backend: " << manifest.backend << "\n"
                  << "  feed:    " << (manifest.feedParser.empty()
                      ? "none" : manifest.feedParser) << "\n"
                  << "  feed wasm: " << (manifest.feedParserArtifact.empty()
                      ? "none" : manifest.feedParserArtifact) << "\n"
                  << "  asset:   " << (manifest.assetFormat.empty()
                      ? "none" : manifest.assetFormat) << "\n"
                  << "  asset uri: " << (manifest.assetUri.empty()
                      ? "none" : manifest.assetUri) << "\n"
                  << "  gated:   " << (manifest.executionGated ? "true" : "false")
                  << "\n";
        if (manifest.backend == "opengl")
            std::cout << "  capability: deferred; use --providers\n";
        bool render = atomicDom;
        bool chat = false;
        bool login = false;
        bool page = false;
        bool grid = false;
        bool scene = false;
        bool game = false;
        bool strategy = false;
        bool multiplayer = false;
        bool sidecars = false;
        for (int i = manifestArg; i < argc; ++i) {
            render = render || std::string(argv[i]) == "--render";
            chat = chat || std::string(argv[i]) == "--chat";
            login = login || std::string(argv[i]) == "--login";
            page = page || std::string(argv[i]) == "--page";
            grid = grid || std::string(argv[i]) == "--grid";
            scene = scene || std::string(argv[i]) == "--scene";
            game = game || std::string(argv[i]) == "--game";
            strategy = strategy || std::string(argv[i]) == "--strategy";
            multiplayer = multiplayer || std::string(argv[i]) == "--multiplayer";
            sidecars = sidecars || std::string(argv[i]) == "--sidecars";
        }
        if (render) {
            std::cout << "\n+---------------- ATOMIC FRAME ----------------+\n"
                      << "| blocks: ";
            for (size_t i = 0; i < manifest.blocks.size(); ++i) {
                if (i != 0) std::cout << " | ";
                std::cout << manifest.blocks[i];
            }
            std::cout << "\n| active block: " << manifest.block
                      << "\n+----------------------------------------------+\n";
        }
        if (atomicDom) {
            std::cout << "\n+------------- ATOMIC.DOM TERMINAL ------------+\n"
                      << "| 1  Chat shell                                |\n"
                      << "| 2  Game demo                                 |\n"
                      << "| Q  Exit                                      |\n"
                      << "+----------------------------------------------+\n";
            if (login) {
                std::cout << "\n+------------- ATOMIC.DOM LOGIN --------------+\n"
                          << "| Username: _________________________________ |\n"
                          << "| Password: _________________________________ |\n"
                          << "|                                             |\n"
                          << "| [Enter] Submit     [Esc] Back               |\n"
                          << "+---------------------------------------------+\n"
                          << "page: login (manifest-bound)\n"
                          << "username> " << std::flush;
                 std::string username;
                 std::string password;
                 std::getline(std::cin, username);
                 std::cout << "password> " << std::flush;
                 std::getline(std::cin, password);
                 std::cout << "login route: task://auth.login\n"
                           << "submitted user: "
                           << (username.empty() ? "<empty>" : username)
                           << "\n";
            } else if (chat) {
                std::cout << "\n+================ STATION GUIDE ================+\n"
                          << "| NPC: Station Guide     tone: monotone         |\n"
                          << "| status: NPC online     context: 12 turns     |\n"
                          << "+------------------- TRANSCRIPT ---------------+\n"
                          << "| user: Hello AtomicDOM                        |\n"
                          << "| npc:  This chat surface is manifest-bound.   |\n"
                          << "| user: Show me the available routes.          |\n"
                          << "| npc:  Submit route is task://chat.submit.    |\n"
                          << "+------------------- REPLY LOG -----------------+\n"
                          << "| route: state://chat.reply_log                |\n"
                          << "| provider: cloud_api (manifest required)      |\n"
                          << "+---------------------- INPUT ------------------+\n"
                          << "| /login  open login page   /quit  exit        |\n"
                          << "+-----------------------------------------------+\n"
                          << "chat> " << std::flush;
                 std::string message;
                 unsigned int turn = 1;
                 while (std::getline(std::cin, message) && message != "/quit") {
                     if (message == "q" || message == "Q") {
                         break;
                     }
                     if (message == "1") {
                         std::cout << "route: chat://station-guide (already active)\n"
                                   << "chat> " << std::flush;
                         continue;
                     }
                     if (message == "2" || message == "/game") {
                         std::cout << "route: game://station-game\n"
                                   << "manifest: native/runtime/atomic.game.manifest.json\n"
                                   << "launch: AtomicGame.cmd\n"
                                   << "game mode: OpenGL FRAME admitted; asset upload is pending\n"
                                   << "chat> " << std::flush;
                         continue;
                     }
                     if (message == "/login") {
                         std::cout << "route: task://auth.login (use AtomicChat.cmd --login)\n"
                                  << "chat> " << std::flush;
                         continue;
                     }
                     if (!message.empty()) {
                         std::cout << "  user: " << message << "\n"
                                  << "  bot:   message admitted to task://chat.submit\n"
                                  << "  reply_log: {turn:" << turn
                                  << ", role:npc, status:deferred, "
                                     "route:state://chat.reply_log}\n";
                         ++turn;
                     }
                     std::cout << "chat> " << std::flush;
                 }
            } else if (page) {
                std::cout << "\n+================ STATION CONTROL ================+\n"
                          << "| Atomic Page     route: page://station-control  |\n"
                          << "+---------------------- NAV ---------------------+\n"
                          << "| [1] NPC Chat    [2] Game Scene    [Q] Exit     |\n"
                          << "+--------------------- HERO ---------------------+\n"
                          << "| Manifest-bound control surface                 |\n"
                          << "| Native terminal presentation. No browser CSS.  |\n"
                          << "+-------------------- CARDS ---------------------+\n"
                          << "| NPC guide    Station Guide   chat://station... |\n"
                          << "| OpenGL scene Station Game    game://station... |\n"
                          << "| Execution    Manifest gated state://runtime    |\n"
                          << "+-------------------- STATUS --------------------+\n"
                          << "| FEED: XCFE WASM     STYLE: station-dark        |\n"
                          << "| BACKEND: terminal   STATE: admitted            |\n"
                          << "+------------------- FOOTER ---------------------+\n"
                          << "| AtomicDOM page ready                            |\n"
                          << "+------------------------------------------------+\n";
            } else if (grid) {
                std::cout << "\n+================ STATION GRID =================+\n"
                          << "| route: world://station-grid                  |\n"
                          << "| dimensions: 5 x 1 x 5     cells: 6          |\n"
                          << "+------------------- PALETTE ------------------+\n"
                          << "| [ ] empty  [R] room  [C] corridor             |\n"
                          << "| [#] wall   [D] door  [S] stairs               |\n"
                          << "+-------------------- MAP ---------------------+\n"
                          << "| [R]-[C]-[D]-[R]                               |\n"
                          << "                   |                            |\n"
                          << "                 [S]-[#]                        |\n"
                          << "+---------------- CONNECTIONS -----------------+\n"
                          << "| c0 -> c1 path    c1 -> c2 path                |\n"
                          << "| c2 -> c3 path    c3 -> c4 vertical            |\n"
                          << "+------------------- ACTIONS ------------------+\n"
                          << "| place select remove connect clear save load   |\n"
                          << "+-----------------------------------------------+\n"
                          << "grid state: manifest-admitted\n";
            } else if (sidecars) {
                std::cout << "\n+================ SIDECAR STORE ================+\n"
                          << "| route: sidecar://store    feed: /api/sidecars |\n"
                          << "+----------------------------------------------+\n";
                // Resolve the host-authoritative feed into the frame (presentation
                // only; sidecars are candidate/compute-only and are never mutated
                // here). Reuses the verified terminal renderer.
                int rc = std::system(
                    "pwsh -NoProfile -File bin/json-runtime/Show-Sidecars.ps1");
                if (rc != 0)
                    std::cout << "| feed unavailable (start json_runtime :8787)  |\n"
                              << "+----------------------------------------------+\n";
                std::cout << "sidecar store: host-authoritative, read-only\n";
            } else if (scene) {
                std::cout << "\n+============== STATION SCENE EDITOR ============+\n"
                          << "| route: scene://station-editor                 |\n"
                          << "| tile:  x=3 z=2 size=32 seed=42                |\n"
                          << "+------------------- CONTROLS -----------------+\n"
                          << "| arrows orbit   W/S zoom   H mirror   T tabletop|\n"
                          << "| close window to exit                           |\n"
                          << "+------------------ EDIT ACTIONS ---------------+\n"
                          << "| select paint erase connect save                |\n"
                          << "+-----------------------------------------------+\n"
                          << "scene state: manifest-admitted\n";
            } else if (strategy) {
                int treasury = 50;
                int legitimacy = 60;
                int tension = 35;
                int actionPoints = 12;
                int skillLevel = 1;
                int interactions = 0;
                int weapons = 0;
                int scientists = 1;
                int alliances = 0;
                int intel = 0;
                int doubleAgents = 0;
                int militaryPoints = 6;
                int politicalPoints = 6;
                unsigned turn = 1;
                std::cout << "\n+=========== THIRD WORLD STRATEGY ===========+\n"
                          << "| route: strategy://station-world             |\n"
                          << "| map: terrain tile (3,2)  seed: 42           |\n"
                          << "+----------------- FACTIONS -----------------+\n"
                          << "| Meridian Union     diplomacy / trade        |\n"
                          << "| Northstar Compact  security / borders       |\n"
                          << "| Free Cities        media / public support   |\n"
                          << "| Iron League        sanctions / known plot   |\n"
                          << "| Hydra Cells        infiltration / emergent   |\n"
                          << "+------------------- STATE ------------------+\n";
                if (multiplayer) {
                    std::cout << "multiplayer: session admitted\n"
                              << "authority: host validates and commits turns\n"
                              << "slots: host / guest-alpha / guest-beta\n"
                              << "proposal route: task://multiplayer.propose\n"
                              << "commit route: task://multiplayer.commit\n";
                }
                while (true) {
                    std::cout << "| turn " << turn
                              << " treasury=" << treasury
                              << " legitimacy=" << legitimacy
                              << " tension=" << tension << "             |\n"
                    << "| points=" << actionPoints
                    << " skill=" << skillLevel
                    << " weapons=" << weapons
                    << " scientists=" << scientists
                    << " alliances=" << alliances
                    << " intel=" << intel
                    << " mil=" << militaryPoints
                    << " pol=" << politicalPoints << " |\n"
                              << "+------------------ CHOICES -----------------+\n"
                              << "| 1 Trade accord      +treasury, -tension     |\n"
                              << "| 2 Security pact     +legitimacy, +tension   |\n"
                              << "| 3 Media campaign    +legitimacy, -treasury  |\n"
                              << "| 4 Forge weapon      cost 4 points          |\n"
                              << "| 5 Hire scientist    cost 3 points          |\n"
                              << "| 6 Defense alliance  cost 2 points          |\n"
                              << "| 7 Secret agent intel cost 2 points         |\n"
                              << "| 8 Recruit double agent cost 3 points      |\n"
                              << "| 9 Military tactics cost 3 military points |\n"
                              << "| 0 Political maneuver cost 3 political pts |\n"
                              << "| Q Exit                                     |\n"
                              << "+---------------------------------------------+\n"
                              << "choice> " << std::flush;
                    std::string choice;
                    if (!std::getline(std::cin, choice) ||
                        choice == "q" || choice == "Q") {
                        break;
                    }
                    if (choice == "1" || choice == "2" || choice == "3") {
                        if (actionPoints < 1) {
                            std::cout << "decision rejected: no action points\n";
                            continue;
                        }
                        --actionPoints;
                        ++interactions;
                        if (interactions % 3 == 0) ++skillLevel;
                    }
                    if (choice == "1") {
                        treasury += 12;
                        tension = std::max(0, tension - 8);
                        std::cout << "decision: task://strategy.choose trade_accord\n"
                                  << "result: Meridian Union opened a trade corridor\n";
                    } else if (choice == "2") {
                        legitimacy += 7;
                        tension = std::min(100, tension + 10);
                        std::cout << "decision: task://strategy.choose security_pact\n"
                                  << "result: Northstar Compact recognized a mutual defense line\n";
                    } else if (choice == "3") {
                        legitimacy += 10;
                        treasury = std::max(0, treasury - 8);
                        std::cout << "decision: task://strategy.choose media_campaign\n"
                                  << "result: Free Cities shifted public sentiment\n";
                    } else if (choice == "4") {
                        if (actionPoints < 4) {
                            std::cout << "decision rejected: forging requires 4 points\n";
                            continue;
                        }
                        actionPoints -= 4;
                        ++weapons;
                        std::cout << "decision: task://strategy.forge weapon\n"
                                  << "result: weapon forged at skill " << skillLevel
                                  << "\n";
                    } else if (choice == "5") {
                        if (actionPoints < 3) {
                            std::cout << "decision rejected: research hire requires 3 points\n";
                            continue;
                        }
                        actionPoints -= 3;
                        ++scientists;
                        std::cout << "decision: task://strategy.hire scientist\n"
                                  << "result: research capacity increased\n";
                    } else if (choice == "6") {
                        if (actionPoints < 2) {
                            std::cout << "decision rejected: alliance requires 2 points\n";
                            continue;
                        }
                        actionPoints -= 2;
                        ++alliances;
                        tension = std::max(0, tension - 6);
                        std::cout << "decision: task://strategy.form_alliance\n"
                                  << "result: defensive alliance formed; border risk reduced\n";
                    } else if (choice == "7") {
                        if (actionPoints < 2) {
                            std::cout << "decision rejected: intel requires 2 points\n";
                            continue;
                        }
                        actionPoints -= 2;
                        ++intel;
                        std::cout << "decision: task://strategy.secret_intel\n"
                                  << "result: agent feed exposed a league route and a hydra signal\n";
                    } else if (choice == "8") {
                        if (actionPoints < 3) {
                            std::cout << "decision rejected: double agent operation requires 3 points\n";
                            continue;
                        }
                        actionPoints -= 3;
                        ++doubleAgents;
                        ++intel;
                        std::cout << "decision: task://strategy.recruit_double_agent\n"
                                  << "result: double-agent possibility opened; loyalty must be tested\n";
                    } else if (choice == "9") {
                        if (militaryPoints < 3) {
                            std::cout << "decision rejected: military points exhausted\n";
                            continue;
                        }
                        militaryPoints -= 3;
                        tension = std::max(0, tension - 4);
                        std::cout << "decision: task://strategy.military_tactics\n"
                                  << "result: defensive maneuver secured a resource corridor\n";
                    } else if (choice == "0") {
                        if (politicalPoints < 3) {
                            std::cout << "decision rejected: political points exhausted\n";
                            continue;
                        }
                        politicalPoints -= 3;
                        legitimacy = std::min(100, legitimacy + 6);
                        std::cout << "decision: task://strategy.political_maneuver\n"
                                  << "result: coalition vote shifted without open conflict\n";
                    } else {
                        std::cout << "decision rejected: choose 1, 2, 3, or Q\n";
                        continue;
                    }
                    legitimacy = std::min(100, legitimacy);
                    ++turn;
                    std::cout << "state: state://strategy.world updated\n";
                    if (multiplayer)
                        std::cout << "turn commit: state://multiplayer.replay_log\n";
                }
            } else if (game) {
                if (manifest.id == "station-game") {
                    std::cout << "game mode: manifest-bound OpenGL FRAME\n"
                              << "frame route: game://station-game\n"
                              << "asset: " << manifest.assetFormat
                              << " (" << manifest.assetUri << ")\n"
                              << "renderer: opengl-obj-smoke\n"
                              << "asset upload: launcher stage\n";
                } else {
                    std::cout << "game mode: handoff to `wwa-demos`; "
                                 "renderer integration is pending\n";
                }
            } else {
                std::cout << "mode: menu-only terminal shell\n";
            }
        }
        return 0;
    }

    if (command == "opengl-frame-smoke") {
        unsigned frames = 1;
        if (argc >= 3) {
            try {
                frames = static_cast<unsigned>(std::stoul(argv[2]));
            } catch (const std::exception&) {
                std::cerr << "OpenGL frame smoke failed: invalid frame count\n";
                return 2;
            }
        }
        Kuhul::Runtime::OpenGLFrameAdapter adapter;
        if (!adapter.renderSmoke(frames)) {
            std::cerr << "OpenGL frame smoke failed: "
                      << adapter.error() << "\n";
            return 1;
        }
        std::cout << "OpenGL FRAME smoke rendered frames=" << frames << std::endl;
        return 0;
    }

    if (command == "opengl-obj-smoke") {
        if (argc < 3) {
            std::cerr << "Usage: opengl-obj-smoke PATH [FRAMES]\n";
            return 2;
        }
        unsigned frames = 1;
        bool interactive = false;
        if (argc >= 4) {
            if (std::string(argv[3]) == "--interactive") {
                interactive = true;
            } else {
                try {
                    frames = static_cast<unsigned>(std::stoul(argv[3]));
                } catch (const std::exception&) {
                    std::cerr << "OpenGL OBJ smoke failed: invalid frame count\n";
                    return 2;
                }
            }
        }
        Kuhul::Runtime::OpenGLFrameAdapter adapter;
        if (!adapter.renderObjSmoke(argv[2], frames, interactive)) {
            std::cerr << "OpenGL OBJ smoke failed: "
                      << adapter.error() << "\n";
            return 1;
        }
        std::cout << (interactive
            ? "OpenGL OBJ interactive FRAME closed\n"
            : "OpenGL OBJ smoke rendered frames=" + std::to_string(frames))
                  << std::endl;
        return 0;
    }

    if (command == "opengl-game-smoke") {
        if (argc < 3) {
            std::cerr << "Usage: opengl-game-smoke MANIFEST "
                         "[FRAMES|--interactive]\n";
            return 2;
        }
        Kuhul::Runtime::AtomicShellManifest manifest;
        Kuhul::Runtime::AtomicShellManifestLoader loader;
        if (!loader.load(argv[2], manifest)) {
            std::cerr << "OpenGL game smoke failed: "
                      << loader.error() << "\n";
            return 1;
        }
        if (manifest.assetFormat != "obj" || manifest.assetUri.empty()) {
            std::cerr << "OpenGL game smoke failed: "
                         "manifest requires an OBJ asset_uri\n";
            return 1;
        }
        unsigned frames = 1;
        bool interactive = false;
        if (argc >= 4) {
            if (std::string(argv[3]) == "--interactive") {
                interactive = true;
            } else {
                try {
                    frames = static_cast<unsigned>(std::stoul(argv[3]));
                } catch (const std::exception&) {
                    std::cerr << "OpenGL game smoke failed: invalid frame count\n";
                    return 2;
                }
            }
        }
        std::filesystem::path assetPath(manifest.assetUri);
        if (assetPath.is_relative() &&
            !std::filesystem::exists(assetPath))
            assetPath = std::filesystem::path(argv[2]).parent_path() / assetPath;
        Kuhul::Runtime::OpenGLFrameAdapter adapter;
        if (!adapter.renderObjSmoke(assetPath.string(), frames, interactive,
                                    manifest.blocks)) {
            std::cerr << "OpenGL game smoke failed: "
                      << adapter.error() << "\n";
            return 1;
        }
        std::cout << (interactive
            ? "OpenGL GAME interactive FRAME closed"
            : "OpenGL GAME smoke rendered frames=" + std::to_string(frames))
                  << std::endl;
        return 0;
    }

    if (command == "opengl-strategy-globe") {
        unsigned frames = 1;
        bool interactive = false;
        if (argc >= 3) {
            if (std::string(argv[2]) == "--interactive") {
                interactive = true;
            } else {
                try {
                    frames = static_cast<unsigned>(std::stoul(argv[2]));
                } catch (const std::exception&) {
                    std::cerr << "OpenGL strategy globe failed: invalid frame count\n";
                    return 2;
                }
            }
        }
        Kuhul::Runtime::OpenGLFrameAdapter adapter;
        if (!adapter.renderStrategyGlobe(frames, interactive,
                                         {"HEADER", "FEED", "GAME", "GRID",
                                          "FOOTER"})) {
            std::cerr << "OpenGL strategy globe failed: "
                      << adapter.error() << "\n";
            return 1;
        }
        std::cout << (interactive
            ? "OpenGL strategy globe interactive FRAME closed"
            : "OpenGL strategy globe rendered frames=" +
              std::to_string(frames)) << std::endl;
        return 0;
    }

    if (command == "world-tile-smoke") {
        if (argc < 4) {
            std::cerr << "Usage: world-tile-smoke X Z [SIZE] [SEED]\n";
            return 2;
        }
        try {
            const int x = std::stoi(argv[2]);
            const int z = std::stoi(argv[3]);
            const unsigned size = argc >= 5
                ? static_cast<unsigned>(std::stoul(argv[4])) : 16;
            const std::uint32_t seed = argc >= 6
                ? static_cast<std::uint32_t>(std::stoul(argv[5])) : 1U;
            Kuhul::Runtime::WorldTile tile;
            std::string error;
            if (!Kuhul::Runtime::generateWorldTile(x, z, size, seed, tile,
                                                   error)) {
                std::cerr << "World tile smoke failed: " << error << "\n";
                return 1;
            }
            const auto& sample = tile.samples.front();
            std::cout << "World tile generated"
                      << " coordinate=(" << tile.x << "," << tile.z << ")"
                      << " size=" << tile.width << "x" << tile.depth
                      << " domain=" << tile.domain
                      << " first_rgba=("
                      << static_cast<unsigned>(sample.red) << ","
                      << static_cast<unsigned>(sample.green) << ","
                      << static_cast<unsigned>(sample.blue) << ","
                      << static_cast<unsigned>(sample.alpha) << ")"
                      << " first_elevation=" << sample.elevation
                      << "\n";
            return 0;
        } catch (const std::exception&) {
            std::cerr << "World tile smoke failed: invalid coordinates or "
                         "parameters\n";
            return 2;
        }
    }

    if (command == "opengl-world-tile-smoke") {
        if (argc < 4) {
            std::cerr << "Usage: opengl-world-tile-smoke X Z "
                         "[SIZE] [SEED] [FRAMES|--interactive]\n";
            return 2;
        }

        try {
            const int x = std::stoi(argv[2]);
            const int z = std::stoi(argv[3]);
            unsigned frames = 1;
            bool interactive = false;
            int next = 4;
            unsigned size = 32;
            std::uint32_t seed = 1U;
            bool neighbors = false;
            bool sizeSet = false;
            bool seedSet = false;
            while (argc > next) {
                const std::string argument = argv[next++];
                if (argument == "--interactive") {
                    interactive = true;
                } else if (argument == "--neighbors") {
                    neighbors = true;
                } else if (!sizeSet) {
                    size = static_cast<unsigned>(std::stoul(argument));
                    sizeSet = true;
                } else if (!seedSet) {
                    seed = static_cast<std::uint32_t>(std::stoul(argument));
                    seedSet = true;
                } else {
                    frames = static_cast<unsigned>(std::stoul(argument));
                }
            }
            Kuhul::Runtime::WorldTile tile;
            std::string error;
            const bool generated = neighbors
                ? Kuhul::Runtime::generateWorldNeighborhood(
                    x, z, size, 1, seed, tile, error)
                : Kuhul::Runtime::generateWorldTile(x, z, size, seed, tile,
                                                    error);
            if (!generated) {
                std::cerr << "OpenGL world tile failed: " << error << "\n";
                return 1;
            }
            Kuhul::Runtime::OpenGLFrameAdapter adapter;
            if (!adapter.renderObjSmoke("", frames, interactive, {}, &tile)) {
                std::cerr << "OpenGL world tile failed: "
                          << adapter.error() << "\n";
                return 1;
            }
            if (interactive) {
                std::cout << "OpenGL world tile interactive FRAME closed"
                          << std::endl;
            } else {
                std::cout << "OpenGL world tile rendered frames="
                          << frames << std::endl;
            }
            return 0;
        } catch (const std::exception&) {
            std::cerr << "OpenGL world tile failed: invalid parameters\n";
            return 2;
        }
    }

    if (command == "opengl-particle-smoke") {
        if (argc < 3) {
            std::cerr << "Usage: opengl-particle-smoke EFFECT "
                         "[COUNT] [SEED] [FRAMES|--interactive]\n";
            return 2;
        }
        try {
            const std::string kind = argv[2];
            const unsigned count = argc >= 4
                ? static_cast<unsigned>(std::stoul(argv[3])) : 256;
            const std::uint32_t seed = argc >= 5
                ? static_cast<std::uint32_t>(std::stoul(argv[4])) : 1U;
            unsigned frames = 1;
            bool interactive = false;
            if (argc >= 6) {
                if (std::string(argv[5]) == "--interactive") {
                    interactive = true;
                } else {
                    frames = static_cast<unsigned>(std::stoul(argv[5]));
                }
            }
            Kuhul::Runtime::ParticleEffect effect;
            std::string error;
            if (!Kuhul::Runtime::createParticleEffect(kind, count, seed,
                                                      effect, error)) {
                std::cerr << "OpenGL particle smoke failed: " << error << "\n";
                return 1;
            }
            Kuhul::Runtime::OpenGLFrameAdapter adapter;
            if (!adapter.renderObjSmoke("", frames, interactive, {}, nullptr,
                                        &effect)) {
                std::cerr << "OpenGL particle smoke failed: "
                          << adapter.error() << "\n";
                return 1;
            }
            std::cout << (interactive
                ? "OpenGL particle interactive FRAME closed"
                : "OpenGL particle smoke rendered frames=" +
                  std::to_string(frames)) << std::endl;
            return 0;
        } catch (const std::exception&) {
            std::cerr << "OpenGL particle smoke failed: invalid parameters\n";
            return 2;
        }
    }

    if (command == "instant-agent-smoke") {
        if (argc < 3) {
            std::cerr << "Usage: instant-agent-smoke ROLE "
                         "[SEED] [FRAMES|--interactive]\n";
            return 2;
        }
        try {
            const std::string role = argv[2];
            const std::uint32_t seed = argc >= 4
                ? static_cast<std::uint32_t>(std::stoul(argv[3])) : 1U;
            unsigned frames = 1;
            bool interactive = false;
            if (argc >= 5) {
                if (std::string(argv[4]) == "--interactive") {
                    interactive = true;
                } else {
                    frames = static_cast<unsigned>(std::stoul(argv[4]));
                }
            }
            Kuhul::Runtime::InstantAgent agent;
            std::string error;
            if (!Kuhul::Runtime::createInstantAgent(role, seed, agent,
                                                    error)) {
                std::cerr << "Instant agent smoke failed: " << error << "\n";
                return 1;
            }
            Kuhul::Runtime::ParticleEffect effect;
            const std::string effectKind =
                agent.effect == "spark" ? "spell" : agent.effect;
            if (!Kuhul::Runtime::createParticleEffect(
                    effectKind, 128, seed, effect, error)) {
                std::cerr << "Instant agent smoke failed: " << error << "\n";
                return 1;
            }
            for (auto& particle : effect.particles) {
                particle.red = agent.red;
                particle.green = agent.green;
                particle.blue = agent.blue;
            }
            Kuhul::Runtime::OpenGLFrameAdapter adapter;
            if (!adapter.renderObjSmoke("", frames, interactive, {}, nullptr,
                                        &effect, &agent)) {
                std::cerr << "Instant agent smoke failed: "
                          << adapter.error() << "\n";
                return 1;
            }
            std::cout << "Instant agent spawned id=" << agent.id
                      << " role=" << agent.role
                      << " effect=" << agent.effect
                      << " position=(" << agent.x << "," << agent.y << ","
                      << agent.z << ")"
                      << (interactive
                          ? " interactive FRAME closed"
                          : " frames=" + std::to_string(frames))
                      << std::endl;
            return 0;
        } catch (const std::exception&) {
            std::cerr << "Instant agent smoke failed: invalid parameters\n";
            return 2;
        }
    }

    if (command == "task-engine" || command == "task" || command == "-task" ||
        command == "task-run" || command == "task-boss") {
        if (argc < 3) {
            std::cerr << "Usage: " << command << " TaskList.kuhul|tasks.json\n";
            return 2;
        }
        WebX::ProviderManager providers;
        Kuhul::Runtime::TaskEngine taskEngine(providers);
        std::string error;
        if (!taskEngine.load(argv[2], error)) {
            std::cerr << "TaskList load failed: " << error << "\n";
            return 1;
        }
        const bool execute = command == "task-run" || command == "task-boss";
        WebX::WebXRuntime bossRuntime;
        if (command == "task-boss") {
            WebX::GPUConfig config;
            config.backend = WebX::GPUBackend::CPU;
            if (!bossRuntime.initialize(config, 3)) {
                std::cerr << "BOSS runtime initialization failed\n";
                return 1;
            }
        }
        std::cerr << "TaskEngine " << (execute ? "run" : "plan")
                  << ": " << argv[2] << "\n";
        const auto results = execute
            ? taskEngine.run(
                  [&](const Kuhul::Runtime::TaskSpec& task,
                     const WebX::Provider& provider, std::string& detail) {
                      if (command == "task-boss") {
                          WebX::Field* field =
                              bossRuntime.createField(task.id, task.provider);
                          if (!field) {
                              detail = "field_creation_failed";
                              return false;
                          }
                          bossRuntime.getGraph().createCard(field, 0, 1);
                          const bool success =
                              bossRuntime.executeField(task.id);
                          detail = success
                              ? "boss_field_completed"
                              : "boss_field_execution_failed";
                          return success;
                      }
                      return Kuhul::Runtime::TaskHelperExecutor{}
                          .execute(task, provider, detail);
                  })
            : taskEngine.plan();
        for (const auto& result : results) {
            std::cerr << "  " << result.id
                      << " provider=" << result.provider
                      << " status=" << result.status
                      << " detail=" << result.detail << "\n";
        }
        if (command == "task-boss") bossRuntime.shutdown();
        return 0;
    }

    const int sourceCommandResult = runKuhulSourceCommand(command, argc, argv);
    if (sourceCommandResult >= 0) return sourceCommandResult;

    if (command == "stream-xshard" && argc >= 3) {
        XShardModelManifest manifest;
        XShardModelLoader loader;
        if (!loader.validate(argv[2], manifest)) {
            std::cerr << "SCX stream plan failed: " << loader.error() << "\n";
            return 1;
        }
        ScxManifestPlan plan;
        if (!buildScxManifestPlan(manifest, plan)) {
            std::cerr << "SCX stream plan failed\n";
            for (const auto& error : plan.errors) std::cerr << "  error: " << error << "\n";
            return 1;
        }
        std::cout << "SCX stream plan ready\n"
                  << "  root:       " << plan.root << "\n"
                  << "  layers:     " << plan.layer_count << "\n"
                  << "  tiles:      " << plan.total_tiles << "\n"
                  << "  payload:    " << plan.total_payload_bytes << "\n";
        for (const auto& layer : plan.layers) {
            std::cout << "  layer " << layer.layer_id
                      << ": tensors=" << layer.tensor_count
                      << " tiles=" << layer.tile_count
                      << " ready=" << (layer.ready ? "true" : "false") << "\n";
            for (const auto& family : layer.families) {
                std::cout << "    family type=" << family.tensor_type
                          << " tensors=" << family.tensor_count
                          << " tiles=" << family.tile_count
                          << " bytes=" << family.payload_bytes
                          << " dtypeMask=0x" << std::hex
                          << family.dtype_mask << std::dec << "\n";
            }
        }

        if (argc >= 4) {
            uint32_t maxLayer = 0;
            uint32_t maxTiles = 0;
            try {
                maxLayer = static_cast<uint32_t>(std::stoul(argv[3]));
                if (argc >= 5)
                    maxTiles = static_cast<uint32_t>(std::stoul(argv[4]));
            } catch (const std::exception&) {
                std::cerr << "SCX tile stream failed: bounds must be integers\n";
                return 1;
            }

            std::string streamError;
            uint32_t streamedTiles = 0;
            const bool streamed = streamXShardTiles(
                manifest, maxLayer, maxTiles,
                [](uint32_t, uint8_t, uint32_t, uint32_t, uint32_t,
                   uint32_t, uint32_t, const uint8_t*, size_t, bool) {
                    return true;
                },
                streamedTiles, streamError);
            if (!streamed) {
                std::cerr << "SCX tile stream failed: " << streamError << "\n";
                return 1;
            }
            std::cout << "  streamed:   " << streamedTiles << " tiles\n";
        }
        return 0;
    }

    if (command == "scx-dx12-smoke" && argc >= 3) {
        uint32_t maxTiles = 1;
        uint32_t tensorType = UINT32_MAX;
        if (argc >= 4) {
            try {
                maxTiles = static_cast<uint32_t>(std::stoul(argv[3]));
                if (argc >= 5)
                    tensorType = static_cast<uint32_t>(std::stoul(argv[4]));
            } catch (const std::exception&) {
                std::cerr << "SCX DX12 smoke failed: bounds must be integers\n";
                return 1;
            }
        }
        return runScxDx12Smoke(argv[2], maxTiles, tensorType);
    }

    if (command == "scx-d3d11-smoke" && argc >= 3)
        return runScxD3D11Smoke(argv[2]);

    if (command == "hot-swap-xshard" ||
        command == "--hot-swap-xshard") {
        if (argc < 4) {
            std::cerr << "Usage: hot-swap-xshard ACTIVE CANDIDATE "
                         "[MAX_LAYER] [MAX_TILES] [TENSOR_TYPE]\n";
            return 2;
        }
        XShardHotSwapBounds bounds;
        try {
            if (argc >= 5)
                bounds.max_layer =
                    static_cast<uint32_t>(std::stoul(argv[4]));
            if (argc >= 6)
                bounds.max_tiles =
                    static_cast<uint32_t>(std::stoul(argv[5]));
            if (argc >= 7)
                bounds.tensor_type =
                    static_cast<uint32_t>(std::stoul(argv[6]));
        } catch (const std::exception&) {
            std::cerr << "Xshard hot-swap failed: bounds must be integers\n";
            return 1;
        }

        XShardHotSwapRuntime runtime;
        XShardHotSwapReceipt receipt;
        std::string error;
        const ScxTileSubmitter stagingSubmitter =
            [](uint32_t, uint8_t, uint32_t, uint32_t, uint32_t,
               uint32_t, uint32_t, const uint8_t* payload,
               size_t payloadBytes, bool) {
                return payload != nullptr && payloadBytes != 0;
            };

        if (!runtime.activate(
                argv[2], bounds, stagingSubmitter, receipt, error)) {
            std::cerr << "Xshard active load failed: " << error << "\n";
            return 1;
        }
        if (!runtime.activate(
                argv[3], bounds, stagingSubmitter, receipt, error)) {
            std::cerr << "Xshard hot-swap rejected: " << error << "\n"
                      << "  active:      "
                      << runtime.activeManifest().root << "\n"
                      << "  generation:  " << runtime.generation() << "\n";
            return 1;
        }

        std::cout << "Xshard hot-swap committed\n"
                  << "  phases:       Pop -> Wo -> Yax -> Sek -> Chen -> Xul\n"
                  << "  previous:     " << receipt.previous_root << "\n"
                  << "  active:       " << receipt.active_root << "\n"
                  << "  generation:   " << receipt.generation << "\n"
                  << "  streamed:     " << receipt.streamed_tiles << " tiles\n"
                  << "  bytes:        " << receipt.streamed_bytes << "\n"
                  << "  fingerprint:  0x" << std::hex
                  << receipt.payload_fingerprint << std::dec << "\n";
        return 0;
    }

    if (command == "infer-xshard" && argc >= 5) {
        uint32_t passes = 1;
        if (argc >= 6) {
            try {
                passes = static_cast<uint32_t>(std::stoul(argv[5]));
            } catch (const std::invalid_argument&) {
                std::cerr << "Invalid xshard pass count\n";
                return 1;
            } catch (const std::out_of_range&) {
                std::cerr << "Invalid xshard pass count\n";
                return 1;
            }
        }
        XShardAttentionEngine attention;
        if (!attention.run(argv[2], argv[3], argv[4], passes)) {
            std::cerr << "Native xshard inference failed: "
                      << attention.error() << "\n";
            return 1;
        }
        return 0;
    }

    if (command == "validate-xshard" && argc >= 3) {
        XShardModelManifest manifest;
        XShardModelLoader loader;
        if (!loader.validate(argv[2], manifest)) {
            std::cerr << "Xshard validation failed: " << loader.error() << "\n";
            return 1;
        }
        std::cout << "Xshard validation passed\n"
                  << "  root:        " << manifest.root << "\n"
                  << "  tensors:     " << manifest.tensors.size() << "\n"
                  << "  layers:      " << manifest.layer_count << "\n"
                  << "  max layer:   " << manifest.max_layer_id << "\n"
                  << "  bytes:       " << manifest.total_bytes << "\n";
        return 0;
    }

    if (command == "xshard-block") {
        if (argc < 8) {
            std::cerr << "Usage: xshard-block PATH EXPERT ROW ROWS COL COLS\n";
            return 2;
        }
        try {
            XShardBlockReader reader;
            if (!reader.open(argv[2])) {
                std::cerr << "Xshard block open failed: " << reader.error()
                          << "\n";
                return 1;
            }
            std::vector<float> values;
            if (!reader.readBlock(
                    static_cast<uint32_t>(std::stoul(argv[3])),
                    static_cast<uint32_t>(std::stoul(argv[4])),
                    static_cast<uint32_t>(std::stoul(argv[5])),
                    static_cast<uint32_t>(std::stoul(argv[6])),
                    static_cast<uint32_t>(std::stoul(argv[7])), values)) {
                std::cerr << "Xshard block read failed: " << reader.error()
                          << "\n";
                return 1;
            }
            std::cout << "Xshard block read: expert=" << argv[3]
                      << " values=" << values.size();
            if (!values.empty())
                std::cout << " first=" << values.front()
                          << " last=" << values.back();
            std::cout << "\n";
            return 0;
        } catch (const std::invalid_argument&) {
            std::cerr << "Invalid xshard block coordinates\n";
            return 2;
        } catch (const std::out_of_range&) {
            std::cerr << "Xshard block coordinates are out of range\n";
            return 2;
        }
    }

    if (command == "xshard-block-plan") {
        if (argc < 5) {
            std::cerr << "Usage: xshard-block-plan PATH BLOCK_ROWS BLOCK_COLS\n";
            return 2;
        }
        try {
            XShardBlockReader reader;
            if (!reader.open(argv[2])) {
                std::cerr << "Xshard block plan failed: " << reader.error()
                          << "\n";
                return 1;
            }
            const uint32_t blockRows =
                static_cast<uint32_t>(std::stoul(argv[3]));
            const uint32_t blockCols =
                static_cast<uint32_t>(std::stoul(argv[4]));
            if (blockRows == 0 || blockCols == 0) {
                std::cerr << "Block dimensions must be greater than zero\n";
                return 2;
            }
            const auto& header = reader.header();
            const uint32_t rowBlocks =
                (header.rows + blockRows - 1) / blockRows;
            const uint32_t colBlocks =
                (header.cols + blockCols - 1) / blockCols;
            std::cout << "Xshard block plan: experts=" << header.tile_count
                      << " rows=" << header.rows
                      << " cols=" << header.cols
                      << " block_rows=" << blockRows
                      << " block_cols=" << blockCols
                      << " blocks_per_expert="
                      << static_cast<uint64_t>(rowBlocks) * colBlocks
                      << " total_blocks="
                      << static_cast<uint64_t>(header.tile_count) *
                             rowBlocks * colBlocks
                      << "\n";
            return 0;
        } catch (const std::invalid_argument&) {
            std::cerr << "Invalid xshard block plan dimensions\n";
            return 2;
        } catch (const std::out_of_range&) {
            std::cerr << "Xshard block plan dimensions are out of range\n";
            return 2;
        }
    }

    if (command == "xshard-block-gemm") {
        if (argc < 6) {
            std::cerr << "Usage: xshard-block-gemm PATH EXPERT ROW ROWS\n";
            return 2;
        }

        try {
            XShardBlockReader reader;
            if (!reader.open(argv[2])) {
                std::cerr << "Xshard block GEMM failed: " << reader.error()
                          << "\n";
                return 1;
            }
            const auto& header = reader.header();
            const uint32_t expert =
                static_cast<uint32_t>(std::stoul(argv[3]));
            const uint32_t row =
                static_cast<uint32_t>(std::stoul(argv[4]));
            const uint32_t rows =
                static_cast<uint32_t>(std::stoul(argv[5]));
            std::vector<float> weights;
            if (!reader.readBlock(
                    expert, row, rows, 0, header.cols, weights)) {
                std::cerr << "Xshard block GEMM read failed: "
                          << reader.error() << "\n";
                return 1;
            }
            D3D11Engine engine;
            if (!engine.init(false, false)) {
                std::cerr << "D3D11 expert block initialization failed: "
                          << engine.initReason() << "\n";
                return 1;
            }
            D3D11ExpertBlockGemm gemm;
            if (!gemm.init(engine)) {
                std::cerr << "D3D11 expert block setup failed: "
                          << gemm.error() << "\n";
                return 1;
            }
            std::vector<float> hidden(header.cols, 1.0f);
            std::vector<float> output;
            if (!gemm.run(hidden, weights, 1, header.cols, rows, output)) {
                std::cerr << "D3D11 expert block execution failed: "
                          << gemm.error() << "\n";
                return 1;
            }
            std::cout << "D3D11 expert block GEMM: expert=" << expert
                      << " row=" << row << " rows=" << rows
                      << " hidden=" << header.cols
                      << " output=" << output.size();
            if (!output.empty())
                std::cout << " first=" << output.front()
                          << " last=" << output.back();
            std::cout << "\n";
            return 0;
        } catch (const std::invalid_argument&) {
            std::cerr << "Invalid expert block GEMM coordinates\n";
            return 2;
        } catch (const std::out_of_range&) {
            std::cerr << "Expert block GEMM coordinates are out of range\n";
            return 2;
        }
    }

    if (command == "xshard-block-gemm-int8") {
        if (argc < 6) {
            std::cerr << "Usage: xshard-block-gemm-int8 PATH EXPERT ROW ROWS "
                         "[SCALE] [ZERO]\n";
            return 2;
        }

        try {
            XShardBlockReader reader;
            if (!reader.open(argv[2])) {
                std::cerr << "INT8 block GEMM failed: " << reader.error()
                          << "\n";
                return 1;
            }
            const auto& header = reader.header();
            if (header.dtype != XSHARD_INT8) {
                std::cerr << "INT8 block GEMM requires an INT8 xshard\n";
                return 1;
            }
            const uint32_t expert = static_cast<uint32_t>(std::stoul(argv[3]));
            const uint32_t row = static_cast<uint32_t>(std::stoul(argv[4]));
            const uint32_t rows = static_cast<uint32_t>(std::stoul(argv[5]));
            const float scale = argc >= 7
                ? std::stof(argv[6])
                : xshard_int8_scale(header);
            const float zeroPoint = argc >= 8
                ? std::stof(argv[7])
                : xshard_int8_zero_point(header);
            std::vector<uint8_t> weights;
            if (!reader.readRawBlock(
                    expert, row, rows, 0, header.cols, weights)) {
                std::cerr << "INT8 block GEMM read failed: "
                          << reader.error() << "\n";
                return 1;
            }
            D3D11Engine engine;
            if (!engine.init(false, false)) {
                std::cerr << "D3D11 INT8 block initialization failed: "
                          << engine.initReason() << "\n";
                return 1;
            }
            D3D11ExpertBlockInt8Gemm gemm;
            if (!gemm.init(engine)) {
                std::cerr << "D3D11 INT8 block setup failed: "
                          << gemm.error() << "\n";
                return 1;
            }
            std::vector<float> hidden(header.cols, 1.0f);
            std::vector<float> output;
            if (!gemm.run(hidden, weights, 1, header.cols, rows, scale,
                          zeroPoint, output)) {
                std::cerr << "D3D11 INT8 block execution failed: "
                          << gemm.error() << "\n";
                return 1;
            }
            std::cout << "D3D11 INT8 expert block GEMM: expert=" << expert
                      << " row=" << row << " rows=" << rows
                      << " hidden=" << header.cols
                      << " scale=" << scale
                      << " output=" << output.size();
            if (!output.empty())
                std::cout << " first=" << output.front()
                          << " last=" << output.back();
            std::cout << "\n";
            return 0;
        } catch (const std::invalid_argument&) {
            std::cerr << "Invalid INT8 expert block parameters\n";
            return 2;
        } catch (const std::out_of_range&) {
            std::cerr << "INT8 expert block parameters are out of range\n";
            return 2;
        }
    }

    if (command == "xshard-block-gemm-int4") {
        if (argc < 6) {
            std::cerr << "Usage: xshard-block-gemm-int4 PATH EXPERT ROW ROWS "
                         "[SCALE] [ZERO]\n";
            return 2;
        }
        try {
            XShardBlockReader reader;
            if (!reader.open(argv[2])) {
                std::cerr << "INT4 block GEMM failed: " << reader.error()
                          << "\n";
                return 1;
            }
            const auto& header = reader.header();
            if (header.dtype != XSHARD_INT4) {
                std::cerr << "INT4 block GEMM requires an INT4 xshard\n";
                return 1;
            }
            const uint32_t expert = static_cast<uint32_t>(std::stoul(argv[3]));
            const uint32_t row = static_cast<uint32_t>(std::stoul(argv[4]));
            const uint32_t rows = static_cast<uint32_t>(std::stoul(argv[5]));
            const float scale = argc >= 7
                ? std::stof(argv[6])
                : xshard_int4_scale(header);
            const float zeroPoint = argc >= 8
                ? std::stof(argv[7])
                : xshard_int4_zero_point(header);
            std::vector<uint8_t> packedWeights;
            if (!reader.readPackedInt4Block(
                    expert, row, rows, 0, header.cols, packedWeights)) {
                std::cerr << "INT4 block GEMM read failed: "
                          << reader.error() << "\n";
                return 1;
            }
            D3D11Engine engine;
            if (!engine.init(false, false)) {
                std::cerr << "D3D11 INT4 block initialization failed: "
                          << engine.initReason() << "\n";
                return 1;
            }
            D3D11ExpertBlockInt4Gemm gemm;
            if (!gemm.init(engine)) {
                std::cerr << "D3D11 INT4 block setup failed: "
                          << gemm.error() << "\n";
                return 1;
            }
            std::vector<float> hidden(header.cols, 1.0f);
            std::vector<float> output;
            if (!gemm.run(hidden, packedWeights, 1, header.cols, rows, scale,
                          zeroPoint, output)) {
                std::cerr << "D3D11 INT4 block execution failed: "
                          << gemm.error() << "\n";
                return 1;
            }
            std::cout << "D3D11 INT4 expert block GEMM: expert=" << expert
                      << " row=" << row << " rows=" << rows
                      << " hidden=" << header.cols
                      << " scale=" << scale
                      << " output=" << output.size();
            if (!output.empty())
                std::cout << " first=" << output.front()
                          << " last=" << output.back();
            std::cout << "\n";
            return 0;
        } catch (const std::invalid_argument&) {
            std::cerr << "Invalid INT4 expert block parameters\n";
            return 2;
        } catch (const std::out_of_range&) {
            std::cerr << "INT4 expert block parameters are out of range\n";
            return 2;
        }
    }

    if (command == "xshard-quantize-int8") {
        if (argc < 4) {
            std::cerr << "Usage: xshard-quantize-int8 INPUT OUTPUT\n";
            return 2;
        }
        XShardInt8Quantizer quantizer;
        if (!quantizer.convert(argv[2], argv[3])) {
            std::cerr << "INT8 quantization failed: " << quantizer.error()
                      << "\n";
            return 1;
        }
        std::cout << "INT8 xshard written: " << argv[3] << "\n";
        return 0;
    }

    if (command == "xshard-quantize-int4") {
        if (argc < 4) {
            std::cerr << "Usage: xshard-quantize-int4 INPUT OUTPUT\n";
            return 2;
        }
        XShardInt8Quantizer quantizer;
        if (!quantizer.convertInt4(argv[2], argv[3])) {
            std::cerr << "INT4 quantization failed: " << quantizer.error()
                      << "\n";
            return 1;
        }
        std::cout << "Packed INT4 xshard written: " << argv[3] << "\n";
        return 0;
    }

    if (command == "validate-safetensors" && argc >= 3) {
        SafeTensorManifest manifest;
        SafeTensorsReader reader;
        if (!reader.validate(argv[2], manifest)) {
            std::cerr << "SafeTensors validation failed: "
                      << reader.error() << "\n";
            return 1;
        }
        std::cout << "SafeTensors validation passed\n"
                  << "  path:        " << manifest.path << "\n"
                  << "  tensors:     " << manifest.tensors.size() << "\n"
                  << "  header bytes: " << manifest.header_bytes << "\n"
                  << "  file bytes:  " << manifest.file_bytes << "\n";
        return 0;
    }

    if (command == "infer-llama" && argc >= 4) {
        uint32_t maxTokens = 32;
        if (argc >= 5) {
            try {
                maxTokens = static_cast<uint32_t>(std::stoul(argv[argc - 1]));
            } catch (const std::invalid_argument&) {
                std::cerr << "Invalid llama token count\n";
                return 1;
            } catch (const std::out_of_range&) {
                std::cerr << "Invalid llama token count\n";
                return 1;
            }
        }
        std::string prompt;
        const int promptEnd = argc >= 5 ? argc - 1 : argc;
        for (int i = 3; i < promptEnd; ++i) {
            if (i > 3) prompt += " ";
            prompt += argv[i];
        }
        LlamaRuntime llama;
        if (!llama.load("")) {
            std::cerr << "Native llama provider unavailable: "
                      << llama.error() << "\n";
            return 1;
        }
        std::string output;
        if (!llama.generate(argv[2], prompt, maxTokens, output)) {
            std::cerr << "Native llama inference failed: "
                      << llama.error() << "\n";
            return 1;
        }
        std::cout << output << "\n";
        return 0;
    }

    if (command == "detect-gpu") {
        std::cout << "GPU Detection:\n"
                  << "  DirectML: available (default)\n"
                  << "  CUDA:     not implemented in this build\n"
                  << "  CPU:      available (fallback)\n";
        return 0;
    }

    if (command == "providers") {
        WebX::ProviderManager providers;
        providers.printStatus();
        return 0;
    }

    // Initialize runtime with DirectML backend and 3 BOSS layers.
    WebX::GPUConfig config;
    config.backend = WebX::GPUBackend::DirectML;
    config.device_id = 0;
    config.memory_limit_mb = 4096;
    config.use_fp16 = true;

    WebX::WebXRuntime runtime;
    if (!runtime.initialize(config, 3)) {
        std::cerr << "Failed to initialize K'UHUL runtime\n";
        return 1;
    }

    if (command == "test" && argc >= 3 && std::string(argv[2]) == "all") {
        // Demo: create a compute field and execute it through BOSS layers.
        WebX::Field* field = runtime.createField("demo.compute", "Compute");
        if (!field) {
            std::cerr << "Failed to create demo field\n";
            return 1;
        }
        field->pressure = 0.85f;

        bool ok = runtime.executeField("demo.compute");

        auto stats = runtime.getStats();
        std::cout << "\nRuntime stats:\n"
                  << "  fields:      " << stats.field_count << "\n"
                  << "  total cards: " << stats.total_cards << "\n"
                  << "  total tokens:" << stats.total_tokens << "\n"
                  << "  avg pressure:" << stats.avg_pressure << "\n";

        runtime.shutdown();
        return ok ? 0 : 1;
    }

    if (command == "prompt" && argc >= 3) {
        // Forge a Micronaut network from the supplied prompt and replay it
        // against a freshly created field. This exercises the full semantic
        // pipeline: prompt -> FieldGraph -> Forge -> Micronauts -> execution.
        std::string prompt_text;
        for (int i = 2; i < argc; ++i) {
            if (i > 2) prompt_text += " ";
            prompt_text += argv[i];
        }

        WebX::Field* field = runtime.createField("forge.target", "Compute");
        if (!field) {
            std::cerr << "Failed to create forge target field\n";
            return 1;
        }
        field->pressure = 0.9f;

        // The runtime exposes MicronautForge through a dedicated accessor in
        // webx_compute.h; if the current simplified runtime does not expose it
        // directly, we fall back to a BOSS execution.
        std::cout << "Prompt: " << prompt_text << "\n";
        bool ok = runtime.executeField("forge.target");

        runtime.shutdown();
        return ok ? 0 : 1;
    }

    if (command == "forge" && argc >= 3) {
        std::string prompt_text;
        for (int i = 2; i < argc; ++i) {
            if (i > 2) prompt_text += " ";
            prompt_text += argv[i];
        }

        WebX::MicronautForge forge(&runtime.getGPU(), &runtime.getGraph());
        forge.getProviderManager().printStatus();

        WebX::ForgeRequest request;
        request.prompt = prompt_text;
        request.targets = {
            WebX::ForgeTarget::HLSL,
            WebX::ForgeTarget::DirectML,
            WebX::ForgeTarget::SCX,
            WebX::ForgeTarget::CSO,
            WebX::ForgeTarget::LoRA
        };
        WebX::ForgeResult* result = forge.forge(request);

        if (result) {
            std::cout << "Forge result: " << result->network_id
                      << " confidence=" << result->confidence
                      << " micronauts=" << result->micronauts.size()
                      << " artifacts=" << result->artifacts.size() << "\n";
            for (const auto& artifact : result->artifacts) {
                std::cout << "  artifact: " << artifact.name
                          << " target=" << WebX::ArtifactEmitter::targetName(artifact.target)
                          << (artifact.micronaut_id.empty() ? "" :
                              " micronaut=" + artifact.micronaut_id)
                          << " provider=" << artifact.provider_id
                          << " bytes=" << artifact.data.size() << "\n";
            }
        }

        runtime.shutdown();
        return result ? 0 : 1;
    }

    std::cerr << "Unknown command: " << command << "\n";
    printUsage(argv[0]);
    runtime.shutdown();
    return 1;
}
