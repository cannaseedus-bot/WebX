#include "KuhulAppCreator.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <vector>

namespace Kuhul::Runtime {

KuhulAppCreator::KuhulAppCreator(WwaRuntimePaths paths)
    : paths_(std::move(paths)) {}

WwaRuntimeStatus KuhulAppCreator::inspect() const {
    WwaRuntimeStatus status;
    status.extension_present =
        std::filesystem::is_regular_file(paths_.wwaExt);
    status.api_present =
        std::filesystem::is_regular_file(paths_.wwaApi);
    status.host_present =
        std::filesystem::is_regular_file(paths_.host);
    if (!status.ready()) {
        status.error = "wwa_runtime_component_missing";
    }
    return status;
}

const std::vector<WwaTemplate>& KuhulAppCreator::templates() {
    static const std::vector<WwaTemplate> available = {
        {"basic", "Minimal K'UHUL app shell"},
        {"dashboard", "Runtime metrics and provider dashboard"},
        {"editor", "Code editor and execution console shell"}
    };
    return available;
}

const std::vector<WwaDesignKit>& KuhulAppCreator::designKits() {
    static const std::vector<WwaDesignKit> available = {
        {"default", "Neutral WebX shell",
         ":root{--accent:#2563eb;--panel:#fff;--surface:#f3f4f6;--text:#202124}"
         "body{background:var(--surface);color:var(--text)}.panel{background:var(--panel);"
         "border:1px solid #d1d5db;border-radius:12px;padding:1rem}"},
        {"midnight", "Dark operator console",
         ":root{--accent:#16f2aa;--panel:#0f172a;--surface:#020617;--text:#e5e7eb}"
         "body{background:radial-gradient(circle at top,#0b1220,#020617 60%,#000);"
         "color:var(--text)}.panel{background:rgba(15,23,42,.9);border:1px solid #334155;"
         "border-radius:12px;padding:1rem;box-shadow:0 0 24px rgba(22,242,170,.08)}"
         ".accent{color:var(--accent)}"},
        {"high-contrast", "High contrast accessibility shell",
         ":root{--accent:#ff0;--panel:#000;--surface:#000;--text:#fff}"
         "body{background:#000;color:#fff}.panel{background:#000;border:2px solid #fff;"
         "border-radius:0;padding:1rem}a,button{color:#ff0}"},
        {"hologram", "ASX holographic 12-column operator grid",
         ":root{--accent:#16f2aa;--panel:rgba(15,23,42,.95);--surface:#020617;--text:#e5f2ff}"
         "body{background:radial-gradient(circle at top,#0b1220,#020617 55%,#000);color:var(--text)}"
         ".panel{background:var(--panel);border:1px solid rgba(148,163,184,.45);border-radius:12px;"
         "padding:1rem;box-shadow:0 0 24px rgba(22,242,170,.12)}.accent{color:var(--accent)}"},
        {"liquid", "Liquid runtime flow and micronaut panels",
         ":root{--accent:#00ffff;--flow:#16f2aa;--panel:#001d3d;--surface:#000814;--text:#e8fff6}"
         "body{background:radial-gradient(circle at 20% 80%,rgba(22,242,170,.1),transparent 50%),"
         "linear-gradient(135deg,#000814,#001d3d);color:var(--text)}.panel{background:rgba(0,29,61,.8);"
         "border:1px solid rgba(22,242,170,.3);border-radius:12px;padding:1rem;backdrop-filter:blur(10px)}"
         ".accent{color:var(--flow)}"},
        {"quantum", "Atomic compression grid with neon execution accents",
         ":root{--accent:#00ff88;--pink:#ff2a6d;--panel:#1a1a2e;--surface:#050510;--text:#e8f4ff}"
         "body{background:radial-gradient(circle at 10% 20%,rgba(0,255,136,.08),transparent 25%),"
         "var(--surface);color:var(--text);font-family:Consolas,monospace}.panel{background:rgba(26,26,46,.9);"
         "border:1px solid var(--accent);border-radius:8px;padding:1rem;box-shadow:0 0 20px rgba(0,255,136,.15)}"
         ".accent{color:var(--accent)}"},
        {"prime-cockpit", "Command cockpit with scanlines, neon frame, and action dock",
         ":root{--accent:#00ffff;--signal:#ffff00;--surface:#000;--panel:rgba(0,0,0,.86);--text:#0ff}"
         "body{background:#000;color:var(--text);font-family:Orbitron,Consolas,monospace}"
         ".panel{background:var(--panel);border:1px solid rgba(0,255,255,.45);border-radius:12px;"
         "padding:1rem;box-shadow:0 0 24px rgba(0,255,255,.2)}.accent{color:var(--signal)}"
         "button{border:2px solid var(--accent);background:rgba(0,255,255,.1);color:var(--accent);"
         "padding:.7rem 1rem;text-transform:uppercase;letter-spacing:.08em}"},
        {"atomic", "Attribute-driven atomic layout, surface, and typography tokens",
         ":root{--accent:#16f2aa;--surface:#020617;--panel:#020817;--text:#e8fff6;--muted:#9ca3af}"
         "body{margin:0;background:radial-gradient(circle at top,#020b18,#020617 55%,#020214);"
         "color:var(--text);font-family:system-ui,sans-serif}.panel{background:linear-gradient(145deg,"
         "var(--panel),#020b18);border:1px solid #1f2937;border-radius:12px;padding:1rem}"
         ".accent{color:var(--accent)}"},
        {"blocks", "Multi-theme glass blocks with HUD, inventory, dialog, and motion",
         ":root{--accent:#4a6cf7;--surface:#0f3460;--panel:rgba(255,255,255,.1);--text:#fff}"
         "body{background:linear-gradient(135deg,#1a1a2e,#16213e 50%,var(--surface));color:var(--text)}"
         ".panel{background:var(--panel);border:1px solid rgba(255,255,255,.2);border-radius:16px;"
         "padding:1rem;backdrop-filter:blur(10px);box-shadow:0 8px 32px rgba(0,0,0,.2)}"
         ".accent{color:var(--accent)}"},
        {"scx", "SCX inspection surface for reversible compression and binary metadata",
         ":root{--accent:#ff6b6b;--surface:#09090b;--panel:#18181b;--text:#f4f4f5;--muted:#a1a1aa}"
         "body{background:linear-gradient(135deg,#09090b,#18181b);color:var(--text);font-family:Consolas,monospace}"
         ".panel{background:var(--panel);border:1px solid #3f3f46;border-radius:8px;padding:1rem}"
         ".accent{color:var(--accent)}"}
    };
    return available;
}

const std::vector<WwaDemo>& KuhulAppCreator::demos() {
    static const std::vector<WwaDemo> available = {
        {"runtime-hologram", "12-column kernel, compression, content, and manifest workspace"},
        {"liquid-runtime", "Micronaut flow cards with runtime, data, execution, and design lanes"},
        {"atomic-compression", "Glyphic compression controls, quantum state, and code streams"},
        {"blocks-hud", "ASX Blocks HUD, shop, inventory, modal, motion, and theme dock"},
        {"node-graph", "FieldGraph node pressure, provider status, and phase timeline"},
        {"prime-cockpit", "Neon command cockpit with tabs, terminal, overlays, and action dock"},
        {"three-file-runtime", "Manifest, kernel, and UI architecture visualizer"},
        {"scx-inspector", "SCX container metadata, compression ratio, and reversible payload viewer"},
        {"atomic-blocks", "Attribute-driven ASX blocks with safe conditional runtime examples"}
    };
    return available;
}

const std::vector<WwaComponent>& KuhulAppCreator::components() {
    static const std::vector<WwaComponent> available = {
        {"app-shell", "Responsive application frame with header and content region", "layout"},
        {"panel", "Theme-aware bordered surface for grouped content", "layout"},
        {"metric-card", "Value, label, delta, and status indicator", "dashboard"},
        {"status-pill", "Compact ready, blocked, or active state marker", "dashboard"},
        {"node-card", "Node capability, pressure, residency, and provider display", "runtime"},
        {"phase-timeline", "Pop through Xul execution phase rail", "runtime"},
        {"code-stream", "Monospace semantic or provider output surface", "developer"},
        {"glyph-button", "Interactive KUHUL glyph action control", "controls"},
        {"hud-bar", "Health, capacity, and runtime resource strip", "game"},
        {"inventory-grid", "Slot-based asset and artifact collection", "game"},
        {"modal", "Accessible confirmation and detail dialog", "controls"},
        {"theme-dock", "Design-kit switcher with persistent visual mode", "controls"},
        {"command-dock", "Primary command actions with active, disabled, and engage states", "controls"},
        {"architecture-triad", "Manifest, kernel, and UI responsibility cards", "runtime"},
        {"fold-rail", "Fourteen-fold or six-phase execution rail with current phase", "runtime"},
        {"scx-inspector", "Header, payload, checksum, and reversible compression details", "developer"},
        {"compression-meter", "Original size, compressed size, ratio, and integrity state", "dashboard"},
        {"micronaut-card", "Personality, capability, provider, and status card", "runtime"},
        {"opcode-button", "Glyph-labelled action button bound to a safe runtime command", "controls"},
        {"atomic-layout", "Attribute-driven row, column, gap, padding, and surface primitives", "layout"}
    };
    return available;
}

bool KuhulAppCreator::create(const std::filesystem::path& parent,
                             const std::string& name,
                             std::filesystem::path& appRoot,
                             std::string& error,
                             const std::string& templateId,
                             const std::string& designKitId,
                             const std::string& serverHost,
                             uint16_t serverPort,
                             const std::string& modelProvider,
                             const std::string& deploymentEnv,
                             const std::string& endpointEnv,
                             const std::string& apiKeyEnv) const {
    if (name.empty() || name == "." || name == ".." ||
        !std::all_of(name.begin(), name.end(), [](unsigned char ch) {
            return std::isalnum(ch) || ch == '-' || ch == '_';
        })) {
        error = "invalid_app_name";
        return false;
    }
    const auto selected = std::find_if(
        templates().begin(), templates().end(),
        [&templateId](const WwaTemplate& item) { return item.id == templateId; });
    if (selected == templates().end()) {
        error = "unknown_app_template:" + templateId;
        return false;
    }
    const auto kit = std::find_if(
        designKits().begin(), designKits().end(),
        [&designKitId](const WwaDesignKit& item) { return item.id == designKitId; });
    if (kit == designKits().end()) {
        error = "unknown_design_kit:" + designKitId;
        return false;
    }
    if (serverHost.empty() ||
        serverHost.find_first_of(" \t\r\n\"'<>/\\") != std::string::npos ||
        serverPort == 0) {
        error = "invalid_app_server";
        return false;
    }
    const bool hasModelBinding = !modelProvider.empty() ||
                                 !deploymentEnv.empty() ||
                                 !endpointEnv.empty() ||
                                 !apiKeyEnv.empty();
    const bool supportedModelProvider =
        modelProvider == "azure_openai" || modelProvider == "openai";
    const auto validEnvName = [](const std::string& value) {
        return !value.empty() &&
               std::all_of(value.begin(), value.end(), [](unsigned char ch) {
                   return std::isalnum(ch) || ch == '_';
               });
    };
    if (hasModelBinding &&
        (!supportedModelProvider || !validEnvName(deploymentEnv) ||
         !validEnvName(endpointEnv) || !validEnvName(apiKeyEnv))) {
        error = "invalid_model_provider_binding";
        return false;
    }
    if (!std::filesystem::is_directory(parent)) {
        error = "app_parent_missing";
        return false;
    }

    appRoot = parent / name;
    if (std::filesystem::exists(appRoot)) {
        error = "app_already_exists";
        return false;
    }
    std::error_code filesystemError;
    if (!std::filesystem::create_directories(appRoot, filesystemError)) {
        error = "app_directory_create_failed:" + filesystemError.message();
        return false;
    }

    std::ofstream index(appRoot / "index.html");
    std::ofstream manifest(appRoot / "manifest.json");
    std::ofstream runtime(appRoot / "app.kuhul");
    std::ofstream componentsFile(appRoot / "components.json");
    std::ofstream demo(appRoot / "demo.html");
    if (!index || !manifest || !runtime || !componentsFile || !demo) {
        error = "app_scaffold_write_failed";
        std::filesystem::remove_all(appRoot, filesystemError);
        return false;
    }

    index << "<!doctype html>\n<html><head><meta charset=\"utf-8\">"
             "<title>" << name << "</title><style>" << kit->css
             << "body{font-family:Segoe UI,sans-serif;margin:2rem}"
             "pre{background:#111;color:#eee;padding:1rem}</style></head><body>"
             "<main id=\"app\"></main><script src=\"app.js\"></script>"
             "</body></html>\n";
    manifest << "{\n"
                "  \"name\": \"" << name << "\",\n"
                "  \"start_url\": \"index.html\",\n"
                "  \"runtime\": \"kuhul\",\n"
                "  \"xjson\": \"1.0\",\n"
                "  \"schema\": \"asx.app\",\n"
                "  \"design_kit\": \"" << designKitId << "\",\n"
                "  \"server\": {\"host\": \"" << serverHost
                << "\", \"port\": " << serverPort << "},\n"
                ;
    if (hasModelBinding) {
        manifest << "  \"model_provider\": {\n"
                    "    \"provider\": \"" << modelProvider << "\",\n"
                    "    \"deployment_env\": \"" << deploymentEnv << "\",\n"
                    "    \"endpoint_env\": \"" << endpointEnv << "\",\n"
                    "    \"api_key_env\": \"" << apiKeyEnv << "\"\n"
                    "  },\n";
    }
    manifest << 
                "  \"nodes\": [\n"
                "    {\"type\":\"tape\",\"id\":\"app.main\",\"entry\":\"app.kuhul\"},\n"
                "    {\"type\":\"route\",\"path\":\"/\",\"component\":\"app.shell\"},\n"
                "    {\"type\":\"style\",\"id\":\"design-kit."
             << designKitId << "\"},\n"
                "    {\"type\":\"component\",\"id\":\"app.components\","
                "\"src\":\"components.json\"},\n"
                "    {\"type\":\"element\",\"tag\":\"iframe\","
                "\"src\":\"demo.html\"}\n"
                "  ]\n"
                "}\n";

    componentsFile << "{\n  \"components\": [\n";
    for (size_t i = 0; i < components().size(); ++i) {
        const auto& component = components()[i];
        componentsFile << "    {\"id\":\"" << component.id
                       << "\",\"category\":\"" << component.category
                       << "\",\"description\":\"" << component.description << "\"}"
                       << (i + 1 == components().size() ? "\n" : ",\n");
    }
    componentsFile << "  ]\n}\n";

    demo << "<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" "
            "content=\"width=device-width,initial-scale=1\"><title>" << name
         << " Demo</title><style>" << kit->css
         << "body{font-family:Segoe UI,sans-serif;margin:0;padding:2rem}"
            ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem}"
            ".metric{font-size:1.8rem}.bar{height:8px;background:#334155;border-radius:99px;overflow:hidden}"
            ".bar i{display:block;height:100%;width:68%;background:var(--accent,#16f2aa)}"
            "pre{white-space:pre-wrap;background:#050505;color:#d1fae5;padding:1rem;border-radius:8px}"
            "</style></head><body><h1 class=\"accent\">" << name
         << "</h1><p>WebX component and design-kit demo</p><section class=\"grid\">"
            "<article class=\"panel\"><strong>Runtime status</strong><div class=\"metric\">READY</div>"
            "<span class=\"accent\">status-pill</span></article>"
            "<article class=\"panel\"><strong>Node pressure</strong><div class=\"metric\">0.68</div>"
            "<div class=\"bar\"><i></i></div><small>node-card / provider: cpu</small></article>"
            "<article class=\"panel\"><strong>Phase timeline</strong><p>Pop → Wo → Yax → Sek → Chen → Xul</p>"
            "<span class=\"accent\">phase-timeline</span></article></section>"
            "<section class=\"panel\" style=\"margin-top:1rem\"><strong>code-stream</strong>"
            "<pre>FieldGraph → Working Set → Yax → Sek → Provider</pre></section>"
            "</body></html>\n";
    runtime << "App " << name << " {\n"
               "    runtime: \"kuhul.runtime\"\n"
               "    entry: \"index.html\"\n"
               "    design_kit: \"" << designKitId << "\"\n"
               "    server_host: \"" << serverHost << "\"\n"
               "    server_port: " << serverPort << "\n"
               "}\n";

    std::ofstream script(appRoot / "app.js");
    if (!script) {
        error = "app_script_write_failed";
        std::filesystem::remove_all(appRoot, filesystemError);
        return false;
    }
    if (templateId == "dashboard") {
        script << "document.getElementById('app').innerHTML = "
                  "'<h1>" << name << "</h1><p>WebX runtime dashboard</p>"
                  "<pre id=\"status\">Loading...</pre>';\n"
                  "fetch('http://" << serverHost << ":" << serverPort
                  << "/health').then(r=>r.json())"
                  ".then(v=>document.getElementById('status').textContent="
                  "JSON.stringify(v,null,2));\n";
    } else if (templateId == "editor") {
        script << "document.getElementById('app').innerHTML = "
                  "'<h1>" << name << "</h1><textarea id=\"code\" rows=\"12\" "
                  "cols=\"80\">// K\\'UHUL source</textarea><pre id=\"out\"></pre>';\n";
    } else {
        script << "document.getElementById('app').textContent = "
                  "'K\\'UHUL app ready';\n";
    }
    return true;
}

bool KuhulAppCreator::launch(const std::filesystem::path& appRoot,
                             std::string& error) const {
    const auto status = inspect();
    if (!status.ready()) {
        error = status.error;
        return false;
    }
    if (!std::filesystem::is_directory(appRoot)) {
        error = "app_root_missing";
        return false;
    }

    std::wstring command = L"\"" + paths_.host.wstring() + L"\" \"" +
                           std::filesystem::absolute(appRoot).wstring() + L"\"";
    std::vector<wchar_t> commandLine(command.begin(), command.end());
    commandLine.push_back(L'\0');

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(nullptr, commandLine.data(), nullptr, nullptr, FALSE,
                        0, nullptr, nullptr, &startup, &process)) {
        error = "wwa_host_launch_failed:" +
                std::to_string(static_cast<unsigned long>(GetLastError()));
        return false;
    }
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    return true;
}

} // namespace Kuhul::Runtime
