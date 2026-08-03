/**
 * cockpit_host.cpp
 * GPU Debug Cockpit — Win32 + WebView2 (x86) host
 *
 * Serves the cockpit HTML via a virtual HTTPS hostname (no file:// security
 * restrictions). The HTML page handles its own demo animation in JS.
 * C++ only pushes real GPU data when available via PushSlots/EmitOp/etc.
 *
 * Build:  python rebuild_cockpit.py
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <wrl.h>
#include <wrl/event.h>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include "WebView2.h"

using namespace Microsoft::WRL;

// ── Config ────────────────────────────────────────────────────────────────────

static const wchar_t* WV2_RUNTIME =
    L"C:\\Users\\canna\\.gpu_trainer\\webview2_runtime"
    L"\\Microsoft.WebView2.FixedVersionRuntime.147.0.3912.60.x86";

// Virtual host maps to the gpu_trainer folder — serves as https://cockpit.local/
static const wchar_t* COCKPIT_URL =
    L"http://127.0.0.1:7420/ASX_Command_Cockpit_BOOT-IFAL.html";
static const wchar_t* NODE_CMD =
    L"node.exe \"C:\\Users\\canna\\.gpu_trainer\\cockpit_server.js\"";

// ── Globals ───────────────────────────────────────────────────────────────────

static HWND                            g_hwnd  = nullptr;
static ComPtr<ICoreWebView2>           g_wv    = nullptr;
static ComPtr<ICoreWebView2Controller> g_ctrl  = nullptr;
static bool                            g_ready = false;

// ── JS helper — must only be called from UI thread ────────────────────────────

static void JS(const std::wstring& s)
{
    if (g_ready && g_wv)
        g_wv->ExecuteScript(s.c_str(), nullptr);
}

// ── Bridge forward declarations ───────────────────────────────────────────────
void Bridge_ChatStart();
void Bridge_ChatStop();
void Bridge_ChatSend(const std::wstring& op);

// ── KUHUL opcode dispatch ─────────────────────────────────────────────────────
// Called on UI thread from WebMessageReceived handler.
// code is the raw opcode string e.g. "⟁ DBG_STEP [Xul]"

static bool starts_with(const std::wstring& s, const wchar_t* prefix)
{
    size_t n = wcslen(prefix);
    return s.size() >= n && s.substr(0, n) == prefix;
}

static void HandleKuhulOpcode(const std::wstring& code)
{
    // Strip leading glyph + whitespace
    std::wstring op = code;
    auto tri = op.find(L'\u27C1'); // ⟁ U+27C1
    if (tri != std::wstring::npos) op = op.substr(tri + 1);
    while (!op.empty() && op[0] == L' ') op = op.substr(1);

    if (starts_with(op, L"DBG_STEP")) {
        JS(L"window.dbg&&window.dbg.step();");
        g_wv->PostWebMessageAsJson(L"{\"type\":\"kuhul_ack\",\"op\":\"DBG_STEP\"}");
    } else if (starts_with(op, L"DBG_RUN")) {
        JS(L"window.dbg&&window.dbg.run();");
        g_wv->PostWebMessageAsJson(L"{\"type\":\"kuhul_ack\",\"op\":\"DBG_RUN\"}");
    } else if (starts_with(op, L"DBG_PAUSE")) {
        JS(L"window.dbg&&window.dbg.pause();");
        g_wv->PostWebMessageAsJson(L"{\"type\":\"kuhul_ack\",\"op\":\"DBG_PAUSE\"}");
    } else if (starts_with(op, L"DBG_FUSED")) {
        // Toggle fusion flag in demo
        JS(L"(function(){var b=document.querySelector('[data-kuhul-click*=DBG_FUSED]');if(b)b.click();})();");
    } else if (starts_with(op, L"STATUS")) {
        // Push current slot state back
        JS(L"window.chrome&&window.chrome.webview&&window.chrome.webview.postMessage("
           L"JSON.stringify({type:'status',slots:window.ASX_SLOTS||null,mode:window.dbg&&window.dbg.mode}));");
    } else if (starts_with(op, L"CHAT_START")) { Bridge_ChatStart();
    } else if (starts_with(op, L"CHAT_SEND"))  { Bridge_ChatSend(op);
    } else if (starts_with(op, L"CHAT_STOP"))  { Bridge_ChatStop();
    }
    // ATTN_FUSED / LOAD_Q / etc. — real jrun dispatch goes here when pipeline is wired
}

// ── Public bridge API (call from UI thread or via PostMessage) ────────────────
// These are the hooks your C++ GPU pipeline calls when real data is ready.

void PushSlots(const char* A, const char* B, const char* C,
               float gpu_ms, float disk_ms, float thr_hps, float max_err)
{
    auto w=[](const char* s){ return std::wstring(s,s+strlen(s)); };
    std::wostringstream ss;
    ss << L"window.ASX_SLOTS={"
       << L"A:'"<<w(A)<<L"',B:'"<<w(B)<<L"',C:'"<<w(C)<<L"',"
       << L"tiles:{A:'',B:'',C:''},"
       << L"stats:{"
         <<L"gpu_ms:"<<std::fixed<<std::setprecision(1)<<gpu_ms<<L","
         <<L"disk_ms:"<<disk_ms<<L","
         <<L"thr_hps:"<<thr_hps<<L","
         <<L"max_err:"<<std::scientific<<std::setprecision(2)<<max_err
       <<L"}};";
    JS(ss.str());
}

void PushActivity(float v)
{
    std::wostringstream ss;
    ss<<L"window.GPU_ACTIVITY&&window.GPU_ACTIVITY("
      <<std::fixed<<std::setprecision(3)<<v<<L");";
    JS(ss.str());
}

void EmitOp(const char* op, int layer, int head, float err, float ms)
{
    std::wostringstream ss;
    ss<<L"window.EmitToUI&&window.EmitToUI('"
      <<std::wstring(op,op+strlen(op))
      <<L" layer="<<layer<<L" head="<<head
      <<L" err="<<std::scientific<<std::setprecision(2)<<err
      <<L" ms="<<std::fixed<<std::setprecision(1)<<ms<<L"');";
    JS(ss.str());
}

// ── Bridge implementation (GPUModel → WebView2) ───────────────────────────────
#include "gpu_model_bridge.cpp"

// ── WndProc ───────────────────────────────────────────────────────────────────

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
    switch (msg) {
    case WM_SIZE:
        if (g_ctrl) { RECT rc; GetClientRect(hwnd,&rc); g_ctrl->put_Bounds(rc); }
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

// ── WebView2 init ─────────────────────────────────────────────────────────────

static bool InitWebView2()
{
    HRESULT hr = CreateCoreWebView2EnvironmentWithOptions(
        WV2_RUNTIME, L".\\wv2_data", nullptr,
        Callback<ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler>(
            [](HRESULT, ICoreWebView2Environment* env) -> HRESULT {
                if (!env) return E_FAIL;
                return env->CreateCoreWebView2Controller(g_hwnd,
                    Callback<ICoreWebView2CreateCoreWebView2ControllerCompletedHandler>(
                        [](HRESULT, ICoreWebView2Controller* ctrl) -> HRESULT {
                            if (!ctrl) return E_FAIL;
                            g_ctrl = ctrl;
                            ctrl->get_CoreWebView2(&g_wv);

                            // Fit to window
                            RECT rc; GetClientRect(g_hwnd,&rc);
                            ctrl->put_Bounds(rc);

                            // Settings
                            ComPtr<ICoreWebView2Settings> cfg;
                            g_wv->get_Settings(&cfg);
                            if (cfg) {
                                cfg->put_IsScriptEnabled(TRUE);
                                cfg->put_AreDefaultScriptDialogsEnabled(TRUE);
                                cfg->put_IsWebMessageEnabled(TRUE);
                                cfg->put_AreDevToolsEnabled(TRUE);
                            }

                            g_wv->add_NavigationCompleted(
                                Callback<ICoreWebView2NavigationCompletedEventHandler>(
                                    [](ICoreWebView2*, ICoreWebView2NavigationCompletedEventArgs*) -> HRESULT {
                                        g_ready = true;
                                        return S_OK;
                                    }).Get(), nullptr);

                            // KUHUL opcode receiver — JS posts {type:"kuhul",code:"⟁ OP args"}
                            g_wv->add_WebMessageReceived(
                                Callback<ICoreWebView2WebMessageReceivedEventHandler>(
                                    [](ICoreWebView2*, ICoreWebView2WebMessageReceivedEventArgs* args) -> HRESULT {
                                        LPWSTR raw = nullptr;
                                        args->TryGetWebMessageAsString(&raw);
                                        if (!raw) {
                                            // Try JSON string
                                            args->get_WebMessageAsJson(&raw);
                                        }
                                        if (raw) {
                                            std::wstring msg(raw);
                                            CoTaskMemFree(raw);
                                            // Parse: {"type":"kuhul","code":"⟁ OP [Xul]"}
                                            auto kp = msg.find(L"\"kuhul\"");
                                            auto cp = msg.find(L"\"code\"");
                                            if (kp != std::wstring::npos && cp != std::wstring::npos) {
                                                auto q1 = msg.find(L'"', cp + 7);
                                                auto q2 = msg.find(L'"', q1 + 1);
                                                if (q1 != std::wstring::npos && q2 != std::wstring::npos)
                                                    HandleKuhulOpcode(msg.substr(q1 + 1, q2 - q1 - 1));
                                            }
                                        }
                                        return S_OK;
                                    }).Get(), nullptr);

                            // Navigate to local Node.js server — full JS, no restrictions
                            g_wv->Navigate(COCKPIT_URL);
                            return S_OK;
                        }).Get());
            }).Get());
    return SUCCEEDED(hr);
}

// ── WinMain ───────────────────────────────────────────────────────────────────

int WINAPI WinMain(HINSTANCE hInst, HINSTANCE, LPSTR, int)
{
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);

    WNDCLASSEXW wc{}; wc.cbSize=sizeof(wc); wc.lpfnWndProc=WndProc;
    wc.hInstance=hInst; wc.hCursor=LoadCursor(nullptr,IDC_ARROW);
    wc.lpszClassName=L"CockpitHost";
    RegisterClassExW(&wc);

    g_hwnd = CreateWindowExW(0, L"CockpitHost", L"KUHUL GPU DEBUG COCKPIT",
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 1400, 860,
        nullptr, nullptr, hInst, nullptr);
    ShowWindow(g_hwnd, SW_SHOW);

    // Spawn Node.js cockpit server (http://127.0.0.1:7420/)
    {
        STARTUPINFOW si{}; si.cb=sizeof(si);
        si.dwFlags=STARTF_USESHOWWINDOW; si.wShowWindow=SW_HIDE;
        PROCESS_INFORMATION pi{};
        wchar_t cmd[512]; wcscpy_s(cmd, NODE_CMD);
        CreateProcessW(nullptr, cmd, nullptr, nullptr, FALSE,
                       CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
        if (pi.hProcess) { CloseHandle(pi.hProcess); CloseHandle(pi.hThread); }
        Sleep(600); // give node ~600ms to bind port before WebView2 navigates
    }

    if (!InitWebView2()) {
        MessageBoxW(nullptr, L"WebView2 init failed.", L"CockpitHost", MB_ICONERROR);
        return 1;
    }

    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    CoUninitialize();
    return 0;
}
