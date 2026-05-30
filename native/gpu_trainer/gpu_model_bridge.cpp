/**
 * gpu_model_bridge.cpp
 * Wires GPUModel → cockpit_host.cpp WebView2 bridge.
 *
 * Paste the BRIDGE SECTION below into cockpit_host.cpp
 * (after the g_ready / g_wv globals, before WndProc).
 *
 * Then in HandleKuhulOpcode(), add:
 *   } else if (starts_with(op, L"CHAT_START"))  { Bridge_ChatStart();        }
 *   } else if (starts_with(op, L"CHAT_SEND"))   { Bridge_ChatSend(op);       }
 *   } else if (starts_with(op, L"CHAT_STOP"))   { Bridge_ChatStop();         }
 *
 * JS side (cockpit HTML, PRIME CHAT PRO):
 *   window.chrome.webview.addEventListener('message', e => {
 *     const msg = JSON.parse(e.data);
 *     if (msg.type === 'token') appendToken(msg.tok, msg.text);
 *     if (msg.type === 'perf')  updatePerf(msg.tps, msg.step);
 *     if (msg.type === 'done')  markDone();
 *   });
 *
 *   // Sending a prompt:
 *   window.chrome.webview.postMessage(JSON.stringify({
 *     type: 'kuhul', code: '⟁ CHAT_SEND Hello world'
 *   }));
 */

// ──────────────────────────────────────────────────────────────────────────────
// BRIDGE SECTION — paste into cockpit_host.cpp
// ──────────────────────────────────────────────────────────────────────────────

#include "gpu_model.h"

#include <windows.h>
#include <wrl.h>
#include <string>
#include <sstream>
#include <iomanip>
#include <memory>

using namespace Microsoft::WRL;

// Forward declarations from cockpit_host.cpp
extern ComPtr<ICoreWebView2> g_wv;
extern bool                  g_ready;

// ── Global model instance (allocate once) ─────────────────────────────────────

static std::unique_ptr<GPUModel> g_model;
static bool                      g_modelReady = false;

// ── Helper: escape a string for JSON embedding ────────────────────────────────

static std::wstring JsonEsc(const std::string& s) {
    std::wstring out;
    out.reserve(s.size() + 4);
    for (unsigned char c : s) {
        if (c == '"')  { out += L"\\\""; }
        else if (c == '\\') { out += L"\\\\"; }
        else if (c == '\n') { out += L"\\n"; }
        else if (c == '\r') { out += L"\\r"; }
        else if (c < 0x20)  { /* skip control chars */ }
        else                { out += wchar_t(c); }
    }
    return out;
}

// ── Post a JSON message to the WebView (must be called from UI thread) ────────

static void PostJSON(const std::wstring& json) {
    if (g_ready && g_wv)
        g_wv->PostWebMessageAsString(json.c_str());
}

// ── Bridge callbacks (called from inference thread → marshalled to UI thread) ──

// In a production build, use PostMessage(g_hwnd, WM_APP, ...) to marshal
// from the inference thread to the UI thread.  For simplicity here we call
// PostJSON directly (D3D11 context is not used in callbacks; g_wv is COM-safe).

static void OnToken(uint32_t tok, const std::string& text) {
    std::wostringstream ss;
    ss << L"{\"type\":\"token\","
       << L"\"tok\":" << tok << L","
       << L"\"text\":\"" << JsonEsc(text) << L"\"}";
    PostJSON(ss.str());
}

static void OnPerf(double tps, uint32_t step) {
    std::wostringstream ss;
    ss << L"{\"type\":\"perf\","
       << std::fixed << std::setprecision(1)
       << L"\"tps\":" << tps << L","
       << L"\"step\":" << step << L"}";
    PostJSON(ss.str());
}

// ── Bridge API (called from HandleKuhulOpcode) ────────────────────────────────

void Bridge_ChatStart() {
    if (g_modelReady) return;
    g_model = std::make_unique<GPUModel>(ModelConfig{
        /* vocab=*/ 256,
        /* dim=  */  64,
        /* n_heads=*/  1,
        /* head_dim=*/ 64,
        /* ffn_dim=*/ 256,
        /* n_layers=*/  1,
        /* max_seq=*/ 512,
        /* temp=   */ 0.8f,
        /* top_k=  */  40
    });
    g_model->init();
    g_model->setTokenCallback(OnToken);
    g_model->setPerfCallback(OnPerf);
    g_modelReady = true;
    PostJSON(L"{\"type\":\"kuhul_ack\",\"op\":\"CHAT_START\"}");
}

void Bridge_ChatStop() {
    if (g_model) g_model->stop();
    PostJSON(L"{\"type\":\"done\"}");
}

// Parse the prompt from the opcode string:  "CHAT_SEND <prompt text here>"
void Bridge_ChatSend(const std::wstring& op) {
    // op = "CHAT_SEND Hello world"
    const wchar_t* prefix = L"CHAT_SEND";
    size_t plen = wcslen(prefix);
    std::wstring ws = (op.size() > plen + 1) ? op.substr(plen + 1) : L"";
    std::string prompt;
    prompt.reserve(ws.size());
    for (wchar_t wc : ws) prompt += (wc < 128 ? char(wc) : '?');

    if (!g_modelReady) Bridge_ChatStart();

    // Emit start marker to UI
    PostJSON(L"{\"type\":\"token\",\"tok\":0,\"text\":\"\"}");

    // Non-blocking — tokens stream via OnToken callback
    g_model->generateAsync(prompt, 256);
}

// ──────────────────────────────────────────────────────────────────────────────
// JS RECEIVER SNIPPET (add to PRIME CHAT PRO renderPrimeChatFull):
// ──────────────────────────────────────────────────────────────────────────────
//
// if(window.chrome&&window.chrome.webview){
//   window.chrome.webview.addEventListener('message',function(e){
//     try{
//       const msg=typeof e.data==='string'?JSON.parse(e.data):e.data;
//       if(msg.type==='token'){
//         // append text to chat output
//         const out=document.getElementById('pchat-out');
//         if(out){ out.textContent+=msg.text; out.scrollTop=out.scrollHeight; }
//       } else if(msg.type==='perf'){
//         const p=document.getElementById('pchat-perf');
//         if(p) p.textContent=msg.tps.toFixed(1)+' tok/s  step '+msg.step;
//       } else if(msg.type==='done'){
//         const btn=document.getElementById('pchat-send');
//         if(btn){ btn.disabled=false; btn.textContent='SEND'; }
//       }
//     }catch(ex){}
//   });
// }
//
// // Send button handler:
// document.getElementById('pchat-send').onclick=()=>{
//   const prompt=document.getElementById('pchat-input').value.trim();
//   if(!prompt)return;
//   document.getElementById('pchat-input').value='';
//   window.KuhulDispatch('\u27C1 CHAT_SEND '+prompt);
// };
