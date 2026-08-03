# Atomic Shell DOM

The Atomic Shell DOM is the controlled presentation plane for the K'UHUL
runtime. It composes independently defined blocks into a visible terminal,
browser, or native frame.

## Block set

```text
FRAME
  HEADER
  MENU
  BODY
  GRID
  FEED
  FOOTER
```

Composable widget blocks include:

```text
BUTTON IMAGE VIDEO TEXT INPUT CARD PANEL GAME
```

Each block owns an independently versioned `atomic.manifest.json`. A manifest
defines the block identity, schema, render state, events, routes, and update
permissions. `FRAME` composes validated block manifests into the active shell.
Composition is data-driven through the FRAME manifest `blocks` array.

Hot-path manifests use local schema routes such as
`atomics://local.dns.route`. Deterministic `hash://` and `cache://` routes are
also valid when they provide faster lookup or immutable identity. External
HTTP/HTTPS schema resolution is not required.

## Authority boundary

The Atomic Shell DOM controls presentation:

- which frame and blocks are visible;
- block layout, state, and rendering;
- user-facing menus, feeds, and status views.

The native runtime controls execution:

- TaskEngine admission and dependency ordering;
- provider selection and capability checks;
- model inference and shard execution;
- security and resource limits.

The DOM must not bypass TaskEngine or invoke native providers directly.

## Style boundary

AtomicDOM is not CSS- or SCSS-dependent. Its manifests should express visual
intent as backend-neutral style tokens:

```json
{
  "style": {
    "theme": "station-dark",
    "accent": "#62e6a7",
    "surface": "#07131b",
    "text": "#e8fff4",
    "density": "compact",
    "radius": 0
  }
}
```

The terminal backend maps these tokens to text layout and ANSI capabilities.
The OpenGL backend maps them to clear colors, materials, and geometry layout.
A browser adapter may export the same tokens to CSS variables; SCSS may be
used by that adapter's build pipeline, but no browser, WebView, CSS parser, or
SCSS compiler is required by the native runtime.

## FRAME lifecycle

The display lifecycle maps to the `FRAME` block:

```java
Display.setDisplayMode(new DisplayMode(800, 600));
Display.create();

while (!Display.isCloseRequested()) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    renderHeader();
    renderMenu();
    renderBody();
    renderGrid();
    renderFeed();
    renderFooter();
    Display.update();
    Display.sync(60);
}

Display.destroy();
```

`Display.create()` allocates the frame, `Display.update()` commits a rendered
frame, and `Display.destroy()` releases it. The same lifecycle may be implemented by a native terminal or OpenGL backend.
Browser/WebView rendering is an optional adapter, not a terminal runtime
requirement.

The native OpenGL FRAME smoke path is available through:

```powershell
.\build-llama\bin\Release\kuhul_engine.exe opengl-frame-smoke 1
```

This creates a bounded WGL context and presents a frame through the installed
OpenGL driver. The generic smoke path does not load scene assets.

OBJ smoke rendering is available through:

```powershell
.\build-llama\bin\Release\kuhul_engine.exe `
  opengl-obj-smoke native\runtime\atomic.triangle.obj 1
```

This parses bounded vertices/faces and presents the mesh through the native
OpenGL FRAME. Use `--interactive` instead of `1` to keep the window open until
closed. glTF/GLB and STL loaders remain future renderer stages.

The GAME manifest can launch the same path without a hardcoded asset:

```powershell
.\build-llama\bin\Release\kuhul_engine.exe `
  opengl-game-smoke native\runtime\atomic.game.manifest.json 1
```

The engine resolves `asset_uri` from the manifest and renders the selected
scene through the OpenGL FRAME.

## Manifest shape

```json
{
  "id": "body",
  "block": "BODY",
  "version": "1.0.0",
  "backend": "terminal",
  "feed_parser": "xcfe://tree-sitter-wasm",
  "blocks": ["HEADER", "TEXT", "BUTTON", "FEED", "FOOTER"],
  "$schema": "atomics://local.dns.route",
  "state": {
    "visible": true,
    "residency": "HOT"
  },
  "routes": {
    "render": "/shell/body/render",
    "update": "/shell/body/update"
  },
  "permissions": {
    "render": true,
    "update": true,
    "execute": false
  }
}
```

Block manifests are updated independently and validated before composition.
The `execute` permission remains false for presentation blocks.

## FEED pipeline

FEED content is parsed by the XCFE Tree-sitter WASM parser:

```text
manifest FEED
  -> XCFE Tree-sitter WASM
  -> syntax tree and validated contract
  -> native kuhul_engine Atomic DOM
  -> widget composition and rendering
```

WASM provides deterministic, sandboxed, incremental parsing only. It does not
own CSS, WebView, DOM composition, provider execution, or model execution.
XML/SVG can feed terminal vector output or an OpenGL FRAME renderer. Native
OpenGL supports 3D scene assets such as glTF/GLB; the terminal renderer remains
the lightweight fallback.

`GAME` is the scene block for a 3D world. Its manifest selects the OpenGL
backend and an asset format; world loading, camera/input state, and simulation
remain execution stages behind the validated presentation contract.

The native manifest validator resolves
`tree-sitter-xcfe/tree-sitter-xcfe.wasm` before admitting a FEED composition.
`KUHUL_XCFE_WASM` may provide an explicit local artifact path; no network
fetch is performed. The artifact must be a local regular file with a valid
WebAssembly v1 module header; invalid or unavailable overrides are rejected.

## Runtime integration

```text
K'UHUL source
  -> AST and bytecode
  -> XJSON state
  -> atomic.manifest.json blocks
  -> FRAME composition
  -> controlled presentation
```

`MicrosoftSDK.ps1` may expose shell commands for selecting and updating
manifests. `kuhul_engine.exe`, TaskEngine, and the selected provider remain
the execution boundary. In llama.cpp mode, llama.cpp/GGML owns token
inference while the shell displays normalized runtime state.

The active validation command is:

```powershell
.\build-llama\bin\Release\kuhul_engine.exe `
  atomic-shell native\runtime\atomic.frame.manifest.json --render
```

The terminal launcher is:

```powershell
.\AtomicDOM.cmd
.\AtomicDOM.cmd --game
.\AtomicChat.cmd
.\AtomicPage.cmd
```

These invoke `kuhul_engine.exe --Atomic.DOM`. The launcher renders the
validated FRAME and a basic Chat/Game/Q terminal menu. `--game` currently
prints the `wwa-demos` handoff; a native game renderer is not yet connected.
`AtomicChat.cmd` selects the manifest-bound chat FRAME at
`native/runtime/atomic.chat.manifest.json` and reports its native task and
history routes. `AtomicChat.cmd --login` selects the login page. The current
pages are deterministic terminal previews; provider execution and keypress
routing remain outside AtomicDOM until the task engine route is connected.
The chat manifest also defines the NPC identity, monotone system prompt,
behavior rules, context window, and reply schema consumed by a provider
adapter.
Within the chat prompt, `1` keeps the chat route active, `2` or `/game`
selects the game handoff, `Q` exits, and `/login` selects the login route.
The game handoff targets `native/runtime/atomic.game.manifest.json`; it can
also be launched directly with `AtomicGame.cmd`.
`AtomicPage.cmd` renders the manifest-bound Station Control page.
