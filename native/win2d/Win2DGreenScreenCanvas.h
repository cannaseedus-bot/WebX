// Win2DGreenScreenCanvas.h — Win2D Canvas × Vector Green Screen × Optical Compute
//
// Merges three systems into one hardware-accelerated Direct2D render pass:
//
//   1. Win2D CanvasSvgDocument (ID2D1SvgDocument)
//      → renders the SVG3D compute graph as Direct2D geometry on HD 4600
//
//   2. Vector Green Screen
//      → S² coordinate grid encoded as SVG path elements
//      → each grid line IS a geodesic (CanvasSvgPathAttribute)
//      → icosphere vertices ARE the cluster positions (CanvasSvgPointsAttribute)
//
//   3. Optical Compute (SH wave lattice)
//      → each optical node = one SVG circle element, radius=energy, fill=SH-phase-color
//      → geodesic ARC trajectories = SVG path elements, stroke=quality hue
//      → SH wave state update = update circle fill via CanvasSvgPaintAttribute
//
// K'UHUL phase lifecycle (matches FibonacciComputeNode.h pattern):
//   Pop   → CreateSvgDocument() + build S² grid SVG skeleton
//   Wo    → bind optical node positions → CanvasSvgPointsAttribute
//   Sek   → SH wave propagation dispatch (opcode_kernels OP_PROPAGATE)
//           + update circle fill colors from new SH state
//   Ch'en → DrawSvgDocument() → D3D11 render → HD 4600 iGPU
//   Xul   → swap buffers, advance tick
//
// Win2D SVG API mapping to optical compute:
//   CanvasSvgPointsAttribute    → icosphere vertex positions (Vector2[N])
//   CanvasSvgPathAttribute      → geodesic ARC paths (Slerp-sampled great circles)
//   CanvasSvgPaintAttribute     → SH phase → hue → fill color per cluster circle
//   CanvasSvgStrokeDashArray    → entropy fog → dash length (high entropy = longer dashes)
//   ID2D1DeviceContext5::DrawSvgDocument() → hardware D3D11 render pass
//
// Why Win2D over raw Direct2D:
//   Win2D wraps ID2D1SvgDocument with WinRT lifetime management
//   CreatePointsAttribute(count, Vector2*) is exactly what we need for icosphere
//   CreatePathAttribute(floats, commands) gives us Bezier geodesic arcs
//   The SVG document IS the compute graph — no separate scene graph needed

#pragma once

#include <windows.h>
#include <d3d11.h>
#include <d2d1_3.h>        // ID2D1DeviceContext5 for DrawSvgDocument
#include <d2d1svg.h>       // ID2D1SvgDocument, ID2D1SvgElement
#include <wrl/client.h>
#include <vector>
#include <string>
#include <cmath>

#include "CanvasSvgDocument.h"
#include "CanvasSvgElement.h"
#include "CanvasSvgPaintAttribute.h"
#include "CanvasSvgPathAttribute.h"
#include "CanvasSvgPointsAttribute.h"
#include "CanvasSvgStrokeDashArrayAttribute.h"

using Microsoft::WRL::ComPtr;

namespace KXML { namespace Win2D {

// ─── Optical node state (mirrors OpticalNode from optical-mesh.js) ───────────

struct OpticalNodeState {
    float  position[3];   // unit sphere (x,y,z)
    float  sh[18];        // SH bands 0..8: (cos,sin) pairs
    float  energy;        // mean SH amplitude
    float  phase;         // π-phase of this cluster
    uint32_t neighborCount;
    uint32_t neighbors[6];
};

// ─── ARC trajectory (mirrors ReplayableArc) ──────────────────────────────────

struct ArcState {
    float  start[3];      // start on S²
    float  end[3];        // end on S²
    float  quality;       // [0,1] — maps to stroke-width and opacity
    float  entropy;       // [0,1] — maps to dash-array length
};

// ─── Win2D Green Screen Canvas ────────────────────────────────────────────────

class Win2DGreenScreenCanvas {
public:
    Win2DGreenScreenCanvas(
        ID3D11Device*          d3dDevice,
        ID3D11DeviceContext*   d3dContext,
        float                  width,
        float                  height,
        float                  sphereRadius = 1.0f);

    ~Win2DGreenScreenCanvas();

    // ── Pop: build the S² coordinate grid SVG skeleton ───────────────────────
    // Creates the green screen: lat/lon grid lines as SVG paths on the sphere
    HRESULT Pop_BuildGreenScreen(int gridStepsLat = 6, int gridStepsLon = 12);

    // ── Wo: bind optical node positions to the SVG document ──────────────────
    // Each node becomes a <circle> element; positions stored as PointsAttribute
    HRESULT Wo_BindOpticalNodes(
        const std::vector<OpticalNodeState>& nodes,
        const std::vector<ArcState>&         arcs = {});

    // ── Sek: update SH state → repaint circle fills ──────────────────────────
    // Call after OP_PROPAGATE; updates fill colors without rebuilding the DOM
    HRESULT Sek_UpdateSHState(const std::vector<OpticalNodeState>& nodes);

    // ── Ch'en: draw the SVG document → D3D11 render target ───────────────────
    // ID2D1DeviceContext5::DrawSvgDocument() → hardware path on HD 4600
    HRESULT Chen_Draw(ID3D11RenderTargetView* rtv);

    // ── Xul: advance tick, release staging resources ──────────────────────────
    void Xul_AdvanceTick();

    // Full cycle convenience (Pop+Wo already called; just Sek+Ch'en+Xul)
    HRESULT UpdateAndDraw(
        const std::vector<OpticalNodeState>& nodes,
        const std::vector<ArcState>&         arcs,
        ID3D11RenderTargetView*              rtv);

    uint32_t Tick() const { return m_tick; }
    float    GlobalCoherence() const { return m_globalCoherence; }

private:
    // ── Helpers ───────────────────────────────────────────────────────────────

    // Project 3D sphere point to 2D canvas (equirectangular)
    D2D1_POINT_2F ProjectToCanvas(const float pos[3]) const;

    // Slerp two sphere points, return projected 2D points for arc sampling
    std::vector<D2D1_POINT_2F> SampleGeodesicArc(
        const float a[3], const float b[3], int steps = 20) const;

    // SH phase → D2D1 color (matches PhaseToColor in optical_sphere.hlsl)
    D2D1_COLOR_F SHPhaseToColor(float phase, float energy) const;

    // Entropy [0,1] → dash array (high entropy = longer dashes = foggy)
    std::vector<float> EntropyToDashArray(float entropy) const;

    // Build one SVG path string from sampled 2D points
    std::wstring BuildSvgPathData(const std::vector<D2D1_POINT_2F>& pts) const;

    // ── D3D11 / Direct2D resources ────────────────────────────────────────────
    ComPtr<ID3D11Device>         m_d3dDevice;
    ComPtr<ID3D11DeviceContext>  m_d3dContext;
    ComPtr<ID2D1Factory6>        m_d2dFactory;
    ComPtr<ID2D1Device5>         m_d2dDevice;
    ComPtr<ID2D1DeviceContext5>  m_d2dContext;   // supports DrawSvgDocument
    ComPtr<ID2D1SvgDocument>     m_svgDocument;  // the green screen + compute graph

    // Root SVG elements (cached for fast per-tick updates)
    ComPtr<ID2D1SvgElement>      m_svgRoot;
    ComPtr<ID2D1SvgElement>      m_greenScreenGroup;  // S² grid lines
    ComPtr<ID2D1SvgElement>      m_nodesGroup;         // optical cluster circles
    ComPtr<ID2D1SvgElement>      m_arcsGroup;          // ARC trajectory paths

    // Per-node circle elements (one per optical node, persistent)
    std::vector<ComPtr<ID2D1SvgElement>> m_nodeCircles;

    float    m_width, m_height, m_sphereRadius;
    uint32_t m_tick = 0;
    float    m_globalCoherence = 0.0f;
    float    m_piPhase = 0.0f;
};

// ─── Inline implementations ───────────────────────────────────────────────────

inline D2D1_POINT_2F Win2DGreenScreenCanvas::ProjectToCanvas(const float pos[3]) const {
    // Equirectangular projection: sphere (x,y,z) → canvas (u,v)
    float x = pos[0], y = pos[1], z = pos[2];
    float theta = acosf(fmaxf(-1.0f, fminf(1.0f, y)));  // [0,π]
    float phi   = atan2f(x, -z);                          // [-π,π]
    float u = (phi / (2.0f * 3.14159265f) + 0.5f);        // [0,1]
    float v = theta / 3.14159265f;                         // [0,1]
    return D2D1::Point2F(u * m_width, v * m_height);
}

inline D2D1_COLOR_F Win2DGreenScreenCanvas::SHPhaseToColor(float phase, float energy) const {
    // Same hue mapping as PhaseToColor in optical_sphere.hlsl
    float hue = fmodf(phase / (2.0f * 3.14159265f), 1.0f);
    float h6  = hue * 6.0f;
    float c   = fminf(energy * 1.5f, 1.0f);
    float x   = c * (1.0f - fabsf(fmodf(h6, 2.0f) - 1.0f));
    float r=0,g=0,b=0;
    if      (h6<1){r=c;g=x;}  else if(h6<2){r=x;g=c;}
    else if (h6<3){g=c;b=x;}  else if(h6<4){g=x;b=c;}
    else if (h6<5){r=x;b=c;}  else         {r=c;b=x;}
    return D2D1::ColorF(r, g, b, 0.8f + energy * 0.2f);
}

inline std::vector<float> Win2DGreenScreenCanvas::EntropyToDashArray(float entropy) const {
    // High entropy (fog) → long dashes; low entropy (clear) → solid line
    if (entropy < 0.1f) return {};   // solid
    float dash = 4.0f + entropy * 20.0f;
    float gap  = 2.0f + entropy * 8.0f;
    return { dash, gap };
}

}} // namespace KXML::Win2D
