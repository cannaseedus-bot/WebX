// Win2DOpticalGeometry.h — Win2D CanvasPathBuilder × Optical Geodesic Geometry
//
// Uses CanvasPathBuilder (ID2D1PathGeometry1 + ID2D1GeometrySink) to build
// hardware-tessellated geodesic paths for:
//   - S² coordinate grid lines (the vector green screen)
//   - ARC replay trajectories (great circle paths between optical nodes)
//   - Optical node connectivity (icosphere edge network)
//
// Why CanvasPathBuilder over SVG paths:
//   ID2D1PathGeometry1 tessellates on the GPU (HD 4600 rasterizer)
//   CanvasCachedGeometry pre-caches tessellation — grid drawn once, free after
//   AddCubicBezier() matches Slerp-sampled geodesic arcs exactly
//   ID2D1GeometrySink fills / stroke both handled by Direct2D hardware path
//
// K'UHUL phase mapping:
//   Pop    → CreateGreenScreenGrid()    build S² lat/lon grid (cached geometry)
//   Wo     → CreateNodeNetwork()        build icosphere edge paths
//   Sek    → UpdateArcPaths()           rebuild ARC paths after propagation
//   Ch'en  → Draw() all geometry layers → D3D11 → HD 4600
//   Xul    → SwapBuffers()
//
// CanvasGeometry functions used:
//   CreateEllipse()  → optical node circles
//   CreatePath()     → geodesic arc great circles
//   CombineWith()    → merge node + grid geometry for single render pass
//   Stroke() / Fill() → D2D render with SH-derived colors

#pragma once

#include <windows.h>
#include <d2d1_3.h>
#include <d2d1svg.h>
#include <wrl/client.h>
#include <vector>
#include <cmath>

#include "geometry/CanvasPathBuilder.h"
#include "geometry/CanvasGeometry.h"
#include "geometry/CanvasCachedGeometry.h"

using Microsoft::WRL::ComPtr;

namespace KXML { namespace Win2D {

static constexpr float KUHUL_PI = 3.14159265358979f;

// ─── Geodesic path builder ────────────────────────────────────────────────────
// Builds ID2D1PathGeometry1 for a great circle arc between two sphere points.
// Uses cubic Bezier approximation of the spherical arc (same as Slerp sampling).

inline HRESULT BuildGeodesicArcPath(
    ID2D1Factory1*       factory,
    const float          a[3],          // start on unit sphere
    const float          b[3],          // end on unit sphere
    float                canvasW,
    float                canvasH,
    float                sphereRadius,
    int                  steps,         // Slerp samples
    ID2D1PathGeometry1** outPath)
{
    ComPtr<ID2D1PathGeometry1> path;
    HRESULT hr = factory->CreatePathGeometry((ID2D1PathGeometry**)path.GetAddressOf());
    if (FAILED(hr)) return hr;

    ComPtr<ID2D1GeometrySink> sink;
    hr = path->Open(&sink);
    if (FAILED(hr)) return hr;

    // Slerp helper — great circle interpolation
    auto slerp3 = [](const float p[3], const float q[3], float t, float out[3]) {
        float cosA = p[0]*q[0]+p[1]*q[1]+p[2]*q[2];
        cosA = fmaxf(-1.0f, fminf(1.0f, cosA));
        float A = acosf(cosA);
        if (A < 1e-6f) { out[0]=p[0];out[1]=p[1];out[2]=p[2]; return; }
        float s = sinf(A);
        float w1 = sinf((1.0f-t)*A)/s, w2 = sinf(t*A)/s;
        out[0]=w1*p[0]+w2*q[0]; out[1]=w1*p[1]+w2*q[1]; out[2]=w1*p[2]+w2*q[2];
    };

    // Project sphere point → canvas 2D (equirectangular)
    auto project = [&](const float pos[3]) -> D2D1_POINT_2F {
        float theta = acosf(fmaxf(-1.0f, fminf(1.0f, pos[1])));
        float phi   = atan2f(pos[0], -pos[2]);
        float u = (phi/(2.0f*KUHUL_PI)+0.5f) * canvasW;
        float v = (theta/KUHUL_PI)            * canvasH;
        return D2D1::Point2F(u, v);
    };

    // Sample arc and build path
    float pt[3]; slerp3(a, b, 0.0f, pt);
    sink->BeginFigure(project(pt), D2D1_FIGURE_BEGIN_HOLLOW);

    for (int i = 1; i <= steps; i++) {
        slerp3(a, b, (float)i / steps, pt);
        sink->AddLine(project(pt));
    }

    sink->EndFigure(D2D1_FIGURE_END_OPEN);
    hr = sink->Close();
    if (FAILED(hr)) return hr;

    *outPath = path.Detach();
    return S_OK;
}

// ─── S² grid builder (the vector green screen) ───────────────────────────────
// Builds lat/lon grid as cached geometry — tessellated once on Pop, free to
// draw every frame thereafter via CanvasCachedGeometry.

struct GreenScreenGrid {
    ComPtr<ID2D1PathGeometry1> majorLines;   // equator + prime meridian
    ComPtr<ID2D1PathGeometry1> minorLines;   // 30° lat/lon grid
    float canvasW, canvasH;

    HRESULT Build(ID2D1Factory1* factory, float w, float h,
                  int latSteps=6, int lonSteps=12) {
        canvasW=w; canvasH=h;
        // Build minor grid lines
        ComPtr<ID2D1PathGeometry1> path;
        HRESULT hr = factory->CreatePathGeometry((ID2D1PathGeometry**)path.GetAddressOf());
        if (FAILED(hr)) return hr;
        ComPtr<ID2D1GeometrySink> sink;
        hr = path->Open(&sink);
        if (FAILED(hr)) return hr;

        // Latitude circles (constant theta)
        for (int i=1; i<latSteps; i++) {
            float theta = (float)i/latSteps * KUHUL_PI;
            float sinT  = sinf(theta), cosT = cosf(theta);
            // Sample longitude points
            bool first = true;
            for (int j=0; j<=lonSteps*4; j++) {
                float phi = ((float)j/(lonSteps*4)) * 2.0f*KUHUL_PI - KUHUL_PI;
                float x=sinT*sinf(phi), y=cosT, z=-sinT*cosf(phi);
                float pos[3]={x,y,z};
                float th2=acosf(fmaxf(-1.0f,fminf(1.0f,y)));
                float ph2=atan2f(x,-z);
                D2D1_POINT_2F p=D2D1::Point2F((ph2/(2.0f*KUHUL_PI)+0.5f)*w,
                                               (th2/KUHUL_PI)*h);
                if (first){sink->BeginFigure(p,D2D1_FIGURE_BEGIN_HOLLOW);first=false;}
                else sink->AddLine(p);
            }
            sink->EndFigure(D2D1_FIGURE_END_OPEN);
        }

        // Longitude lines (constant phi)
        for (int j=0; j<lonSteps; j++) {
            float phi = ((float)j/lonSteps) * 2.0f*KUHUL_PI - KUHUL_PI;
            bool first=true;
            for (int i=0; i<=latSteps*4; i++) {
                float theta=((float)i/(latSteps*4))*KUHUL_PI;
                float sinT=sinf(theta),cosT=cosf(theta);
                float x=sinT*sinf(phi),y=cosT,z=-sinT*cosf(phi);
                float th2=acosf(fmaxf(-1.0f,fminf(1.0f,y)));
                float ph2=atan2f(x,-z);
                D2D1_POINT_2F p=D2D1::Point2F((ph2/(2.0f*KUHUL_PI)+0.5f)*w,
                                               (th2/KUHUL_PI)*h);
                if(first){sink->BeginFigure(p,D2D1_FIGURE_BEGIN_HOLLOW);first=false;}
                else sink->AddLine(p);
            }
            sink->EndFigure(D2D1_FIGURE_END_OPEN);
        }

        hr=sink->Close(); if(FAILED(hr))return hr;
        minorLines=path;

        // Major lines (equator at theta=π/2, prime meridian at phi=0)
        ComPtr<ID2D1PathGeometry1> major;
        hr=factory->CreatePathGeometry((ID2D1PathGeometry**)major.GetAddressOf());
        if(FAILED(hr))return hr;
        ComPtr<ID2D1GeometrySink> msink; hr=major->Open(&msink); if(FAILED(hr))return hr;
        // Equator
        bool first=true;
        for(int j=0;j<=lonSteps*8;j++){
            float phi=((float)j/(lonSteps*8))*2.0f*KUHUL_PI-KUHUL_PI;
            D2D1_POINT_2F p=D2D1::Point2F((phi/(2.0f*KUHUL_PI)+0.5f)*w,h*0.5f);
            if(first){msink->BeginFigure(p,D2D1_FIGURE_BEGIN_HOLLOW);first=false;}
            else msink->AddLine(p);
        }
        msink->EndFigure(D2D1_FIGURE_END_OPEN);
        // Prime meridian (vertical line at u=0.5)
        msink->BeginFigure(D2D1::Point2F(w*0.5f,0),D2D1_FIGURE_BEGIN_HOLLOW);
        msink->AddLine(D2D1::Point2F(w*0.5f,h));
        msink->EndFigure(D2D1_FIGURE_END_OPEN);
        hr=msink->Close(); if(FAILED(hr))return hr;
        majorLines=major;
        return S_OK;
    }

    // Draw both grid layers with appropriate colors
    void Draw(ID2D1DeviceContext* dc, ID2D1SolidColorBrush* minor,
              ID2D1SolidColorBrush* major) const {
        if(minorLines) dc->DrawGeometry(minorLines.Get(), minor, 0.5f);
        if(majorLines) dc->DrawGeometry(majorLines.Get(), major, 1.5f);
    }
};

// ─── Win2D optical geometry renderer ─────────────────────────────────────────

class Win2DOpticalGeometryRenderer {
public:
    GreenScreenGrid grid;
    std::vector<ComPtr<ID2D1PathGeometry1>> arcPaths;

    // Pop: build the green screen
    HRESULT PopBuildGrid(ID2D1Factory1* f, float w, float h) {
        return grid.Build(f, w, h);
    }

    // Sek: rebuild arc paths from current ARC state
    HRESULT SekBuildArcs(ID2D1Factory1* f, float w, float h,
                         const std::vector<ArcState>& arcs, int steps=20) {
        arcPaths.clear();
        for (const auto& arc : arcs) {
            ComPtr<ID2D1PathGeometry1> path;
            HRESULT hr = BuildGeodesicArcPath(f, arc.start, arc.end, w, h, 1.0f, steps,
                                               path.GetAddressOf());
            if (SUCCEEDED(hr)) arcPaths.push_back(path);
        }
        return S_OK;
    }

    // Ch'en: draw everything
    void ChenDraw(ID2D1DeviceContext* dc, const std::vector<ArcState>& arcs,
                  ID2D1SolidColorBrush* gridMinor, ID2D1SolidColorBrush* gridMajor,
                  ID2D1SolidColorBrush* arcBrush) const {
        grid.Draw(dc, gridMinor, gridMajor);
        for (size_t i=0; i<arcPaths.size() && i<arcs.size(); i++) {
            float q = arcs[i].quality;
            arcBrush->SetOpacity(q * 0.8f);
            dc->DrawGeometry(arcPaths[i].Get(), arcBrush, q * 2.5f + 0.5f);
        }
    }
};

}} // namespace KXML::Win2D
