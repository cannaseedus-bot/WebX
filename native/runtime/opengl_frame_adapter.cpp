#include "opengl_frame_adapter.h"

#include <windows.h>
#include <GL/gl.h>

#include <string>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>

namespace Kuhul::Runtime {

namespace {

struct Vertex {
    float x;
    float y;
    float z;
};

struct Triangle {
    Vertex a;
    Vertex b;
    Vertex c;
};

Vertex faceNormal(const Triangle& triangle) {
    const float ux = triangle.b.x - triangle.a.x;
    const float uy = triangle.b.y - triangle.a.y;
    const float uz = triangle.b.z - triangle.a.z;
    const float vx = triangle.c.x - triangle.a.x;
    const float vy = triangle.c.y - triangle.a.y;
    const float vz = triangle.c.z - triangle.a.z;
    const float nx = uy * vz - uz * vy;
    const float ny = uz * vx - ux * vz;
    const float nz = ux * vy - uy * vx;
    const float length = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (length <= 0.000001f) return {0.0f, 0.0f, 1.0f};
    return {nx / length, ny / length, nz / length};
}

bool loadObj(const std::string& path, std::vector<Triangle>& triangles,
             std::string& error) {
    std::ifstream file(path);
    if (!file) {
        error = "opengl_obj_unreadable";
        return false;
    }
    std::vector<Vertex> vertices;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream input(line);
        std::string kind;
        input >> kind;
        if (kind == "v") {
            Vertex vertex{};
            if (!(input >> vertex.x >> vertex.y >> vertex.z)) {
                error = "opengl_obj_invalid_vertex";
                return false;
            }
            vertices.push_back(vertex);
        } else if (kind == "f") {
            std::vector<int> indices;
            std::string token;
            while (input >> token) {
                const size_t slash = token.find('/');
                const std::string index = token.substr(0, slash);
                try {
                    indices.push_back(std::stoi(index));
                } catch (const std::exception&) {
                    error = "opengl_obj_invalid_face";
                    return false;
                }
            }
            if (indices.size() < 3) {
                error = "opengl_obj_face_too_small";
                return false;
            }
            for (size_t i = 1; i + 1 < indices.size(); ++i) {
                const int ia = indices[0] - 1;
                const int ib = indices[i] - 1;
                const int ic = indices[i + 1] - 1;
                if (ia < 0 || ib < 0 || ic < 0 ||
                    ia >= static_cast<int>(vertices.size()) ||
                    ib >= static_cast<int>(vertices.size()) ||
                    ic >= static_cast<int>(vertices.size())) {
                    error = "opengl_obj_face_index_out_of_range";
                    return false;
                }
                triangles.push_back({vertices[ia], vertices[ib], vertices[ic]});
            }
        }
    }
    if (triangles.empty()) {
        error = "opengl_obj_contains_no_faces";
        return false;
    }
    return true;
}

LRESULT CALLBACK frameWindowProc(HWND window, UINT message,
                                 WPARAM wParam, LPARAM lParam) {
    if (message == WM_CLOSE) {
        DestroyWindow(window);
        return 0;
    }
    if (message == WM_DESTROY) {
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcA(window, message, wParam, lParam);
}

void drawAtomicBlockLayout(const std::vector<std::string>& blocks) {
    if (blocks.empty()) return;

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    const float panelAlpha = 0.22f;
    for (const auto& block : blocks) {
        float left = -1.45f;
        float right = 1.45f;
        float bottom = -1.35f;
        float top = 1.35f;
        if (block == "HEADER") {
            bottom = 1.05f;
            top = 1.45f;
        } else if (block == "FOOTER") {
            bottom = -1.45f;
            top = -1.15f;
        } else if (block == "MENU") {
            right = -0.95f;
            bottom = -1.05f;
            top = 1.02f;
        } else if (block == "GRID") {
            left = 0.98f;
            bottom = -1.05f;
            top = 1.02f;
        } else if (block == "FEED") {
            left = -0.92f;
            right = 0.92f;
            bottom = -1.05f;
            top = -0.82f;
        } else if (block == "BODY" || block == "GAME") {
            left = -0.92f;
            right = 0.92f;
            bottom = -0.78f;
            top = 1.02f;
        } else {
            continue;
        }

        if (block == "GAME" || block == "BODY") {
            glColor4f(0.06f, 0.18f, 0.28f, panelAlpha);
        } else if (block == "MENU") {
            glColor4f(0.15f, 0.32f, 0.48f, panelAlpha);
        } else if (block == "GRID") {
            glColor4f(0.18f, 0.42f, 0.30f, panelAlpha);
        } else {
            glColor4f(0.32f, 0.24f, 0.12f, panelAlpha);
        }
        glBegin(GL_QUADS);
        glVertex2f(left, bottom);
        glVertex2f(right, bottom);
        glVertex2f(right, top);
        glVertex2f(left, top);
        glEnd();

        glColor4f(0.45f, 0.78f, 0.95f, 0.75f);
        glBegin(GL_LINE_LOOP);
        glVertex2f(left, bottom);
        glVertex2f(right, bottom);
        glVertex2f(right, top);
        glVertex2f(left, top);
        glEnd();
    }
    glDisable(GL_BLEND);
}

} // namespace

bool OpenGLFrameAdapter::renderSmoke(unsigned frames) {
    error_.clear();
    if (frames == 0) frames = 1;

    const char* className = "KuhulAtomicOpenGLFrame";
    HINSTANCE instance = GetModuleHandleA(nullptr);
    WNDCLASSA windowClass{};
    windowClass.hInstance = instance;
    windowClass.lpfnWndProc = frameWindowProc;
    windowClass.lpszClassName = className;
    windowClass.style = CS_OWNDC;
    if (!RegisterClassA(&windowClass) &&
        GetLastError() != ERROR_CLASS_ALREADY_EXISTS) {
        error_ = "opengl_register_window_class_failed";
        return false;
    }

    HWND window = CreateWindowExA(
        0, className, "K'UHUL Atomic DOM", WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, 800, 600,
        nullptr, nullptr, instance, nullptr);
    if (!window) {
        error_ = "opengl_create_window_failed";
        return false;
    }

    HDC dc = GetDC(window);
    PIXELFORMATDESCRIPTOR format{};
    format.nSize = sizeof(format);
    format.nVersion = 1;
    format.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    format.iPixelType = PFD_TYPE_RGBA;
    format.cColorBits = 32;
    format.cDepthBits = 24;
    format.iLayerType = PFD_MAIN_PLANE;
    const int pixelFormat = ChoosePixelFormat(dc, &format);
    if (pixelFormat == 0 || !SetPixelFormat(dc, pixelFormat, &format)) {
        error_ = "opengl_set_pixel_format_failed";
        ReleaseDC(window, dc);
        DestroyWindow(window);
        return false;
    }

    HGLRC context = wglCreateContext(dc);
    if (!context || !wglMakeCurrent(dc, context)) {
        error_ = "opengl_create_context_failed";
        if (context) wglDeleteContext(context);
        ReleaseDC(window, dc);
        DestroyWindow(window);
        return false;
    }

    ShowWindow(window, SW_SHOW);
    for (unsigned frame = 0; frame < frames; ++frame) {
        MSG message{};
        while (PeekMessageA(&message, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&message);
            DispatchMessageA(&message);
        }
        glViewport(0, 0, 800, 600);
        glClearColor(0.03f, 0.08f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        SwapBuffers(dc);
    }

    wglMakeCurrent(nullptr, nullptr);
    wglDeleteContext(context);
    ReleaseDC(window, dc);
    DestroyWindow(window);
    UnregisterClassA(className, instance);
    return true;
}

bool OpenGLFrameAdapter::renderObjSmoke(const std::string& path,
                                        unsigned frames, bool interactive,
                                        const std::vector<std::string>& blocks) {
    std::vector<Triangle> triangles;
    if (!loadObj(path, triangles, error_)) return false;
    if (frames == 0) frames = 1;

    const char* className = "KuhulAtomicOpenGLObjFrame";
    HINSTANCE instance = GetModuleHandleA(nullptr);
    WNDCLASSA windowClass{};
    windowClass.hInstance = instance;
    windowClass.lpfnWndProc = frameWindowProc;
    windowClass.lpszClassName = className;
    windowClass.style = CS_OWNDC;
    if (!RegisterClassA(&windowClass) &&
        GetLastError() != ERROR_CLASS_ALREADY_EXISTS) {
        error_ = "opengl_register_obj_window_class_failed";
        return false;
    }
    HWND window = CreateWindowExA(
        0, className, "K'UHUL Atomic OBJ", WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, 800, 600,
        nullptr, nullptr, instance, nullptr);
    if (!window) {
        error_ = "opengl_create_obj_window_failed";
        return false;
    }
    HDC dc = GetDC(window);
    PIXELFORMATDESCRIPTOR format{};
    format.nSize = sizeof(format);
    format.nVersion = 1;
    format.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    format.iPixelType = PFD_TYPE_RGBA;
    format.cColorBits = 32;
    format.cDepthBits = 24;
    format.iLayerType = PFD_MAIN_PLANE;
    const int pixelFormat = ChoosePixelFormat(dc, &format);
    if (pixelFormat == 0 || !SetPixelFormat(dc, pixelFormat, &format)) {
        error_ = "opengl_set_obj_pixel_format_failed";
        ReleaseDC(window, dc);
        DestroyWindow(window);
        return false;
    }
    HGLRC context = wglCreateContext(dc);
    if (!context || !wglMakeCurrent(dc, context)) {
        error_ = "opengl_create_obj_context_failed";
        if (context) wglDeleteContext(context);
        ReleaseDC(window, dc);
        DestroyWindow(window);
        return false;
    }
    ShowWindow(window, SW_SHOW);
    unsigned frame = 0;
    bool running = true;
    float yaw = 0.0f;
    float pitch = 0.0f;
    float distance = 3.0f;
    while (running && (interactive || frame < frames)) {
        MSG message{};
        while (PeekMessageA(&message, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&message);
            DispatchMessageA(&message);
            if (message.message == WM_QUIT) running = false;
        }
        if (!running) break;
        glViewport(0, 0, 800, 600);
        glClearColor(0.03f, 0.08f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        const float aspect = 800.0f / 600.0f;
        glFrustum(-0.75 * aspect, 0.75 * aspect, -0.75, 0.75,
                  1.0, 100.0);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        if (interactive) {
            if (GetAsyncKeyState(VK_LEFT) & 0x8000) yaw -= 1.5f;
            if (GetAsyncKeyState(VK_RIGHT) & 0x8000) yaw += 1.5f;
            if (GetAsyncKeyState(VK_UP) & 0x8000) pitch -= 1.5f;
            if (GetAsyncKeyState(VK_DOWN) & 0x8000) pitch += 1.5f;
            if (GetAsyncKeyState('W') & 0x8000) distance -= 0.04f;
            if (GetAsyncKeyState('S') & 0x8000) distance += 0.04f;
            if (distance < 1.25f) distance = 1.25f;
            if (distance > 12.0f) distance = 12.0f;
            if (pitch < -80.0f) pitch = -80.0f;
            if (pitch > 80.0f) pitch = 80.0f;
        }
        glTranslatef(0.0f, 0.0f, -distance);
        glRotatef(pitch, 1.0f, 0.0f, 0.0f);
        glRotatef(yaw, 0.0f, 1.0f, 0.0f);
        const GLfloat lightPosition[] = {-2.0f, 3.0f, 4.0f, 1.0f};
        const GLfloat lightDiffuse[] = {0.95f, 0.92f, 0.82f, 1.0f};
        const GLfloat lightAmbient[] = {0.18f, 0.20f, 0.24f, 1.0f};
        glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);
        glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
        glLightfv(GL_LIGHT0, GL_DIFFUSE, lightDiffuse);
        glLightfv(GL_LIGHT0, GL_AMBIENT, lightAmbient);
        glEnable(GL_COLOR_MATERIAL);
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
        glColor3f(0.2f, 0.85f, 0.55f);
        glBegin(GL_TRIANGLES);
        for (const auto& triangle : triangles) {
            const Vertex normal = faceNormal(triangle);
            glNormal3f(normal.x, normal.y, normal.z);
            for (const auto& vertex : {triangle.a, triangle.b, triangle.c})
                glVertex3f(vertex.x, vertex.y, vertex.z);
        }
        glEnd();
        glDisable(GL_COLOR_MATERIAL);
        glDisable(GL_LIGHT0);
        glDisable(GL_LIGHTING);
        drawAtomicBlockLayout(blocks);
        SwapBuffers(dc);
        ++frame;
        if (interactive) Sleep(16);
    }
    wglMakeCurrent(nullptr, nullptr);
    wglDeleteContext(context);
    ReleaseDC(window, dc);
    DestroyWindow(window);
    UnregisterClassA(className, instance);
    return true;
}

} // namespace Kuhul::Runtime
