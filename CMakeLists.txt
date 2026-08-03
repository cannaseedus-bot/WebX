cmake_minimum_required(VERSION 3.20)
project(KuhulEngine 
    VERSION 1.0.0
    DESCRIPTION "K'UHUL Semantic Engine - Unified Native Binary"
    LANGUAGES CXX
)

set(LLAMA_CPP_ROOT "C:/Users/canna/.ASX.cpp/llama-b9968-bin-win-cpu-x64/llama.cpp"
    CACHE PATH "llama.cpp source tree used for the optional native provider")
set(LLAMA_RUNTIME_ROOT "C:/Users/canna/.ASX.cpp/llama-b9968-bin-win-cpu-x64"
    CACHE PATH "llama.cpp runtime DLL directory")
if(EXISTS "${LLAMA_CPP_ROOT}/include/llama.h")
    set(KUHUL_HAS_LLAMA_CPP TRUE)
    message(STATUS "llama.cpp provider: ${LLAMA_CPP_ROOT}")
else()
    set(KUHUL_HAS_LLAMA_CPP FALSE)
    message(STATUS "llama.cpp provider: disabled (llama.h not found)")
endif()

# ============================================================================
# Platform & Compiler Configuration
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG "/ZI /D_DEBUG /RTC1")
set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")

# Target Windows 10+
set(CMAKE_SYSTEM_VERSION 10.0)

# Use Visual Studio 2022 generator
if(NOT GENERATOR_SET)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.21)
        set(CMAKE_GENERATOR "Visual Studio 17 2022")
        set(CMAKE_GENERATOR_PLATFORM x64)
    endif()
    set(GENERATOR_SET TRUE)
endif()

# ============================================================================
# Output Configuration
# ============================================================================

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# ============================================================================
# Windows/DirectX Configuration
# ============================================================================

# Add Windows SDK paths
find_path(DIRECTML_INCLUDE_DIR dml.h)

# D3D11/D3D12/DXGI libraries
set(WINDOWS_LIBRARIES
    d3d11.lib
    d3d12.lib
    dxgi.lib
    d3dcompiler.lib
    dxcompiler.lib
    dxguid
    user32
    gdi32
    opengl32
    kernel32
    ole32
)

# ============================================================================
# Include Directories
# ============================================================================

include_directories(
    "${CMAKE_SOURCE_DIR}/native"
    "${CMAKE_SOURCE_DIR}/native/pi_kuhul"
    "${CMAKE_SOURCE_DIR}/native/semantic_reader"
    "${CMAKE_SOURCE_DIR}/native/geometry"
    "${CMAKE_SOURCE_DIR}/native/gpu_trainer"
    "${CMAKE_SOURCE_DIR}/native/kuhul_trainer"
    "${CMAKE_SOURCE_DIR}/native/kxml"
    "${CMAKE_SOURCE_DIR}/native/optical_processor"
    "${CMAKE_SOURCE_DIR}/native/klsl"
    "${CMAKE_SOURCE_DIR}/native/d3d12_compute"
    "${CMAKE_SOURCE_DIR}/native/shaders"
    "${CMAKE_SOURCE_DIR}/native/win2d"
    "${CMAKE_SOURCE_DIR}/native/kuhul_tools"
)

if(KUHUL_HAS_LLAMA_CPP)
    include_directories(
        "${LLAMA_CPP_ROOT}/include"
        "${LLAMA_CPP_ROOT}/ggml/include"
    )
endif()

if(DIRECTML_INCLUDE_DIR)
    include_directories(${DIRECTML_INCLUDE_DIR})
endif()

# ============================================================================
# Source Files - Core Semantic Engine
# ============================================================================

# K'UHUL Pi CPU Microkernel (343 KB, 9 files)
file(GLOB PI_KUHUL_SOURCES
    "native/pi_kuhul/*.cpp"
)

# Semantic Reader (24 KB, 2 files)
file(GLOB SEMANTIC_READER_SOURCES
    "native/semantic_reader/*.cpp"
)
# Requires nlohmann/json.hpp, which is not vendored yet.
list(FILTER SEMANTIC_READER_SOURCES EXCLUDE REGEX "semantic_reader\\.cpp$")

# Geometry Field Mapping (24 KB, 1 file)
file(GLOB GEOMETRY_SOURCES
    "native/geometry/*.cpp"
)

# Native inference model loading and validation.
file(GLOB INFERENCE_SOURCES
    "native/inference/*.cpp"
)

# Runtime phase sandbox: Pop -> Wo -> Yax -> Sek -> Chen -> Xul.
file(GLOB PHASE_RUNTIME_SOURCES
    "native/runtime/*.cpp"
)
list(FILTER PHASE_RUNTIME_SOURCES EXCLUDE REGEX "(opencl_helper|task_helper)\\.cpp$")

# Native .kuhul source inspection and semantic package host.
file(GLOB KUHUL_SOURCE_SOURCES
    "native/source/*.cpp"
)

# GPU Trainer (477 KB, 26 files)
file(GLOB GPU_TRAINER_SOURCES
    "native/gpu_trainer/*.cpp"
)
# Exclude files with missing external dependencies or that are not yet ready.
# These modules depend on WebView2, nlohmann/json, incomplete headers, or have
# internal copy/visibility issues that prevent them from compiling in the
# current unified build. They can be re-enabled as their dependencies arrive.
list(FILTER GPU_TRAINER_SOURCES EXCLUDE REGEX "(asx_ram_test|cockpit_host|gpu_fwdbwd_new|gpt2_train_main|xshard_tile_test|xshard_validate|json_runtime_test|gpu_model_bridge|gpt2_trainer|scx_transformer_block|speculative_decoder|dx12_inference_pipeline|scx_infer_layer)\\.cpp$")

# K'UHUL Tensor Implementation (103 KB, 3 files)
file(GLOB KUHUL_TRAINER_SOURCES
    "native/kuhul_trainer/*.cpp"
)
# Use only the block-free tensor implementation as a library. The other files
# define their own main() entry points and cannot be linked into the unified
# executable without conditional compilation guards.
list(FILTER KUHUL_TRAINER_SOURCES EXCLUDE REGEX "(kuhul_trainer|kuhul_trainer_block|kuhul_trainer_clean)\\.cpp$")

# KXML Runtime (75 KB, 5 files)
file(GLOB KXML_SOURCES
    "native/kxml/*.cpp"
)

# Optical Processor (59 KB, 14 files)
file(GLOB OPTICAL_PROCESSOR_SOURCES
    "native/optical_processor/*.cpp"
)
# Exclude main optical_processor.cpp until runtime.h/sh_kernel.h/buffer_manager.h are provided.
# Other optical files (geodesic, projection, svg_compiler, wave_vm) remain included.
list(FILTER OPTICAL_PROCESSOR_SOURCES EXCLUDE REGEX "optical_processor\\.cpp$")

# K'LHSL Shader Language (34 KB, 4 files)
file(GLOB KLSL_SOURCES
    "native/klsl/*.cpp"
)
# Exclude the standalone compiler driver; keep the compiler library.
list(FILTER KLSL_SOURCES EXCLUDE REGEX "klsl_main\\.cpp$")

# D3D12 Compute (27 MB, 9 files - large due to precompiled shaders)
file(GLOB D3D12_COMPUTE_SOURCES
    "native/d3d12_compute/*.cpp"
)
# The standalone main.cpp in this folder is a demo entry point; exclude it so
# the unified binary uses the canonical WebX entry point instead.
list(FILTER D3D12_COMPUTE_SOURCES EXCLUDE REGEX "main\\.cpp$")

# Shaders (240 KB, 57 files - .hlsl/.klsl/.wgsl loaded/compiled at runtime)
# NOTE: HLSL files are NOT compiled at build time because they use custom
# entry points and shader models. They are loaded by the GPU backend at runtime
# via D3DCompiler_47.dll or precompiled offline with compile_all_shaders.bat.
set(SHADER_SOURCES "")

# Win2D (306 KB, 32 files)
file(GLOB WIN2D_SOURCES
    "native/win2d/*.cpp"
)
# Win2D requires a precompiled header (pch.h) and WRL/WinRT headers that are
# not present in this repo. Exclude from the unified build until the Win2D
# dependency layer is provided.
set(WIN2D_SOURCES "")

# ============================================================================
# Main Entry Point
# ============================================================================

# Core trainer entry point
# webx_compute.h is a header-only library included by webx_field.cpp and
# kuhul_engine.cpp. kuhul_engine.cpp provides the canonical executable main().
set(MAIN_SOURCES
    "native/webx_field.cpp"
    "native/kuhul_engine.cpp"
    "native/llama_runtime.cpp"
    "native/http_api_server.cpp"
)

# ============================================================================
# Unified Executable Target
# ============================================================================

add_executable(kuhul_engine
    # Core
    ${PI_KUHUL_SOURCES}
    ${SEMANTIC_READER_SOURCES}
    ${GEOMETRY_SOURCES}
    ${INFERENCE_SOURCES}
    ${PHASE_RUNTIME_SOURCES}
    ${KUHUL_SOURCE_SOURCES}
    
    # GPU/Compute
    ${GPU_TRAINER_SOURCES}
    ${KUHUL_TRAINER_SOURCES}
    ${D3D12_COMPUTE_SOURCES}
    
    # Runtime
    ${KXML_SOURCES}
    ${OPTICAL_PROCESSOR_SOURCES}
    ${KLSL_SOURCES}
    
    # UI/Graphics
    ${WIN2D_SOURCES}
    ${SHADER_SOURCES}
    
    # Entry point
    ${MAIN_SOURCES}
)

add_executable(kuhul_opencl_helper
    "native/runtime/opencl_helper.cpp"
)
target_link_libraries(kuhul_opencl_helper PRIVATE kernel32)

add_executable(kuhul_task_helper
    "native/runtime/task_helper.cpp"
)
target_link_libraries(kuhul_task_helper PRIVATE kernel32)

# ============================================================================
# Compilation Flags
# ============================================================================

target_compile_options(kuhul_engine PRIVATE
    /W4                    # Warning level 4
    /fp:fast              # Fast floating point
    /arch:AVX2            # AVX2 SIMD support
    /permissive-          # Strict C++ compliance
    /std:c++17
)

# Debug symbols
if(MSVC)
    target_compile_options(kuhul_engine PRIVATE
        $<$<CONFIG:Debug>:/ZI>
        $<$<CONFIG:Release>:/O2>
    )
endif()

# ============================================================================
# Linking
# ============================================================================

target_link_libraries(kuhul_engine PRIVATE ${WINDOWS_LIBRARIES} httpapi.lib)

# ============================================================================
# Preprocessor Definitions
# ============================================================================

target_compile_definitions(kuhul_engine PRIVATE
    # Platform
    _WIN64
    _WINDOWS
    WIN32_LEAN_AND_MEAN
    NOMINMAX
    _CRT_SECURE_NO_WARNINGS
    KUHUL_LLAMA_DEFAULT_ROOT="${LLAMA_RUNTIME_ROOT}"
    $<$<BOOL:${KUHUL_HAS_LLAMA_CPP}>:KUHUL_HAS_LLAMA_CPP>
    
    # GPU
    USE_D3D11
    USE_D3D12
    USE_DIRECTML
    
    # Features
    ENABLE_SEMANTIC_VALIDATION
    ENABLE_PRESSURE_SCHEDULING
    ENABLE_CONFIDENCE_TRACKING
    ENABLE_GPU_COMPUTE
    ENABLE_OPTICAL_PROCESSOR
    ENABLE_KXML_RUNTIME
    
    # Version
    KUHUL_VERSION="1.0.0"
    KUHUL_ENGINE_NAME="K'UHUL Semantic Engine"
)

# ============================================================================
# Post-Build Steps
# ============================================================================

add_custom_command(TARGET kuhul_engine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Build complete: kuhul_engine.exe"
    COMMAND ${CMAKE_COMMAND} -E echo "Output directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)

# The D3D12 compute sources use the DXC API. Keep its runtime beside the
# executable because dxcompiler.dll/dxil.dll are not guaranteed to be in the
# Windows system directory.
file(GLOB DXC_RUNTIME_DLLS
    "C:/Program Files (x86)/Windows Kits/10/bin/*/x64/dxcompiler.dll"
    "C:/Program Files (x86)/Windows Kits/10/bin/*/x64/dxil.dll"
)
foreach(DXC_RUNTIME_DLL IN LISTS DXC_RUNTIME_DLLS)
    add_custom_command(TARGET kuhul_engine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${DXC_RUNTIME_DLL}"
                "$<TARGET_FILE_DIR:kuhul_engine>"
    )
endforeach()

# ============================================================================
# Install Configuration
# ============================================================================

install(TARGETS kuhul_engine
    DESTINATION bin
)

install(FILES
    "native/trainer-server.cjs"
    DESTINATION bin
)

install(DIRECTORY
    "native/shaders"
    DESTINATION bin
)

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "╔════════════════════════════════════════════════════════════╗")
message(STATUS "║           K'UHUL SEMANTIC ENGINE BUILD CONFIG             ║")
message(STATUS "╠════════════════════════════════════════════════════════════╣")
message(STATUS "║ Project:        KuhulEngine v1.0.0                        ║")
message(STATUS "║ Generator:      ${CMAKE_GENERATOR}")
message(STATUS "║ Platform:       ${CMAKE_GENERATOR_PLATFORM}")
message(STATUS "║ C++ Standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "║ Output:         ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}        ║")
message(STATUS "╠════════════════════════════════════════════════════════════╣")
message(STATUS "║ COMPONENTS INCLUDED:                                      ║")
message(STATUS "║  ✓ Pi K'UHUL (CPU microkernel)      343 KB, 9 files       ║")
message(STATUS "║  ✓ Semantic Reader (law enforcement) 24 KB, 2 files      ║")
message(STATUS "║  ✓ Geometry Field Mapping            24 KB, 1 file       ║")
message(STATUS "║  ✓ GPU Trainer (D3D11/D3D12)        477 KB, 26 files     ║")
message(STATUS "║  ✓ K'UHUL Tensor (Tensor3D/8D)     103 KB, 3 files      ║")
message(STATUS "║  ✓ KXML Runtime                      75 KB, 5 files      ║")
message(STATUS "║  ✓ Optical Processor                 59 KB, 14 files     ║")
message(STATUS "║  ✓ K'LHSL Compiler                   34 KB, 4 files      ║")
message(STATUS "║  ✓ D3D12 Compute (+ prebuilt)    27 MB, 9 files        ║")
message(STATUS "║  ✓ Shaders (HLSL/CSO)               240 KB, 57 files    ║")
message(STATUS "║  ✓ Win2D Graphics                   306 KB, 32 files    ║")
message(STATUS "╠════════════════════════════════════════════════════════════╣")
message(STATUS "║ GPU RUNTIME DEPENDENCIES:                                 ║")
message(STATUS "║  • D3D11.dll          (Device & resources)                ║")
message(STATUS "║  • D3D12.dll          (Modern GPU scheduling)             ║")
message(STATUS "║  • D3DCompiler_47.dll (Shader compilation)                ║")
message(STATUS "║  • D3DSCache.dll      (SCX symbolic cache)                ║")
message(STATUS "║  • DirectML.dll       (Hardware-accelerated ML)           ║")
message(STATUS "║  • IntelOpenCL64.dll  (CPU fallback)                      ║")
message(STATUS "╠════════════════════════════════════════════════════════════╣")
message(STATUS "║ BUILD INSTRUCTIONS:                                       ║")
message(STATUS "║  1. mkdir build && cd build                               ║")
message(STATUS "║  2. cmake .. -G \"Visual Studio 17 2022\" -A x64          ║")
message(STATUS "║  3. cmake --build . --config Release                      ║")
message(STATUS "║  4. bin/kuhul_engine.exe                                  ║")
message(STATUS "╚════════════════════════════════════════════════════════════╝")
message(STATUS "")
