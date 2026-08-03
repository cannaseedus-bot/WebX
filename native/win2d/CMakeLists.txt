cmake_minimum_required(VERSION 3.20)
project(KXML_Win2D LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Win2D NuGet (adjust to your install path) ───────────────────────────────
set(WIN2D_INCLUDE "${CMAKE_SOURCE_DIR}/../../../Win2D/winrt/inc")
set(WIN2D_LIB     "${CMAKE_SOURCE_DIR}/../../../Win2D/lib")

# ─── HLSL compute shaders ─────────────────────────────────────────────────────
add_custom_command(
    OUTPUT  ${CMAKE_CURRENT_BINARY_DIR}/FibonacciCS.cso
    COMMAND dxc.exe -T cs_6_0 -E CSMain
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/FibonacciCS.hlsl
            -Fo ${CMAKE_CURRENT_BINARY_DIR}/FibonacciCS.cso
    DEPENDS shaders/FibonacciCS.hlsl
    COMMENT "Compile FibonacciCS.hlsl"
)

add_custom_command(
    OUTPUT  ${CMAKE_CURRENT_BINARY_DIR}/StabilizeCS.cso
    COMMAND dxc.exe -T cs_6_0 -E CSMain
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/StabilizeCS.hlsl
            -Fo ${CMAKE_CURRENT_BINARY_DIR}/StabilizeCS.cso
    DEPENDS shaders/StabilizeCS.hlsl
    COMMENT "Compile StabilizeCS.hlsl"
)

# ─── KXML Win2D compute nodes ─────────────────────────────────────────────────
add_executable(kxml_win2d_demo
    FibonacciComputeNode.h
    StableOps.h
    ${CMAKE_CURRENT_BINARY_DIR}/FibonacciCS.cso
    ${CMAKE_CURRENT_BINARY_DIR}/StabilizeCS.cso
)

target_compile_definitions(kxml_win2d_demo PRIVATE
    WIN32_LEAN_AND_MEAN NOMINMAX _UNICODE UNICODE
)

target_include_directories(kxml_win2d_demo PRIVATE
    ${WIN2D_INCLUDE}
    # Windows SDK DirectX headers are auto-resolved by MSVC
)

target_link_libraries(kxml_win2d_demo PRIVATE
    windowsapp d3d11 dxgi
)

# Copy CSOs to output
add_custom_command(TARGET kxml_win2d_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_BINARY_DIR}/FibonacciCS.cso
        ${CMAKE_CURRENT_BINARY_DIR}/StabilizeCS.cso
        $<TARGET_FILE_DIR:kxml_win2d_demo>/
)
