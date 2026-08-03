cmake_minimum_required(VERSION 3.20)
project(d3d12_compute_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(d3d12_compute_demo
  main.cpp
)

set_source_files_properties(shader.hlsl PROPERTIES VS_TOOL_OVERRIDE "None")
set_source_files_properties(triangle.hlsl PROPERTIES VS_TOOL_OVERRIDE "None")

target_compile_definitions(d3d12_compute_demo PRIVATE UNICODE _UNICODE)

target_link_libraries(d3d12_compute_demo
  d3d12
  dxgi
  d3dcompiler
)

add_custom_command(TARGET d3d12_compute_demo POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${CMAKE_CURRENT_SOURCE_DIR}/shader.hlsl
          $<TARGET_FILE_DIR:d3d12_compute_demo>/shader.hlsl
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${CMAKE_CURRENT_SOURCE_DIR}/triangle.hlsl
          $<TARGET_FILE_DIR:d3d12_compute_demo>/triangle.hlsl
)

# ============================================================================
# SCX-MoE DLL for Training/Inference GPU Acceleration
# ============================================================================

add_library(sxme_compute SHARED sxme_compute.cpp)

set_source_files_properties(shader_sxme.hlsl PROPERTIES VS_TOOL_OVERRIDE "None")

target_compile_definitions(sxme_compute PRIVATE UNICODE _UNICODE)

target_link_libraries(sxme_compute
  d3d12
  dxgi
  d3dcompiler
  dxcompiler
)

# Copy shader to output directory for runtime compilation
add_custom_command(TARGET sxme_compute POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${CMAKE_CURRENT_SOURCE_DIR}/shader_sxme.hlsl
          $<TARGET_FILE_DIR:sxme_compute>/shader_sxme.hlsl
)
