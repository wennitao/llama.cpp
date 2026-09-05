set(TARGET llama-hetero-decode-attn)
find_package(OpenCL REQUIRED)
add_executable(${TARGET} hetero-decode-attn.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE ggml OpenCL::OpenCL ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src ${CMAKE_SOURCE_DIR}/ggml/src/ggml-hexagon/htp)
target_compile_definitions(${TARGET} PRIVATE CL_TARGET_OPENCL_VERSION=300)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Control tool: GPU busy with no memory traffic (see gpu-keepalive.c).
add_executable(llama-gpu-keepalive gpu-keepalive.c)
install(TARGETS llama-gpu-keepalive RUNTIME)
target_link_libraries(llama-gpu-keepalive PRIVATE OpenCL::OpenCL)
target_compile_definitions(llama-gpu-keepalive PRIVATE CL_TARGET_OPENCL_VERSION=300)

# Control tool: GPU DRAM streamer at a duty cycle (see gpu-stream.c).
add_executable(llama-gpu-stream gpu-stream.c)
install(TARGETS llama-gpu-stream RUNTIME)
target_link_libraries(llama-gpu-stream PRIVATE OpenCL::OpenCL)
target_compile_definitions(llama-gpu-stream PRIVATE CL_TARGET_OPENCL_VERSION=300)
