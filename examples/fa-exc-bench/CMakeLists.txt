set(TARGET llama-fa-exc-bench)
find_package(OpenCL REQUIRED)

# the bench compiles the production fold kernels: their OpenCL source is the raw string in ggml-hexagon.cpp
set(HFOLD_CPP ${CMAKE_SOURCE_DIR}/ggml/src/ggml-hexagon/ggml-hexagon.cpp)
set(HFOLD_INC ${CMAKE_CURRENT_BINARY_DIR}/hfold-cl-src.inc)
add_custom_command(
    OUTPUT  ${HFOLD_INC}
    COMMAND ${CMAKE_COMMAND} -DIN=${HFOLD_CPP} -DOUT=${HFOLD_INC} -P ${CMAKE_CURRENT_SOURCE_DIR}/extract-cl.cmake
    DEPENDS ${HFOLD_CPP} ${CMAKE_CURRENT_SOURCE_DIR}/extract-cl.cmake
    COMMENT "Extracting the fold OpenCL kernels from ggml-hexagon.cpp")

add_executable(${TARGET} fa-exc-bench.cpp ${HFOLD_INC})
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE ggml OpenCL::OpenCL ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_SOURCE_DIR}/ggml/src/ggml-hexagon/htp)
target_compile_definitions(${TARGET} PRIVATE CL_TARGET_OPENCL_VERSION=300)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
