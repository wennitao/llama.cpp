cmake_minimum_required(VERSION 3.22.2)
project(ggml-htp C CXX ASM)

include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)

include_directories(
    ${HEXAGON_SDK_ROOT}/incs
    ${HEXAGON_SDK_ROOT}/incs/stddef
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR})

set(HTP_LIB ggml-htp-${DSP_VERSION})

add_library(${HTP_LIB} SHARED
    main.c
    htp_iface_skel.c
    work-queue.c
    dma-queue.c
    hmx-queue.c
    htp-tensor.c
    matmul-ops.c
    flash-attn-ops.c
    gated-delta-net-ops.c
    binary-ops.c
    unary-ops.c
    sum-rows-ops.c
    softmax-ops.c
    xattn-score-ops.c
    act-ops.c
    rope-ops.c
    set-rows-ops.c
    get-rows-ops.c
    cpy-ops.c
    repeat-ops.c
    ssm-conv.c
    cumsum-ops.c
    fill-ops.c
    concat-ops.c
    diag-ops.c
    solve-tri-ops.c
    pad-ops.c
    argsort-ops.c
    im2col-ops.c
)

target_compile_definitions(${HTP_LIB} PRIVATE
    $<IF:$<BOOL:${HEXAGON_HTP_DEBUG}>,HTP_DEBUG=1,NDEBUG=1>
    $<IF:$<BOOL:${HEXAGON_HTP_DEBUG}>,FARF_HIGH=1,>)

# Rendezvous-spin tuning knobs. Left unset the headers pick their own defaults
# (hex-utils.h, work-queue.h); set them to sweep the pause/duty tradeoff without
# editing sources. HEX_PAUSE_SHORT_IMM=255 is the bit-exact no-op control for the
# pause immediate, and WORK_QUEUE_FAST_SPIN=0 is the control for the fast window.
if (HEX_PAUSE_SHORT_IMM)
    target_compile_definitions(${HTP_LIB} PRIVATE HEX_PAUSE_SHORT_IMM=${HEX_PAUSE_SHORT_IMM})
endif()
if (DEFINED WORK_QUEUE_FAST_SPIN AND NOT WORK_QUEUE_FAST_SPIN STREQUAL "")
    target_compile_definitions(${HTP_LIB} PRIVATE WORK_QUEUE_FAST_SPIN=${WORK_QUEUE_FAST_SPIN})
endif()
if (DEFINED HMX_QUEUE_COND_WAKE AND NOT HMX_QUEUE_COND_WAKE STREQUAL "")
    target_compile_definitions(${HTP_LIB} PRIVATE HMX_QUEUE_COND_WAKE=${HMX_QUEUE_COND_WAKE})
endif()


build_idl(htp_iface.idl ${HTP_LIB})

set_target_properties(${HTP_LIB} PROPERTIES EXPORT_COMPILE_COMMANDS ON)

install(TARGETS ${HTP_LIB})
