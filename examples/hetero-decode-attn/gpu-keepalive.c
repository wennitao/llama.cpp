// gpu-keepalive: a control for the heterogeneous decode measurements (docs/backend/snapdragon/
// heterogeneous-npu-gpu.md, Section 4i "Short context"). It keeps the Adreno busy with back-to-back
// pure-ALU kernels that touch no memory, so any change in HTP-only decode speed while it runs is a
// memory-system / DVFS side effect of GPU activity, not GPU work. Usage: llama-gpu-keepalive <seconds>
// [ms_per_kernel=20].
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

static const char * src =
"__kernel void spin(__global uint * out, uint iters) {\n"
"    uint x = get_global_id(0) + 1u;\n"
"    for (uint i = 0; i < iters; ++i) { x = x * 1664525u + 1013904223u; x ^= x >> 13; }\n"
"    if (x == 0xFFFFFFFFu) out[get_global_id(0)] = x;\n"   // practically never; keeps the loop alive
"}\n";

int main(int argc, char ** argv) {
    double secs = argc > 1 ? atof(argv[1]) : 10.0;
    double ms_per = argc > 2 ? atof(argv[2]) : 20.0;
    cl_platform_id plat; cl_device_id dev; cl_int err;
    if (clGetPlatformIDs(1, &plat, NULL) != CL_SUCCESS) { fprintf(stderr, "no platform\n"); return 1; }
    if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL) != CL_SUCCESS) { fprintf(stderr, "no gpu\n"); return 1; }
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, NULL, &err);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, NULL, &err);
    if (clBuildProgram(prog, 1, &dev, "-cl-std=CL2.0", NULL, NULL) != CL_SUCCESS) {
        char log[4096]; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        fprintf(stderr, "build failed: %s\n", log); return 1;
    }
    cl_kernel k = clCreateKernel(prog, "spin", &err);
    cl_mem out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4096, NULL, &err);
    size_t gws = 256, lws = 64;   // 4 work-groups: enough to count as busy, tiny footprint
    cl_uint iters = 1u << 20;
    clSetKernelArg(k, 0, sizeof(out), &out);
    // calibrate
    clSetKernelArg(k, 1, sizeof(iters), &iters);
    clEnqueueNDRangeKernel(q, k, 1, NULL, &gws, &lws, 0, NULL, NULL); clFinish(q);
    double t0 = now_s();
    clEnqueueNDRangeKernel(q, k, 1, NULL, &gws, &lws, 0, NULL, NULL); clFinish(q);
    double per = now_s() - t0;
    iters = (cl_uint) ((double) iters * (ms_per * 1e-3) / (per > 1e-6 ? per : 1e-6));
    if (iters < 1024) iters = 1024;
    clSetKernelArg(k, 1, sizeof(iters), &iters);
    fprintf(stderr, "gpu-keepalive: calibrated %u iters per kernel (~%.1f ms), running %.1f s\n", iters, ms_per, secs);
    double end = now_s() + secs; long n = 0;
    while (now_s() < end) {
        for (int i = 0; i < 4; i++) clEnqueueNDRangeKernel(q, k, 1, NULL, &gws, &lws, 0, NULL, NULL);
        clFinish(q); n += 4;
    }
    fprintf(stderr, "gpu-keepalive: done, %ld kernels\n", n);
    return 0;
}
