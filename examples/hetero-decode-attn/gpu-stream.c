// gpu-stream: a control for the heterogeneous decode measurements (docs/backend/snapdragon/
// heterogeneous-npu-gpu.md, Section 4i "Is there DRAM bandwidth left"). It streams a large buffer
// through the Adreno at a chosen duty cycle and reports GB/s once per second, so the aggregate DRAM
// bandwidth of HTP + GPU and the HTP's slowdown under co-traffic can be measured.
// Usage: llama-gpu-stream <seconds> [mb=256] [work-groups=1024] [duty=1.0]
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

static double now_s(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

static const char * src =
"__kernel void stream(__global const float4 * in, __global float * out, uint n4) {\n"
"    uint gid = get_global_id(0), gsz = get_global_size(0);\n"
"    float4 acc = (float4)(0.0f);\n"
"    for (uint i = gid; i < n4; i += gsz) acc += in[i];\n"
"    float s = acc.x + acc.y + acc.z + acc.w;\n"
"    if (s == 12345.678f) out[gid] = s;\n"
"}\n";

int main(int argc, char ** argv) {
    double secs = argc > 1 ? atof(argv[1]) : 10.0;
    size_t mb = argc > 2 ? (size_t) atoi(argv[2]) : 256;
    size_t wgs = argc > 3 ? (size_t) atoi(argv[3]) : 1024;
    double duty = argc > 4 ? atof(argv[4]) : 1.0; if (duty <= 0.05) duty = 0.05; if (duty > 1.0) duty = 1.0;
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
    cl_kernel k = clCreateKernel(prog, "stream", &err);
    size_t bytes = mb << 20;
    cl_mem in = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "alloc failed %d\n", err); return 1; }
    cl_mem out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, wgs * 64 * 4, NULL, &err);
    float pat = 1.0f; clEnqueueFillBuffer(q, in, &pat, sizeof(pat), 0, bytes, 0, NULL, NULL); clFinish(q);
    cl_uint n4 = (cl_uint) (bytes / 16);
    size_t lws = 64, gws = wgs * lws;
    clSetKernelArg(k, 0, sizeof(in), &in); clSetKernelArg(k, 1, sizeof(out), &out); clSetKernelArg(k, 2, sizeof(n4), &n4);
    clEnqueueNDRangeKernel(q, k, 1, NULL, &gws, &lws, 0, NULL, NULL); clFinish(q);  // warm
    double t_end = now_s() + secs, t_rep = now_s() + 1.0, t0 = now_s(); long n = 0, n_rep = 0;
    while (now_s() < t_end) {
        double tb = now_s();
        for (int i = 0; i < 4; i++) clEnqueueNDRangeKernel(q, k, 1, NULL, &gws, &lws, 0, NULL, NULL);
        clFinish(q); n += 4; n_rep += 4;
        double t = now_s();
        if (duty < 1.0) { double busy = t - tb; usleep((useconds_t) (busy * (1.0 - duty) / duty * 1e6)); t = now_s(); }
        if (t >= t_rep) { fprintf(stderr, "gpu-stream: %.1f GB/s\n", (double) n_rep * bytes / (t - (t_rep - 1.0)) / 1e9); t_rep = t + 1.0; n_rep = 0; }
    }
    double t1 = now_s();
    fprintf(stderr, "gpu-stream: total %ld passes of %zu MB in %.1f s = %.1f GB/s avg\n", n, mb, t1 - t0, (double) n * bytes / (t1 - t0) / 1e9);
    return 0;
}
