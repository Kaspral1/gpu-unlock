#!/usr/bin/env python3
import ctypes
import time


def load_cuda():
    candidates = (
        "libcuda.so.1",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib64/libcuda.so.1",
        "/usr/lib/libcuda.so.1",
    )
    for path in candidates:
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    raise SystemExit(
        "Error: could not load libcuda.so.1. Make sure the NVIDIA driver is installed."
    )


cuda = load_cuda()


def check(err, msg=""):
    if err != 0:
        raise RuntimeError(f"CUDA error {err} at {msg}")


def make_ptx(f32_accum=True, unroll=64):
    body = []
    if f32_accum:
        for _ in range(unroll):
            body.append("    mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%f0, %f1, %f2, %f3}, {%r0, %r1}, {%r2}, {%f0, %f1, %f2, %f3};")
            body.append("    mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%f4, %f5, %f6, %f7}, {%r0, %r1}, {%r2}, {%f4, %f5, %f6, %f7};")
            body.append("    mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%f8, %f9, %f10, %f11}, {%r0, %r1}, {%r2}, {%f8, %f9, %f10, %f11};")
            body.append("    mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%f12, %f13, %f14, %f15}, {%r0, %r1}, {%r2}, {%f12, %f13, %f14, %f15};")
    else:
        for _ in range(unroll):
            body.append("    mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%r4, %r5}, {%r0, %r1}, {%r2}, {%r4, %r5};")
            body.append("    mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%r6, %r7}, {%r0, %r1}, {%r2}, {%r6, %r7};")
            body.append("    mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%r8, %r9}, {%r0, %r1}, {%r2}, {%r8, %r9};")
            body.append("    mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%r10, %r11}, {%r0, %r1}, {%r2}, {%r10, %r11};")
    insts = "\n".join(body)
    return f"""
.version 6.5
.target sm_75
.address_size 64

.visible .entry bench_mma(
    .param .u64 out_ptr,
    .param .u32 iters
)
{{
    .reg .pred %p<2>;
    .reg .b32 %r<16>;
    .reg .f32 %f<16>;
    .reg .b64 %rd<4>;

    ld.param.u64 %rd0, [out_ptr];
    ld.param.u32 %r12, [iters];
    mov.u32 %r0, 0x3c003c00;
    mov.u32 %r1, 0x3c003c00;
    mov.u32 %r2, 0x3c003c00;
    mov.u32 %r4, 0;
    mov.u32 %r5, 0;
    mov.u32 %r6, 0;
    mov.u32 %r7, 0;
    mov.u32 %r8, 0;
    mov.u32 %r9, 0;
    mov.u32 %r10, 0;
    mov.u32 %r11, 0;
    mov.f32 %f0, 0f00000000;
    mov.f32 %f1, 0f00000000;
    mov.f32 %f2, 0f00000000;
    mov.f32 %f3, 0f00000000;
    mov.f32 %f4, 0f00000000;
    mov.f32 %f5, 0f00000000;
    mov.f32 %f6, 0f00000000;
    mov.f32 %f7, 0f00000000;
    mov.f32 %f8, 0f00000000;
    mov.f32 %f9, 0f00000000;
    mov.f32 %f10, 0f00000000;
    mov.f32 %f11, 0f00000000;
    mov.f32 %f12, 0f00000000;
    mov.f32 %f13, 0f00000000;
    mov.f32 %f14, 0f00000000;
    mov.f32 %f15, 0f00000000;

    mov.u32 %r13, 0;
LOOP:
{insts}
    add.u32 %r13, %r13, 1;
    setp.lt.u32 %p0, %r13, %r12;
    @%p0 bra LOOP;

    setp.eq.u32 %p1, %r13, 0xffffffff;
    @!%p1 bra DONE;
    add.f32 %f0, %f0, %f4;
    add.f32 %f0, %f0, %f8;
    add.f32 %f0, %f0, %f12;
    add.u32 %r4, %r4, %r6;
    add.u32 %r4, %r4, %r8;
    add.u32 %r4, %r4, %r10;
    st.global.f32 [%rd0], %f0;
    st.global.u32 [%rd0+4], %r4;
DONE:
    ret;
}}
""".encode("utf-8")


def main():
    check(cuda.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    check(cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")

    name_buf = ctypes.create_string_buffer(256)
    check(cuda.cuDeviceGetName(name_buf, 256, dev), "cuDeviceGetName")

    sm_count = ctypes.c_int()
    check(cuda.cuDeviceGetAttribute(ctypes.byref(sm_count), 16, dev), "SM_COUNT")

    ctx = ctypes.c_void_p()
    check(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate")

    d_out = ctypes.c_ulonglong()
    check(cuda.cuMemAlloc_v2(ctypes.byref(d_out), 64), "cuMemAlloc")

    def run_bench(f32_accum, label):
        unroll = 64
        ptx = make_ptx(f32_accum=f32_accum, unroll=unroll)
        mod = ctypes.c_void_p()
        check(cuda.cuModuleLoadData(ctypes.byref(mod), ptx), f"cuModuleLoadData ({label})")
        func = ctypes.c_void_p()
        check(cuda.cuModuleGetFunction(ctypes.byref(func), mod, b"bench_mma"), "cuModuleGetFunction")

        grid = sm_count.value * 8
        block = 256
        warps_total = grid * (block // 32)
        iters = ctypes.c_uint(2000)

        args = (ctypes.c_void_p * 2)(
            ctypes.cast(ctypes.byref(d_out), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(iters), ctypes.c_void_p),
        )

        for _ in range(5):
            check(cuda.cuLaunchKernel(func, grid, 1, 1, block, 1, 1, 0, None, args, None), "warmup")
        check(cuda.cuCtxSynchronize(), "sync warmup")

        runs = 20
        t0 = time.perf_counter()
        for _ in range(runs):
            check(cuda.cuLaunchKernel(func, grid, 1, 1, block, 1, 1, 0, None, args, None), "run")
        check(cuda.cuCtxSynchronize(), "sync")
        dt = (time.perf_counter() - t0) / runs

        flops = warps_total * iters.value * (unroll * 4) * 2048
        tflops = (flops / dt) / 1e12
        cuda.cuModuleUnload(mod)
        return tflops

    print(f"GPU: {name_buf.value.decode()} ({sm_count.value} SMs)")
    tflops_f16 = run_bench(False, "FP16 accumulate")
    tflops_f32 = run_bench(True, "FP16 -> FP32 accumulate")
    ratio = (tflops_f32 / tflops_f16) * 100.0
    status = "UNLOCKED" if ratio >= 85.0 else "CAPPED (stock GeForce)"
    print(f"Tensor MMA FP16 -> FP16 accumulate (100% ref): {tflops_f16:.2f} TFLOPS")
    print(f"Tensor MMA FP16 -> FP32 accumulate:            {tflops_f32:.2f} TFLOPS")
    print(f"FP32-accum / FP16-accum ratio:                 {ratio:.1f}% [{status}]")


if __name__ == "__main__":
    main()
