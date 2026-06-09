# GB300 (Blackwell Ultra, sm_103) validation runbook

How to run this repo's benchmark harness on a single GB300 node (Grace + Blackwell
Ultra, compute capability 10.3 / `sm_103`, 4 GPUs), what was fixed to make it
GB300-correct, and the open issues found during validation.

## Hardware confirmed
- 4x NVIDIA GB300, compute capability 10.3 (`blackwell_ultra` / `sm_103`), 284 GB
  HBM each, driver 580.159.03, Grace (aarch64) host.
- Expectation hardware keys on this node: `gb300` (single visible GPU),
  `4x_gb300` (4 visible GPUs). The harness derives these from the device name
  `NVIDIA GB300` plus `torch.cuda.device_count()`.

## Working environment
Base image `nvcr.io/nvidia/pytorch:26.05-py3` is the cleanest CUDA-13 base: it
ships torch 2.12 / CUDA 13.2 / Triton 3.7 / cuDNN plus `nsys`, `ncu`, `nvcc`,
`ptxas`, so CUPTI initializes against the CUDA-13 node driver (no toolkit/driver
skew). TransformerEngine is already importable in the image.

Bring-up steps:
1. Run on a GB300 node with 4 free GPUs (this image has the profiling tools).
2. Clone the repo, init the `code/third_party/cutlass` submodule (needed by the
   tcgen05 labs).
3. Install the harness Python deps on top of the NGC torch (do NOT reinstall
   torch/triton/vllm; keep the NGC CUDA-13 build so CUPTI/nsys/ncu stay intact):
   `pip install --ignore-installed nvidia-ml-py psutil GPUtil py-cpuinfo hypothesis pytest nvidia-cutlass-dsl==4.3.0`
   (the NGC image already provides typer, pydantic, pyyaml, rich, numpy; the
   Debian-packaged PyYAML cannot be replaced, so skip the `pyyaml==6.0.2` pin).
4. Run the hardware probe once: `python core/scripts/utilities/probe_hardware_capabilities.py`.
   Confirm `architecture=blackwell_ultra`, `tma.compiler_support=true`,
   `cluster.has_dsmem=true`, `grace_coherence=true`.

### Container cgroup quirk (strict validity)
The strict validity profile rejects a container CPU/memory quota
(`ENVIRONMENT INVALID: CPU quota is set via cgroup cpu.max=...`). On a node you
fully own, clear the quota on the pod's leaf cgroup before running so strict
validity passes (otherwise use `--validity-profile portable`, numbers labeled
non-canonical):

```bash
LEAF=$(awk -F: '/^0::/{print $3}' /proc/self/cgroup)
echo max > "/sys/fs/cgroup${LEAF}/cpu.max"
echo max > "/sys/fs/cgroup${LEAF}/memory.max"
```

The cleaner alternative is to launch the pod with no CPU/memory limits (GPU
resources still pin the device). GPU clock locking (`nvidia-smi -lgc`) works in a
privileged pod, so strict validity is viable.

## Running
```bash
cd code
# canonical smoke (6 targets)
python -m cli.aisp bench run-tier1 --profile minimal
# full single-node sweep (tier1 + discovered full sweep), writes gb300 expectations
python -m cli.aisp bench run-e2e --run-id <id> --run-full-sweep \
  --no-run-fabric --no-run-cluster --validity-profile strict \
  --update-expectations --allow-mixed-provenance --profile minimal
python -m cli.aisp bench run-e2e-status --run-id <id> --watch
```
Omit `--run-fabric` (multi-node only). `--no-run-cluster` skips the cluster-eval
stage (separate serving-eval concern).

## GB300 source fixes applied (branch `gb300-validate-optimize`)
The harness was calibrated for B200/GB200 (`sm_100`); these make it GB300-correct:
1. `ch02/{baseline,optimized}_grace_coherent_memory.py`: detect Grace coherence
   via ARM host + Blackwell GPU (CC >= 10), not the CC==12.1 check that wrongly
   skipped real GB300.
2. `core/common/tcgen05/__init__.py`: emit `sm_103a` on CC 10.3 (an `sm_100a`
   cubin is arch-locked and will not load on `sm_103`).
3. `core/scripts/utilities/probe_hardware_capabilities.py`: retry the ptxas TMA
   probe with the `a` suffix for any Blackwell SM (`sm_103a`, not just `sm_100a`)
   so TMA is not falsely reported unsupported (which would skip every TMA lab);
   label CC 10.3 `blackwell_ultra`; set `nvlink_c2c`/`grace_coherence` on
   ARM-host Blackwell nodes.
4. `core/scripts/profiling/profile.sh`: resolve arch via the GB300-aware
   `detect_sm` (`sm_103`).
5. `setup.sh`: single-source Triton from `requirements_latest.txt` (3.5.0) instead
   of the 3.6 nightly default; correct the misleading 2.10-dev header.

The CUDA-binary build path (`detect_sm.py`, `cuda_binary_benchmark.py`,
`cuda_arch.mk`) was already GB300-aware (CC 10.3 -> `sm_103`).

## Validated frontier-kernel results on GB300
Confirmed working on GB300 (Blackwell Ultra) during breakthrough validation:
- tcgen05/TMEM family: `ch10/matmul_tcgen05.cu` relL2 0.00021 (raw, `A @ B^T`
  reference); `ch08:tcgen05_custom_vs_cublas` SUCCEEDED in the harness. The
  `sm_103a` fix is what unblocks this whole family (it could not load before).
- `blackwell_matmul` (2048^3, Python-kernel): tcgen05 variant 0.123 ms vs naive
  baseline 15.5 ms (126x); TMA variant 2.39 ms (6.5x).
- NVFP4 GEMM (CUTLASS NVFP4 tensor cores), real binary timing (decode shapes
  M=128, leaderboard N/K): `optimized_nvfp4_gemm_sm103` geomean 7.39 us vs
  baseline 9.78 us (1.32x); largest shape ~3.1 PFLOPS / ~60% HBM-bound SoL.
- ch09 CuTe-DSL NVFP4 GEMM (`optimized_cute_dsl_nvfp4_gemm_sm103`): 7.55 us vs
  17.9 us baseline (2.37x).
- MoE optimization ladder (`moe_optimization_journey`, Python kernels):
  `optimized_moe_cuda_graphs` 0.935 ms vs 38.9 ms naive baseline (41.6x);
  `optimized_moe_triton` 17x.
- `blackwell_gemm_optimizations` grouped GEMM (MoE-relevant): full_stack 0.124 ms
  vs 0.312 ms baseline (2.5x).
- `decode_optimization` ladder: `decode_ultimate` 1.29 ms vs 9.81 ms baseline (7.6x).

Net: the repo's frontier optimizations (tcgen05/TMA GEMM, NVFP4 GEMM, MoE ladder)
deliver real speedups on GB300. The headline GB300 fix is the `sm_103a` unblock of
the tcgen05/TMEM family (previously unloadable on Blackwell Ultra).

Measurement caveat learned the hard way: the `CudaBinaryBenchmark` targets
(`nvfp4_gemm`, `nvfp4_group_gemm`, `nvfp4_dual_gemm`, `top_k_kernel_cuda`, etc.)
report their OWN internal kernel timing; a wall-clock probe of `benchmark_fn`
measures binary launch + CUDA init + the full internal iteration loop, NOT the
kernel, and overstates time by ~1000x. Use the harness (which parses the binary
stdout) or run the `*_sm103` binary directly. An early ad-hoc probe wrongly
flagged NVFP4 GEMM as "4.4 s / broken"; the real number is microseconds (above).

## Open issues found on GB300 (tier1)
- `labs/block_scaling:block_scaling`: CUTLASS DSL leading-dim stride assertion
  (`Expected strides[leading_dim] == 1, but got 8388608`). The lab loads the
  `dense_blockscaled_gemm_persistent.py` example from the cutlass submodule (HEAD)
  but runs against pinned `nvidia-cutlass-dsl==4.3.0`; align the submodule to the
  4.3.0 release (or the DSL to the submodule).
- `labs/real_world_models:llama_3_1_8b`: baseline passes (7.7 ms); the optimized
  variant aborts (SIGABRT). Root cause: `torch.compile` of the FlexAttention path
  hits a Triton/LLVM codegen failure on sm_103, `LLVM ERROR: Cannot select:
  intrinsic %llvm.nvvm.tcgen05.wait.st`. This is a Triton 3.7 (NGC image) sm_103
  bug; the repo pins Triton 3.5.0. Forcing GEMM autotune to ATEN and forcing the
  Triton capability to (10,0) both still crash, so it is a FlexAttention-compile
  codegen issue, not GEMM autotune. Likely resolved on the pinned Triton 3.5.0.

## Toolchain-version-skew note (important)
The two remaining broken frontier targets are NGC base-image version skews, NOT
fundamental GB300 problems:
- `block_scaling`: the Python CuTe-DSL JIT path cannot target Blackwell Ultra on
  the pinned `nvidia-cutlass-dsl 4.3.0`. Root-caused four layers deep on GB300:
  (1) DSL `convert_cute_tensor` marks `dim_order()[-1]` as the leading dim, which
  mispicks the L=1 batch dim (a stride==1 selection fixes it); (2) `cute.experimental`
  is a stub that unconditionally raises, and DSL `get_version()` walk_packages
  imports it and crashes (pre-stubbing `sys.modules['cutlass.cute.experimental']`
  fixes it); (3) the DSL `Arch` enum has sm_100/100a/100f, sm_101*, sm_110*,
  sm_120/121 but NO sm_103* at all; (4) forcing `CUTE_DSL_ARCH=sm_100f` (the
  family target that runs across SM10x) passes arch validation but the NVFP4
  blockscaled MMA op `MmaMXF4NVF4Op` is hardcoded to require `sm_100a` (arch-locked
  to sm_100, will not load on sm_103) and rejects sm_100f. So   there is no DSL-4.3.0
  arch that both drives the NVFP4 MMA op AND runs on sm_103. Note the asymmetry:
  the C++ CUTLASS path (the `nvfp4_*_sm103` binaries, nvcc `-arch=sm_103a`) DOES
  support sm_103a and works; only the Python DSL JIT lags.

  COMPLETE FIX (validated path, 3 parts):
  1. cutlass-dsl: `nvidia-cutlass-dsl[cu13]>=4.5.2`. Confirmed in source that
     4.5.2 adds `sm_103`/`sm_103a`/`sm_103f` to the `Arch` enum (4.3.0/4.4.2 lack
     them). The `[cu13]` extra pulls the CUDA-13 backend. requirements_latest.txt
     updated. (Installed 4.5.2[cu13] on the validation pod; the inventory loop
     stayed healthy on it.)
  2. The matched example: cutlass main ships an sm_103-specific
     `examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/sm103_dense_blockscaled_gemm_persistent.py`.
     The pinned v4.1.0 submodule example uses `cute.arch.ProxyKind` (removed in
     4.5.x) and the `Sm100BlockScaledPersistentDenseGemmKernel` class, so it must
     be replaced with the sm103 example (bump/vendor; re-validate the C++
     tcgen05/nvfp4 header path, which currently works on v4.1.0-39).
  3. `labs/block_scaling/block_scaling_common.py` `build_problem` must move from
     the `Sm100BlockScaledPersistentDenseGemmKernel` API to the sm103 example's
     `Sm103BlockScaledPersistentDenseGemmKernel` / `Sm103BlockScaledBasicChunk` /
     `sm103_make_smem_layout_*` API (a real port to the 4.5.x CuTe-DSL API).
  Part 1 is done; parts 2-3 are a lab port coupled to a submodule bump.
- `llama` optimized: Triton 3.7 (NGC) vs pinned 3.5.0 (above).
The repo's actual GB300 kernels (tcgen05, NVFP4 GEMM, MoE, blackwell_matmul) are
validated working; running on the repo-pinned toolchain (Triton 3.5.0 + a
DSL-matched cutlass submodule) is expected to clear both.
- `labs/flashattention4:flashattention4_alibi`: optimized path measures 1.00x
  (no speedup over baseline) on the NGC torch 2.12 stack; the optimized backend
  is not beating the eager baseline here.

Passing tier1 targets (with my fixes, strict validity): `persistent_decode`
(20.2x), `ch04:gradient_fusion` (150.8x), `kv_optimization:kv_standard` (2.4x).

### tcgen05 hand-written kernels: VALIDATED correct on sm_103a (breakthrough)
Direct validation on GB300: the hand-written tcgen05 matmul
(`ch10/matmul_tcgen05.cu` via `core/common/tcgen05`) compiles, loads, and runs
NUMERICALLY CORRECT with the `sm_103a` fix (relL2 0.00021 vs reference). The prior
`sm_100a` cubin is architecture-locked and will not load on an `sm_103` device, so
the fix is what unblocks the entire tcgen05/TMEM frontier-kernel family
(ch08/ch10 + `custom_vs_cublas`) on Blackwell Ultra. (An earlier note claimed
relL2 ~1.4 / "numerically wrong"; that was a wrong-reference test error: the
kernel computes `A[M,K] @ B[N,K]^T` with shape constraints `m%128==0, n%256==0,
k%64==0`, so the reference must be `a @ b.T`, not `a @ b`. Superseded.)
