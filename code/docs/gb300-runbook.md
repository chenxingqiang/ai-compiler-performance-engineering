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

## L4 ncu-grounded Speed-of-Light: ch09 NVFP4 GEMM (decode shapes, sm_103)

ncu `--set`-style metrics (kernel-replay, isolated, deterministic over 8 launches)
on `optimized_cutlass_gemm_fp4_sm103`, GB300 (148 SMs), shape M=128/N=7168/K=16384:

- Kernel: `SM100_MMA_MXF4_SS<float_e2m1_t, ...>` (NVFP4 tensor-core blockscaled),
  CTA tile 128x64x256, grid (1, 112, 1) = 112 CTAs, block 256.
- `gpu__dram_throughput.avg.pct_of_peak` = 43-46% (mean ~45%).
- `sm__throughput.avg.pct_of_peak` = 33-36% (mean ~35%).
- Wall-clock (un-profiled) 13.1 us = ~2.29 PFLOPS = 15% of the NVFP4 FLOP roofline
  (GB300 nvfp4_dense 15.0 PFLOPS), and ~59% of the HBM-bound minimum
  (B = 58.7 MB / 8.0 TB/s = 7.7 us).

VERDICT (L4): at the decode shape (M=128) this kernel is OCCUPANCY/LATENCY-bound,
not compute- or memory-bound. M=128 == tile_M so there is only ONE row-tile;
N=7168 / tile_N=64 = 112 col-tiles, so the grid is 112 CTAs and under-fills the
148-SM GPU (1 CTA/SM, 36 SMs idle, no second wave). Neither the tensor cores (SM
35%) nor HBM (DRAM 45%) are saturated. This is the physics of small-M GEMM, not a
kernel defect: the CUTLASS NVFP4 path itself is already the well-optimized
(tensor-core, TMA, blockscaled) implementation.

NEXT LEVER (Grind Mandate, DRAFT until measured): raise the CTA count above the SM
count to fill the GPU and add a second wave. Two candidates, both per-shape and
recompile-gated: (a) smaller tile_N (64 -> 32 doubles col-tiles to 224 > 148 SMs);
(b) split-K across the large K=16384 reduction (more CTAs + an epilogue reduce).
Production serving stacks pick these per shape via a tile heuristic; the book's
ch09 example pins one tile to teach the path, so this is a characterization of the
shape regime, not a defect to patch in the example. The C++ CUTLASS path supports
sm_103a and runs the NVFP4 tensor cores correctly (the MMA atom above is the FP4
tensor-core op), so the "utilize the latest hardware" box is checked; the residual
gap is occupancy at decode-M.

Companion data point, the repo's own hand-written tcgen05 dense GEMM
(`ch10:matmul_tcgen05_vs_cublas`, the educational CUDA-C++ tcgen05 kernel): on
GB300 it measures ~4x SLOWER than cuBLAS (baseline custom-tcgen05 2.78 ms,
optimized cuBLAS = best_speedup 4.0x), i.e. ~25% of the vendor library. In
K/R/H/P/A terms the educational kernel is K3 / R4 (CUDA C++ + tcgen05) / H4 / P2
(~25% of SoTA) and cuBLAS is P4 (SoTA). That 4x gap is the lab's intended lesson (a
hand-written tensor-core kernel vs the tuned vendor library), not a fixable defect,
and it is consistent across the tcgen05 teaching kernels (`matmul_tcgen05_epilogue`
is 1.27x over its own naive baseline). The two frontier GEMM SoL reads together:
the vendor NVFP4 path is occupancy-bound at decode-M (right kernel, shape-limited),
and the educational tcgen05 path is P2-vs-P4 off cuBLAS by design.

Measurement caveat learned the hard way: the `CudaBinaryBenchmark` targets
(`nvfp4_gemm`, `nvfp4_group_gemm`, `nvfp4_dual_gemm`, `top_k_kernel_cuda`, etc.)
report their OWN internal kernel timing; a wall-clock probe of `benchmark_fn`
measures binary launch + CUDA init + the full internal iteration loop, NOT the
kernel, and overstates time by ~1000x. Use the harness (which parses the binary
stdout) or run the `*_sm103` binary directly. An early ad-hoc probe wrongly
flagged NVFP4 GEMM as "4.4 s / broken"; the real number is microseconds (above).

## Open issues found on GB300 (tier1)
- `labs/block_scaling:block_scaling`: RESOLVED (2026-06-09). Was a CUTLASS DSL
  leading-dim stride assertion under pinned `nvidia-cutlass-dsl==4.3.0` (no sm_103
  in the DSL `Arch` enum). Fixed by pinning `nvidia-cutlass-dsl[cu13]>=4.5.2`,
  vendoring the sm_103 example, and making `block_scaling_common.py` arch-aware.
  Validated on GB300: verify 0.0 abs error, 1.96x speedup (software BF16 dequant
  0.0867 ms -> hardware NVFP4 blockscaled 0.0443 ms). See the toolchain-skew note
  below for the full root cause + the 3-part fix.
- `labs/real_world_models:llama_3_1_8b`: RESOLVED (2026-06-09). Was a SIGABRT in the
  optimized variant. Matched A/B on ONE toolchain (Triton 3.7, sm_103, same GPU,
  only the compile mode differs): `mode="max-autotune"` aborts with `LLVM ERROR:
  Cannot select: intrinsic %llvm.nvvm.tcgen05.wait.st` (rc=134), `mode="default"`
  runs clean. So the trigger is max-autotune's Triton 3.7 codegen emitting an
  sm_103 `tcgen05.wait.st` its LLVM backend cannot lower (vanilla flex_attention
  with max-autotune compiles fine, so it is the model-compile autotune path, not
  FlexAttention itself). Fix: `_safe_compile_mode()` in
  `labs/real_world_models/llama_3_1_8b_optimization.py` falls back max-autotune ->
  default ONLY on sm_103 + Triton >= 3.6 (warns), preserving max-autotune on the
  pinned Triton 3.5.0 and on every other arch. Validated on GB300: optimized now
  runs (no abort) at 2.97 ms vs eager baseline 7.54 ms = 2.54x.

## Toolchain-version-skew note (important)
Both targets below were NGC base-image version skews, NOT fundamental GB300
problems, and BOTH are now RESOLVED (2026-06-09): block_scaling via the cutlass-dsl
4.5.2 + vendored sm_103 example + arch-aware lab port; llama via the
`_safe_compile_mode()` max-autotune -> default fallback on sm_103 + Triton >= 3.6.
Full root cause + fix for each:
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

  COMPLETE FIX (all 3 parts DONE + validated on GB300, 2026-06-09):
  1. cutlass-dsl: `nvidia-cutlass-dsl[cu13]>=4.5.2` in requirements_latest.txt.
     4.5.2 adds `sm_103`/`sm_103a`/`sm_103f` to the `Arch` enum (4.3.0/4.4.2 lack
     them); the `[cu13]` extra pulls the CUDA-13 backend.
  2. Vendored (not a submodule bump) the sm_103-specific example to
     `labs/block_scaling/vendor/sm103_dense_blockscaled_gemm_persistent.py`
     (cutlass main, commit 1fc71b3, BSD-3, byte-identical; see vendor/README.md).
     It imports only from the pip `cutlass` package (`cutlass.utils.blackwell_helpers`,
     `cutlass.utils.blockscaled_layout`), so it runs standalone with no sibling-file
     deps and the v4.1.0 C++ tcgen05/nvfp4 header path is left untouched (still works).
  3. `block_scaling_common.py` is now arch-aware: `_resolve_cutlass_example_path()`
     picks the vendored sm_103 example on Blackwell Ultra (CC 10.3) else the sm_100
     submodule example; `_select_blockscaled_kernel()` returns the `Sm103...` class
     and passes the extra trailing `use_tma_store=True` that sm_103 `can_implement`
     and `__init__` require (sm_100 does not); and load injects `module.cutlass_torch`
     (the sm_103 example imports `cutlass.torch` locally inside `run()`). The SF
     setup is unchanged: the sm_103 example uses the identical
     `cvt_sf_MKL_to_M32x4xrm_K4xrk_L` + atom_m=(32,4)/atom_k=4 layout as sm_100.

  VALIDATED on the GB300 pod (sm_103, cutlass-dsl 4.5.2[cu13]): the lab's own
  harness path `build_problem(compile_hardware=True)` -> `verify_close()` ->
  `run_hardware()` succeeds; verify_close = 0.0 max/mean abs error vs the software
  reference (and the example's internal ref check passes at tol 0.1); measured
  baseline (software BF16 dequant) 0.0867 ms vs optimized (hardware NVFP4
  blockscaled) 0.0443 ms = 1.96x speedup (8192x8192x1024, sf_vec=16, mma (256,128),
  cluster (2,1)). The strict-validity harness expectation write happens when the
  inventory loop reaches the lab (the foreign-process guard only blocked an ad-hoc
  concurrent run, not the fix).
- `llama` optimized (RESOLVED): the max-autotune Triton 3.7 sm_103 `tcgen05.wait.st`
  abort, fixed by the `_safe_compile_mode()` fallback (max-autotune -> default on
  sm_103 + Triton >= 3.6, preserving max-autotune on the pinned 3.5.0). Validated
  optimized 2.97 ms vs eager 7.54 ms = 2.54x; matched A/B above.
The repo's actual GB300 kernels (tcgen05, NVFP4 GEMM, MoE, blackwell_matmul) are
validated working. Both former toolchain-skew breakages are now fixed in-repo, so
the full suite is expected to be all-green on both the NGC image and the pinned
toolchain.
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

## sm_100a hardcode in lab loaders (the Phase-0 fix was incomplete)

Found 2026-06-09 (the loop surfaced `ch10:matmul_tcgen05_pipelined` failing with
`CUDA error: no kernel image is available for execution on the device`). The
Phase-0 `sm_103a` fix only covered `core/common/tcgen05/__init__.py`, but the
gencode hardcode `-gencode=arch=compute_100a,code=sm_100a` is duplicated across the
lab CUDA loaders. An sm_100a cubin is arch-locked and will NOT load on sm_103
(GB300), so every lab that compiles its kernel through one of these loaders fails
with "no kernel image" on GB300 when the loop reaches the labs phase.

Fixed (6 active loaders) so they build a fat binary that loads on BOTH B200 (sm_100)
and GB300 (sm_103). The two shapes:
- Loaders with a `get_device_capability()` guard (`tcgen05_loader.py`): emit
  sm_103a on CC 10.3, else sm_100a (matches the core/common/tcgen05 fix).
- Static `extra_cuda_cflags` lists (`cutlass_gemm/__init__.py` + its CMakeLists,
  `fullstack_cluster/capstone_extension_tcgen05.py`,
  `nvfp4_group_gemm/custom_cuda_submission.py`, `nvfp4_gemm/optimized_submission.py`,
  `nvfp4_dual_gemm/optimized_submission.py`): ADD the sm_103a gencode next to the
  sm_100a one (fat binary; no device branch needed in scope).

Validated on GB300 (clear stale cache + recompile): `matmul_tcgen05_pipelined` runs
correct (12288^2), and `nvfp4_gemm` compiles + runs with no kernel-image error.
The remaining four loaders use the identical gencode mechanism so the same fix
applies; the loop exercises them in the labs phase (guarded by the watchdog below).

Comprehensive arch sweep (2026-06-09): scanned every lab `.py` for arch flags
(`code=sm_100`/`sm_100a`, `gencode`, `TORCH_CUDA_ARCH`). Result, all ACTIVE CUDA
targets now have an sm_103 image:
- 6 loaders above: fixed (sm_103a / fat binary), validated end-to-end (all 6 labs run).
- `persistent_decode/optimized_persistent_decode_cuda.py` (target
  `persistent_decode_cuda`): was the LAST sm_100-only active target
  (`code=sm_100`, no sm_103, no PTX -> no kernel image on GB300). Fixed to the
  sibling arch set (sm_100/103/120/121) and validated (compiles + runs on sm_103).
  Distinct from the `persistent_decode` target that already passed tier1.
- Already GB300-ready (no change): `persistent_decode/tma_extension.py` +
  `optimized_tma_prefill_decode.py` (sm_100/103/120/121),
  `fullstack_cluster/capstone_extension.py` (sm_100+sm_103),
  `blackwell_matmul/grace_blackwell_extension.py` (sm_103+sm_100+PTX, why
  blackwell_matmul ran at 126x).

Not fixed (intentionally): the nvfp4 competition-submission ARCHIVES
(`top_submission_candidates/`, `modal697_candidates/`, `submission_*`, `cand_*`,
`candidate_*`, the `optimized_submission_*cacheA*` variants) carry the same latent
hardcode but have no `get_benchmark` (not run by the inventory). Bulk-apply the
same one-line gencode addition if any is ever activated.

## GB300 hazard: intermittent tcgen05 cluster-graph hang + slow hang-detection

Found 2026-06-09 during the inventory loop: `ch10:tcgen05_cluster_pipeline`
(optimized = cluster-launched tcgen05 GEMM replayed from a CUDA graph) HUNG in its
optimized-timing phase, pinning a GPU at 100% with no progress for ~55 min (the
benchmark heartbeat froze at the identical snapshot the whole time). It wedged the
entire inventory until the stuck isolated_runner was killed, after which the loop
recorded the target failed and advanced normally (ch10 finished, ch11 started).

Two distinct issues:

1. The kernel hang is INTERMITTENT. In isolation on a clean GPU the exact path
   (direct cluster matmul, CUDA-graph capture, and 50 graph replays) all run clean,
   so it is not deterministic. Treat the cluster-launch + CUDA-graph-replay
   combination for tcgen05 as a GB300 (sm_103) reliability hazard, not a guaranteed
   failure. Root-causing further needs a reliable repro (not yet found).

2. Hang-detection was far too slow. `BenchmarkDefaults.measurement_timeout_seconds`
   is 1200s, and with the timeout multiplier the effective per-target bound was
   ~3600s, so a hung sub-second microbench was not reaped for ~60 min. The harness
   DOES kill on `subprocess.TimeoutExpired` (a parent-side `communicate(timeout=...)`
   plus a process-group SIGTERM/SIGKILL), so the in-process SIGALRM cannot-interrupt-
   a-CUDA-kernel problem is handled; the issue is purely that the bound is too
   generous for catching a hang quickly.

Mitigations applied: `optimized_tcgen05_cluster_pipeline.get_config()` now pins
`setup_timeout_seconds=180` + `measurement_timeout_seconds=180` (it is a sub-second
microbench, so 180s covers a cold compile and the fast measurement while failing a
hang ~10x sooner). For resilient unattended inventory runs on GB300, prefer a tight
per-target `measurement_timeout` and `timeout_multiplier=1` so any hang fails fast
instead of eating the per-chapter wall-clock budget.

Generic guard: `docs/gb300-inventory-watchdog.sh` reaps ANY hung target during a
long `bench run` inventory, not just this one. It watches the run-progress
snapshot's embedded `timestamp`; when it stops advancing for `FREEZE_LIMIT` (default
600s) it kills the hung `isolated_runner` so the parent loop records a failure and
advances. It uses the frozen-timestamp signal (not a wall-clock age), so a
slow-but-progressing target (including a long training lab) is never killed; only a
genuinely stuck worker is. Run it alongside an unattended inventory:
`bash docs/gb300-inventory-watchdog.sh &`.
