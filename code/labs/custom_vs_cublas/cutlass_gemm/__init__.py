"""
CUTLASS Blackwell GEMM with CollectiveBuilder

Uses pre-compiled CMake library for fast loading.
"""

import os
import sys
from pathlib import Path

_LAB_DIR = Path(__file__).parent
_BUILD_DIR = _LAB_DIR / "build"
_LIB_PATH = _BUILD_DIR / "cutlass_blackwell_gemm.so"


def load_cutlass_gemm():
    """Load the pre-compiled CUTLASS Blackwell GEMM library."""
    import torch
    from torch.utils.cpp_extension import load
    
    # If pre-compiled library exists, try to use it
    if _LIB_PATH.exists():
        # Use JIT compilation with same source - it will use cache if available
        pass
    
    # JIT compile (handles all linking correctly)
    source_file = _LAB_DIR / "cutlass_gemm.cu"
    cutlass_inc = _LAB_DIR.parent.parent.parent / "third_party" / "cutlass" / "include"
    cutlass_util_inc = _LAB_DIR.parent.parent.parent / "third_party" / "cutlass" / "tools" / "util" / "include"
    
    if not source_file.exists():
        raise FileNotFoundError(f"CUTLASS source not found: {source_file}")
    
    print("  [Loading CUTLASS Blackwell GEMM (first time compiles ~3 mins)...]")
    module = load(
        name="cutlass_blackwell_gemm",
        sources=[str(source_file)],
        extra_cuda_cflags=[
            "-std=c++20",
            f"-I{cutlass_inc}",
            f"-I{cutlass_util_inc}",
            "-gencode=arch=compute_100a,code=sm_100a",
            "-gencode=arch=compute_103a,code=sm_103a",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
        ],
        extra_cflags=["-std=c++20"],
        extra_ldflags=["-lcuda"],
        verbose=False,
    )
    return module


_module = None

def cutlass_gemm(a, b, alpha=1.0, beta=0.0):
    """
    CUTLASS Blackwell FP16 GEMM: C = alpha * A @ B^T + beta * C
    
    Uses CUTLASS CollectiveBuilder for optimal Blackwell performance:
    - SM100 TMA with multicast
    - True warp specialization
    - PipelineTmaUmmaAsync
    - 2x2 cluster launch
    """
    global _module
    if _module is None:
        _module = load_cutlass_gemm()
    return _module.cutlass_gemm(a, b, alpha, beta)

