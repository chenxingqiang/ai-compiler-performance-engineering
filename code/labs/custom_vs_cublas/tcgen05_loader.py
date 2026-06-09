"""
Self-contained tcgen05 kernel loader for the Matching cuBLAS lab.

This module JIT-compiles the tcgen05 GEMM kernels without depending on
any other chapter or common code.

ONLY includes working kernels that exist in this directory.
"""

from __future__ import annotations

import hashlib
import importlib.util
from functools import lru_cache
from pathlib import Path
import sys

import torch
from torch.utils.cpp_extension import _get_build_directory, load

_LAB_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _LAB_DIR.parents[1]

# CUTLASS include paths
_CUTLASS_CANDIDATES = [
    _REPO_ROOT / "third_party" / "cutlass" / "include",
    _REPO_ROOT / "third_party" / "TransformerEngine" / "3rdparty" / "cutlass" / "include",
]


def _find_cutlass_include() -> Path | None:
    """Find CUTLASS include directory."""
    for cand in _CUTLASS_CANDIDATES:
        if cand.exists():
            return cand
    return None


def _get_cuda_flags() -> list[str]:
    """Get CUDA compiler flags for tcgen05."""
    flags = ["-std=c++20"]
    
    cutlass_inc = _find_cutlass_include()
    if cutlass_inc:
        flags.append(f"-I{cutlass_inc}")
    else:
        raise RuntimeError("CUTLASS include directory not found.")
    
    major, minor = torch.cuda.get_device_capability()
    if major == 10 and minor >= 3:
        # Blackwell Ultra (GB300, sm_103). sm_100a cubins are arch-locked and give
        # "no kernel image is available" on sm_103, so target sm_103a explicitly.
        flags.append("-gencode=arch=compute_103a,code=sm_103a")
    elif major >= 10:
        flags.append("-gencode=arch=compute_100a,code=sm_100a")
    else:
        raise RuntimeError(f"tcgen05 requires SM 10.0+ (Blackwell). Got SM {major}.{minor}")
    
    return flags


def _load_kernel(source_file: Path, name_prefix: str):
    """Generic kernel loader with caching."""
    if not source_file.exists():
        raise FileNotFoundError(f"{source_file.name} not found in {_LAB_DIR}")
    
    cuda_flags = _get_cuda_flags()
    src_hash = hashlib.md5(source_file.read_bytes()).hexdigest()[:8]
    build_name = f"{name_prefix}_{src_hash}"
    build_dir = Path(_get_build_directory(build_name, verbose=False))
    shared_object = build_dir / f"{build_name}.so"

    if shared_object.exists():
        loaded = sys.modules.get(build_name)
        if loaded is not None and getattr(loaded, "__file__", None) == str(shared_object):
            return loaded

        spec = importlib.util.spec_from_file_location(build_name, shared_object)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to import cached extension {shared_object}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[build_name] = module
        spec.loader.exec_module(module)
        return module
    
    print(f"  [Compiling {source_file.name} (first time only)...]")
    module = load(
        name=build_name,
        sources=[str(source_file)],
        extra_cuda_cflags=cuda_flags,
        extra_cflags=["-std=c++20"],
        extra_ldflags=["-lcuda"],
        verbose=False,
    )
    return module


# =============================================================================
# Stage 2: Basic tcgen05
# =============================================================================

@lru_cache(maxsize=1)
def load_tcgen05_module():
    """JIT-compile the basic tcgen05 GEMM kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_gemm.cu", "lab_tcgen05")


def matmul_tcgen05(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute tcgen05 GEMM: C = A @ B^T"""
    return load_tcgen05_module().matmul_tcgen05(a, b)


# =============================================================================
# Stage 3: 2-Stage Pipeline
# =============================================================================

@lru_cache(maxsize=1)
def load_tcgen05_pipelined_module():
    """JIT-compile the 2-stage pipelined kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_pipelined.cu", "lab_tcgen05_pipelined")


def matmul_tcgen05_pipelined(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute 2-stage pipelined tcgen05 GEMM: C = A @ B^T"""
    return load_tcgen05_pipelined_module().matmul_tcgen05_pipelined(a, b)


# =============================================================================
# Stage 4: 3-Stage Pipeline
# =============================================================================

@lru_cache(maxsize=1)
def load_tcgen05_3stage_module():
    """JIT-compile the 3-stage pipelined kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_3stage.cu", "lab_tcgen05_3stage")


def matmul_tcgen05_3stage(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute 3-stage pipelined tcgen05 GEMM: C = A @ B^T"""
    return load_tcgen05_3stage_module().matmul_tcgen05_3stage(a, b)


# =============================================================================
# Stage 5: Swizzled Tiles
# =============================================================================

@lru_cache(maxsize=1)
def load_tcgen05_swizzled_module():
    """JIT-compile the swizzled tile scheduling kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_swizzled.cu", "lab_tcgen05_swizzled")


def matmul_tcgen05_swizzled(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute swizzled tcgen05 GEMM: C = A @ B^T"""
    return load_tcgen05_swizzled_module().matmul_tcgen05_swizzled(a, b)


# =============================================================================
# Stage 6: Cluster (2x1) 
# =============================================================================

@lru_cache(maxsize=1)
def load_tcgen05_cluster_module():
    """JIT-compile the cluster launch kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_cluster.cu", "lab_tcgen05_cluster")


def matmul_tcgen05_cluster(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute tcgen05 GEMM with 2x1 cluster: C = A @ B^T"""
    return load_tcgen05_cluster_module().matmul_tcgen05_cluster(a, b)


# =============================================================================
# Stage 7: 4-Stage Deep Pipeline
# =============================================================================

@lru_cache(maxsize=1)
def load_tcgen05_warp_spec_module():
    """JIT-compile the 4-stage warp-specialized kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_warp_spec.cu", "lab_tcgen05_warp_spec")


def matmul_tcgen05_warp_spec(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute 4-stage deep pipelined tcgen05 GEMM: C = A @ B^T"""
    return load_tcgen05_warp_spec_module().matmul_tcgen05_warp_spec(a, b)


# =============================================================================
# Stage 8: No-Wait Pattern (KEY BREAKTHROUGH!)
# =============================================================================

@lru_cache(maxsize=1)
def load_tcgen05_no_wait_module():
    """JIT-compile the no-wait pattern kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_no_wait.cu", "lab_tcgen05_no_wait")


def matmul_tcgen05_no_wait(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute no-wait tcgen05 GEMM: C = A @ B^T
    
    KEY OPTIMIZATION: Don't wait for MMA barrier after each k-tile!
    +43% performance improvement.
    """
    return load_tcgen05_no_wait_module().matmul_tcgen05_no_wait(a, b)


# =============================================================================
# Stage 9: No-Wait + Swizzle
# =============================================================================

@lru_cache(maxsize=1)
def load_tcgen05_no_wait_swizzle_module():
    """JIT-compile the no-wait swizzled kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_no_wait_swizzle.cu", "lab_tcgen05_no_wait_swizzle")


def matmul_tcgen05_no_wait_swizzle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute no-wait + swizzled tcgen05 GEMM: C = A @ B^T"""
    return load_tcgen05_no_wait_swizzle_module().matmul_tcgen05_no_wait_swizzle(a, b)


# =============================================================================
# Stage 10: TMA Before Wait (Warp Parallel)
# =============================================================================

@lru_cache(maxsize=1)
def load_tcgen05_warp_parallel_module():
    """JIT-compile the warp-parallel kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_warp_parallel.cu", "lab_tcgen05_warp_parallel")


def matmul_tcgen05_warp_parallel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute warp-parallel tcgen05 GEMM: C = A @ B^T
    
    Issues next TMA before waiting for current one.
    """
    return load_tcgen05_warp_parallel_module().matmul_tcgen05_warp_parallel(a, b)
