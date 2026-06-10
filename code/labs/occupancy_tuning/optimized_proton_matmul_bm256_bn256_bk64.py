#!/usr/bin/env python3
"""Optimized: Triton matmul with a large 256x256x64 tile (GB300 sweep champion).

Uses 256x256x64 blocks with 8 warps. On GB300 (sm_103) this was the fastest tile in a
direct config sweep at 8192x8192x256 (~551 TFLOPS, ~1.62x over the 64x64x32 baseline's
~340), ahead of the 128x256x64 wide-N variant. Maximizes compute density per program on
Blackwell Ultra's high-bandwidth SM.
"""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark
from labs.occupancy_tuning.triton_matmul_schedules import (
    MatmulSchedule,
    TritonMatmulProtonBenchmark,
)

SCHEDULE = MatmulSchedule(
    name="bm256_bn256_bk64",
    block_m=256,
    block_n=256,
    block_k=64,
    num_warps=8,
    notes="Large 256x256 tile, GB300 config-sweep champion (~1.62x over the small-tile baseline).",
)


class OptimizedProtonMatmulLargeTile(TritonMatmulProtonBenchmark):
    """Optimized Triton matmul with a large 256x256x64 tile (GB300 champion).

    Block config: 256x256x64, 8 warps
    Use case: maximize compute density per program on Blackwell's high-bandwidth SM.
    """

    def __init__(self, size: int = 8192):
        super().__init__(
            schedule=SCHEDULE,
            size=size,
            iterations=10,
            warmup=5,
        )


def get_benchmark() -> BaseBenchmark:
    return OptimizedProtonMatmulLargeTile()
