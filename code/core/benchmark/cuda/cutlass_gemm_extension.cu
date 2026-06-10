// CUTLASS GEMM PyTorch extension for Chapter 14 benchmarks.
//
// Implements a thin wrapper around CUTLASS to provide real CUTLASS kernels
// accessible from Python benchmarks. The extension accepts row-major FP16
// matrices and produces FP16 output while accumulating in FP32.
//
// On GB300 (Blackwell Ultra, sm_103) the CUTLASS path uses the CUTLASS 3.x Sm100
// collective builder (GemmUniversalAdapter, TMA warp-specialized 2SM), so it runs
// on the Blackwell tensor cores. The previous version declared cutlass::arch::Sm80,
// which compiled the Ampere HMMA (mma.sync.m16n8k16) path on Blackwell and ran ~3x
// slower than cuBLAS (531 vs 1576 TFLOPS @ 4096^3 FP16), crippling the
// cuBLAS-vs-CUTLASS comparison this lab teaches. NB: building this extension for
// Blackwell needs the arch 'a' variant (sm_103a / sm_100a) for the tcgen05/TMA
// path; see cutlass_binding.py.

#include <torch/extension.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/packed_stride.hpp"

#include "cute/tensor.hpp"

using namespace cute;

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t status = (call);                                          \
        if (status != cudaSuccess) {                                          \
            throw std::runtime_error(std::string("CUDA error: ") +            \
                                     cudaGetErrorString(status));             \
        }                                                                     \
    } while (0)

namespace {

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

using ElementA = cutlass::half_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;  // 8

using ElementB = cutlass::half_t;
using LayoutB = cutlass::layout::RowMajor;  // K x N row-major (C = A @ B)
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;  // 8

using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;  // 8
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;  // 8

using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;

using TileShape = Shape<_256, _128, _64>;
using ClusterShape = Shape<_2, _1, _1>;

using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueSchedule>::CollectiveOp;

using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmSm100;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator, TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED

void validate_inputs(const torch::Tensor& A, const torch::Tensor& B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be on CUDA device");
    TORCH_CHECK(B.is_cuda(), "Input B must be on CUDA device");
    TORCH_CHECK(A.dtype() == torch::kFloat16, "Input A must be float16");
    TORCH_CHECK(B.dtype() == torch::kFloat16, "Input B must be float16");
    TORCH_CHECK(A.dim() == 2, "Input A must be 2D");
    TORCH_CHECK(B.dim() == 2, "Input B must be 2D");
    TORCH_CHECK(
        A.size(1) == B.size(0),
        "Inner dimensions must match for GEMM (A: ",
        A.sizes(),
        ", B: ",
        B.sizes(),
        ")"
    );
}

}  // namespace

torch::Tensor cutlass_gemm_fp16(const torch::Tensor& A, const torch::Tensor& B) {
    validate_inputs(A, B);
    c10::cuda::CUDAGuard device_guard(A.device());

#if !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    TORCH_CHECK(false, "CUTLASS Sm100 GEMM requires CUDA 13.0+ and SM100+ (build with sm_100a/sm_103a)");
#else
    const int m = static_cast<int>(A.size(0));
    const int k = static_cast<int>(A.size(1));
    const int n = static_cast<int>(B.size(1));

    auto C = torch::empty({m, n}, A.options());

    auto const* ptr_A = reinterpret_cast<ElementA const*>(A.data_ptr<at::Half>());
    auto const* ptr_B = reinterpret_cast<ElementB const*>(B.data_ptr<at::Half>());
    auto* ptr_C = reinterpret_cast<ElementC*>(C.data_ptr<at::Half>());

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        {ptr_A, stride_A, ptr_B, stride_B},
        {{1.0f, 0.0f}, ptr_C, stride_C, ptr_C, stride_D}};
    args.scheduler.max_swizzle_size = 1;

    Gemm gemm_op;
    const size_t workspace_size = Gemm::get_workspace_size(args);
    auto workspace = torch::empty({static_cast<int64_t>(workspace_size)},
                                  A.options().dtype(torch::kUInt8));

    auto support = gemm_op.can_implement(args);
    TORCH_CHECK(support == cutlass::Status::kSuccess,
                "CUTLASS arguments unsupported: ", cutlassGetStatusString(support));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto status = gemm_op.initialize(args, workspace.data_ptr(), stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS initialize failed: ", cutlassGetStatusString(status));
    status = gemm_op.run(stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS GEMM failed: ", cutlassGetStatusString(status));
    CUDA_CHECK(cudaGetLastError());

    return C;
#endif
}

torch::Tensor cublas_gemm_fp16(const torch::Tensor& A, const torch::Tensor& B) {
    validate_inputs(A, B);
    c10::cuda::CUDAGuard device_guard(A.device());

    const int64_t m = A.size(0);
    const int64_t k = A.size(1);
    const int64_t n = B.size(1);

    auto C = torch::empty({m, n}, A.options());

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    TORCH_CUDABLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        static_cast<int>(n),
        static_cast<int>(m),
        static_cast<int>(k),
        &alpha,
        B.data_ptr<at::Half>(),
        CUDA_R_16F,
        static_cast<int>(n),
        A.data_ptr<at::Half>(),
        CUDA_R_16F,
        static_cast<int>(k),
        &beta,
        C.data_ptr<at::Half>(),
        CUDA_R_16F,
        static_cast<int>(n),
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CUDA_CHECK(cudaGetLastError());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "cublas_gemm_fp16",
        &cublas_gemm_fp16,
        "Explicit cuBLAS GEMM (FP16 input/output, FP32 accumulate)"
    );
    m.def(
        "cutlass_gemm_fp16",
        &cutlass_gemm_fp16,
        "CUTLASS Sm100 GEMM (FP16 input/output, FP32 accumulate)"
    );
}
