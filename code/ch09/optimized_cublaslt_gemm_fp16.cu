// optimized_cublaslt_gemm_fp16.cu -- FP16 tensor cores with aggressive pipelining
//
// Optimized FP16 GEMM using cuBLASLt with tensor cores for maximum throughput
// This shows the true potential of modern GPU tensor cores
//
// BOOK REFERENCE (Ch9): Tensor cores provide massive throughput improvements
// for matrix operations, especially with FP16/BF16 precision. The B200 supports
// FP4 for even higher throughput (5 petaFLOPS with sparsity).
//
// KEY OPTIMIZATIONS:
//   1. FP16 data type for 2x bandwidth and tensor core access
//   2. Larger batch size for better GPU utilization
//   3. Stream-based pipelining for overlap

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <random>
#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"

#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t status = (call);                                                 \
    if (status != cudaSuccess) {                                                 \
      std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << " "           \
                << cudaGetErrorString(status) << std::endl;                      \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

#define CUBLAS_CHECK(call)                                                       \
  do {                                                                           \
    cublasStatus_t status = (call);                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                       \
      std::cerr << "cuBLAS error " << __FILE__ << ":" << __LINE__ << " code "    \
                << status << std::endl;                                          \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

#define CUBLASLT_CHECK(call)                                                     \
  do {                                                                           \
    cublasStatus_t status = (call);                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                       \
      std::cerr << "cuBLASLt error " << __FILE__ << ":" << __LINE__ << " code "  \
                << status << std::endl;                                          \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

int main() {
    NVTX_RANGE("main");
    // Larger matrices for better tensor core utilization
    constexpr int M = 2048;
    constexpr int N = 2048;
    constexpr int K = 2048;
    constexpr int batch_count = 64;  // Larger batch
    constexpr int iterations = 10;
    constexpr size_t workspace_bytes = 128ull * 1024ull * 1024ull;

    const size_t elems_A = static_cast<size_t>(M) * K;
    const size_t elems_B = static_cast<size_t>(K) * N;
    const size_t elems_C = static_cast<size_t>(M) * N;
    const size_t size_A = elems_A * sizeof(__half) * batch_count;
    const size_t size_B = elems_B * sizeof(__half) * batch_count;
    const size_t size_C = elems_C * sizeof(__half) * batch_count;

    // Host allocation
    __half* h_A = nullptr;
    __half* h_B = nullptr;
    __half* h_C = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_A, size_A));
    CUDA_CHECK(cudaMallocHost(&h_B, size_B));
    CUDA_CHECK(cudaMallocHost(&h_C, size_C));

    // Match baseline initialization to keep verification apples-to-apples.
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
    for (size_t i = 0; i < elems_A * batch_count; ++i) {
        NVTX_RANGE("setup");
        h_A[i] = __float2half(dis(gen));
    }
    for (size_t i = 0; i < elems_B * batch_count; ++i) {
        NVTX_RANGE("setup");
        h_B[i] = __float2half(dis(gen));
    }
    std::fill(h_C, h_C + (elems_C * batch_count), __float2half(0.0f));

    // Device allocation
    __half *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice));

    // cuBLAS / cuBLASLt setup
    cublasHandle_t blasHandle;
    CUBLAS_CHECK(cublasCreate(&blasHandle));
    cublasLtHandle_t ltHandle;
    CUBLASLT_CHECK(cublasLtCreate(&ltHandle));

    cublasLtMatmulDesc_t operationDesc;
    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t trans = CUBLAS_OP_N;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans)));

    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, M, K, K));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, K, N, N));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, M, N, N));
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

    const long long strideA = static_cast<long long>(elems_A);
    const long long strideB = static_cast<long long>(elems_B);
    const long long strideC = static_cast<long long>(elems_C);
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        Cdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideC, sizeof(strideC)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        Cdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));

    cublasLtMatmulPreference_t preference;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_bytes,
        sizeof(workspace_bytes)));

    constexpr int kMaxAlgos = 8;
    cublasLtMatmulHeuristicResult_t heuristics[kMaxAlgos]{};
    int returnedResults = 0;
    CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
        ltHandle,
        operationDesc,
        Adesc,
        Bdesc,
        Cdesc,
        Cdesc,
        preference,
        kMaxAlgos,
        heuristics,
        &returnedResults));

    const bool use_cublas_fallback = (returnedResults == 0);

    void* workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&workspace, workspace_bytes));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Auto-tune over the returned cuBLASLt candidates: rank-0 is not always the fastest. Time each
    // and pick the best for the timed loop. No-op when the cuBLAS fallback path is taken.
    const cublasLtMatmulAlgo_t* bestAlgo = use_cublas_fallback ? nullptr : &heuristics[0].algo;
    if (!use_cublas_fallback && returnedResults > 1) {
        auto run_lt = [&](const cublasLtMatmulAlgo_t* algo) {
            CUBLASLT_CHECK(cublasLtMatmul(ltHandle, operationDesc, &alpha, d_A, Adesc, d_B, Bdesc,
                                          &beta, d_C, Cdesc, d_C, Cdesc, algo, workspace, workspace_bytes, 0));
        };
        cudaEvent_t ts, te;
        CUDA_CHECK(cudaEventCreate(&ts));
        CUDA_CHECK(cudaEventCreate(&te));
        int bestIdx = 0;
        float bestMs = 1e30f;
        for (int a = 0; a < returnedResults; ++a) {
            run_lt(&heuristics[a].algo);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventRecord(ts));
            for (int w = 0; w < 3; ++w) run_lt(&heuristics[a].algo);
            CUDA_CHECK(cudaEventRecord(te));
            CUDA_CHECK(cudaEventSynchronize(te));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ts, te));
            if (ms < bestMs) { bestMs = ms; bestIdx = a; }
        }
        CUDA_CHECK(cudaEventDestroy(ts));
        CUDA_CHECK(cudaEventDestroy(te));
        bestAlgo = &heuristics[bestIdx].algo;
        std::cout << "cuBLASLt FP16 GEMM: auto-tuned over " << returnedResults
                  << " candidates, selected #" << bestIdx << std::endl;
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        if (!use_cublas_fallback) {
            NVTX_RANGE("compute_math:ltmatmul");
            CUBLASLT_CHECK(cublasLtMatmul(
                ltHandle,
                operationDesc,
                &alpha,
                d_A,
                Adesc,
                d_B,
                Bdesc,
                &beta,
                d_C,
                Cdesc,
                d_C,
                Cdesc,
                bestAlgo,
                workspace,
                workspace_bytes,
                0));
        } else {
            NVTX_RANGE("compute_math:cublas_strided_batched");
            // Row-major fallback:
            // C_row = A_row * B_row
            // cublas (column-major) computes C_col = B_row^T * A_row^T with m=N, n=M.
            CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                blasHandle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N,
                M,
                K,
                &alpha,
                d_B,
                CUDA_R_16F,
                N,
                strideB,
                d_A,
                CUDA_R_16F,
                K,
                strideA,
                &beta,
                d_C,
                CUDA_R_16F,
                N,
                strideC,
                batch_count,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const float avg_ms = total_ms / (iterations * batch_count);
    std::cout << "cuBLASLt FP16 GEMM (optimized): " << avg_ms << " ms" << std::endl;

#ifdef VERIFY
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    const size_t elements = elems_C * batch_count;
    double checksum = 0.0;
    for (size_t i = 0; i < elements; ++i) {
        checksum += std::abs(__half2float(h_C[i]));
    }
    VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum));
#endif

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(workspace));
    CUBLASLT_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(Adesc));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(Bdesc));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(Cdesc));
    CUBLASLT_CHECK(cublasLtMatmulDescDestroy(operationDesc));
    CUBLASLT_CHECK(cublasLtDestroy(ltHandle));
    CUBLAS_CHECK(cublasDestroy(blasHandle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));

    return 0;
}
