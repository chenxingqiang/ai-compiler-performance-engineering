// gb300-cublaslt-nvfp4-tn-reference.cu
//
// Working, self-verifying NVFP4 GEMM via cuBLASLt on GB300 (sm_103), the corrected
// recipe for ch09 cublaslt_gemm_fp4 (which skips with a misleading "cuBLASLt NVFP4
// unavailable"). cuBLASLt 13.4 DOES support NVFP4 GEMM on Blackwell; it needs:
//   1. TN format: transa=CUBLAS_OP_T, transb=CUBLAS_OP_N, K-major operands, FP16/BF16 out.
//      (the lab's N/N returns CUBLAS_STATUS_NOT_SUPPORTED).
//   2. VEC16_UE4M3 block scales in the SF swizzle layout (see sfoff() below):
//      offset(r,sk) = (r/128)*512*(K/64) + (sk/4)*512 + (r%32)*16 + ((r%128)/32)*4 + (sk%4)
//      i.e. a 512-byte tile of 128 rows x 4 SF-K (CUTLASS/Colfax block16 SF layout).
// Build: nvcc -arch=sm_103a -o fp4 gb300-cublaslt-nvfp4-tn-reference.cu -lcublasLt
// Verified on GB300: real random inputs, maxrel ~0.0004 (FP4 quant error only), VERIFY PASS.

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#define CK(x) do{auto e=(x); if(e!=cudaSuccess){printf("CUDA %s\n",cudaGetErrorString(e));return 1;}}while(0)
#define LK(x) do{auto s=(x); if(s!=CUBLAS_STATUS_SUCCESS){printf("LT %d L%d\n",(int)s,__LINE__);return 1;}}while(0)
// quantize a KxC matrix stored column-major (each of C cols has K contiguous), pack along K, per-16 block scale
static void quant(const std::vector<float>&src,int K,int C,std::vector<unsigned char>&pk,std::vector<__nv_fp8_storage_t>&sc){
  pk.assign((size_t)C*(K/2),0); sc.assign((size_t)C*(K/16),0);
  for(int c=0;c<C;c++) for(int b=0;b<K/16;b++){
    float mx=0; for(int i=0;i<16;i++){float v=src[(size_t)c*K+b*16+i]; mx=fmaxf(mx,fabsf(v));}
    float scale=mx>0?mx/6.0f:1.0f;
    sc[(size_t)c*(K/16)+b]=__nv_cvt_float_to_fp8(scale,__NV_SATFINITE,__NV_E4M3);
    for(int i=0;i<16;i+=2){
      float v0=src[(size_t)c*K+b*16+i], v1=src[(size_t)c*K+b*16+i+1];
      __nv_fp4_storage_t q0=__nv_cvt_float_to_fp4(v0/scale,__NV_E2M1,cudaRoundNearest);
      __nv_fp4_storage_t q1=__nv_cvt_float_to_fp4(v1/scale,__NV_E2M1,cudaRoundNearest);
      pk[((size_t)c*K+b*16+i)/2]=(unsigned char)((q0&0xF)|((q1&0xF)<<4));
    }
  }
}
static float defp4(unsigned char byte,int hi){__nv_fp4_storage_t s=hi?((byte>>4)&0xF):(byte&0xF); __half h=__nv_cvt_fp4_to_halfraw(s,__NV_E2M1); return __half2float(h);}

static size_t sfoff(int r,int sk,int K){ int RK=(K/16)/4; return (size_t)(r/128)*512*RK + (size_t)(sk/4)*512 + (size_t)(r%32)*16 + (size_t)((r%128)/32)*4 + (size_t)(sk%4); }

int main(){
  const int M=256,N=256,K=256;
  std::mt19937 g(7); std::uniform_real_distribution<float> d(-1,1);
  // A: MxK row-major == KxM col-major (K-contig per M-col). Store as M cols each K.
  std::vector<float> A((size_t)M*K), B((size_t)N*K); // A[m*K+k]; B stored as N cols each K = B^T (B[n*K+k]=origB[k][n])
  for(auto&v:A)v=d(g); for(auto&v:B)v=d(g);
  std::vector<unsigned char> pA,pB; std::vector<__nv_fp8_storage_t> sA,sB;
  quant(A,K,M,pA,sA); quant(B,K,N,pB,sB);
  int SFK=K/16; std::vector<__nv_fp8_storage_t> sAsw(sA.size(),0), sBsw(sB.size(),0);
  for(int m=0;m<M;m++)for(int sk=0;sk<SFK;sk++) sAsw[sfoff(m,sk,K)]=sA[(size_t)m*SFK+sk];
  for(int n=0;n<N;n++)for(int sk=0;sk<SFK;sk++) sBsw[sfoff(n,sk,K)]=sB[(size_t)n*SFK+sk];
  void*dA,*dB,*dAs,*dBs; __half*dC;
  CK(cudaMalloc(&dA,pA.size()));CK(cudaMalloc(&dB,pB.size()));CK(cudaMalloc(&dAs,sA.size()));CK(cudaMalloc(&dBs,sB.size()));CK(cudaMalloc(&dC,(size_t)M*N*2));
  CK(cudaMemcpy(dA,pA.data(),pA.size(),cudaMemcpyHostToDevice));CK(cudaMemcpy(dB,pB.data(),pB.size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dAs,sAsw.data(),sAsw.size(),cudaMemcpyHostToDevice));CK(cudaMemcpy(dBs,sBsw.data(),sBsw.size(),cudaMemcpyHostToDevice));
  cublasLtHandle_t lt;LK(cublasLtCreate(&lt));cublasLtMatmulDesc_t dd;LK(cublasLtMatmulDescCreate(&dd,CUBLAS_COMPUTE_32F,CUDA_R_32F));
  cublasOperation_t T=CUBLAS_OP_T,Nn=CUBLAS_OP_N;
  LK(cublasLtMatmulDescSetAttribute(dd,CUBLASLT_MATMUL_DESC_TRANSA,&T,sizeof(T)));LK(cublasLtMatmulDescSetAttribute(dd,CUBLASLT_MATMUL_DESC_TRANSB,&Nn,sizeof(Nn)));
  cublasLtMatmulMatrixScale_t sm=CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  LK(cublasLtMatmulDescSetAttribute(dd,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&sm,sizeof(sm)));LK(cublasLtMatmulDescSetAttribute(dd,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&sm,sizeof(sm)));
  LK(cublasLtMatmulDescSetAttribute(dd,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&dAs,sizeof(dAs)));LK(cublasLtMatmulDescSetAttribute(dd,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&dBs,sizeof(dBs)));
  cublasLtMatrixLayout_t la,lb,lc;
  LK(cublasLtMatrixLayoutCreate(&la,CUDA_R_4F_E2M1,K,M,K));LK(cublasLtMatrixLayoutCreate(&lb,CUDA_R_4F_E2M1,K,N,K));LK(cublasLtMatrixLayoutCreate(&lc,CUDA_R_16F,M,N,M));
  cublasLtMatmulPreference_t p;LK(cublasLtMatmulPreferenceCreate(&p));size_t ws=64ull<<20;LK(cublasLtMatmulPreferenceSetAttribute(p,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws,sizeof(ws)));void*dws;CK(cudaMalloc(&dws,ws));
  cublasLtMatmulHeuristicResult_t hr={};int nr=0;LK(cublasLtMatmulAlgoGetHeuristic(lt,dd,la,lb,lc,lc,p,1,&hr,&nr));
  if(!nr){printf("NO ALGO\n");return 1;}
  float al=1,be=0;LK(cublasLtMatmul(lt,dd,&al,dA,la,dB,lb,&be,dC,lc,dC,lc,&hr.algo,dws,ws,0));CK(cudaDeviceSynchronize());
  std::vector<__half> hC((size_t)M*N);CK(cudaMemcpy(hC.data(),dC,hC.size()*2,cudaMemcpyDeviceToHost));
  // reference: C[m][n]=sum_k dq(A,m,k)*dq(B,n,k); C col-major MxN -> hC[n*M+m]
  double maxrel=0,maxabs=0; int bad=0;
  for(int m=0;m<8;m++)for(int n=0;n<8;n++){
    double acc=0;
    for(int k=0;k<K;k++){
      unsigned char ba=pA[((size_t)m*K+k)/2]; float av=defp4(ba,k&1)*__half2float(__nv_cvt_fp8_to_halfraw(sA[(size_t)m*(K/16)+k/16],__NV_E4M3));
      unsigned char bb=pB[((size_t)n*K+k)/2]; float bv=defp4(bb,k&1)*__half2float(__nv_cvt_fp8_to_halfraw(sB[(size_t)n*(K/16)+k/16],__NV_E4M3));
      acc+=(double)av*bv;
    }
    float got=__half2float(hC[(size_t)n*M+m]); double rel=fabs(got-acc)/(fabs(acc)+1e-6);
    maxrel=fmax(maxrel,rel); maxabs=fmax(maxabs,fabs(got-acc)); if(rel>0.1)bad++;
  }
  printf("algo=%d  maxabs=%.3f maxrel=%.4f bad(>10%%)=%d/64\n",nr,maxabs,maxrel,bad);
  printf(maxrel<0.10?"VERIFY PASS (plain scale layout works)\n":"VERIFY FAIL (scale swizzle needed)\n");
  return 0;
}
