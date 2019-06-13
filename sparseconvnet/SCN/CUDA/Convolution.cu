// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "RuleBookIterator.h"
#define TACC double

template <typename T>
__global__ void Convolution_fp_bias_(T *output_features, T *bias, Int nPlanes,
				     Int nActive) {
  Int n = blockIdx.x * 32 + threadIdx.x;
  T b = bias[n];
  output_features += n;
  for (Int row = blockIdx.y; row < nActive; row += gridDim.y) {
    output_features[row * nPlanes] = b;
  }
}

template <typename T>
void Convolution_fp_bias(T *oF, T *b, Int nPlanes, Int nActive) {
  if (nPlanes / 32 > 0)
    Convolution_fp_bias_<<<dim3(nPlanes / 32, 4096), 32>>>(oF, b, nPlanes,
							   nActive);
  if (nPlanes % 32 > 0) {
    Int o = nPlanes / 32 * 32;
    Convolution_fp_bias_<<<dim3(1, 4096), nPlanes - o>>>(oF + o, b + o, nPlanes,
							 nActive);
  }
}

template <typename T>
__global__ void Convolution_bp_bias_(T *d_oF, T *d_b, Int nPlanes, Int nActive) {
  Int n = blockIdx.x * 32 + threadIdx.x;
  d_oF+=n;
  TACC t = 0;
  for (Int row = blockIdx.y; row < nActive; row += gridDim.y)
    t += d_oF[row * nPlanes ];
  atomicAdd(&d_b[n], t);
}
template <typename T>
void Convolution_bp_bias(T *d_oF, T *d_b, Int nPlanes, Int nActive) {
  if (nPlanes / 32 > 0)
    Convolution_bp_bias_<<<dim3(nPlanes / 32, 32), 32>>>(d_oF, d_b, nPlanes, nActive);
  if (nPlanes % 32 > 0) {
    Int o = nPlanes / 32 * 32;
    Convolution_bp_bias_<<<dim3(1, 32), nPlanes - o>>>(d_oF + o, d_b + o, nPlanes, nActive);
  }
}


// .._nPlanes == planes per nGroup
// weight = nGroups x input_nPlanes x output_nPlanes
//        = nGroups x M*K           x N*K

template <typename T, Int K, Int V>
__global__ void
dConvolution_KMxKN_forwardA(T *inFeatures, T *outFeatures, T *w, Int *rules,
			    Int nHot, Int input_nPlanes, Int input_stride,
			    Int output_nPlanes, Int output_stride) {
  // nHot must be a multiple of K!!

  // Input x Weight -> Output
  // blockDim=(K,K/V,1), gridDim=(nBlocks,N,nGroups) Volkov-blocks
  // K is a multiple of V,

  // nHot x KM -> nHot x KN - parallel over N,nHot - loop over M

  Int M = input_nPlanes / K;
  // N = gridDim.y == output_nPlanes/K
  Int n = blockIdx.y;
  Int g = blockIdx.z;
  inFeatures += g * input_nPlanes;
  outFeatures += n * K + g * output_nPlanes;
  w += n * K + g * input_nPlanes * output_nPlanes;

  TACC O[V];
  __shared__ T W[K][K];
  __shared__ T I[K][K];
  Int R0[V];
  Int R1[V];
  const int tx = threadIdx.x;
  int ty[V];
#pragma unroll
  for (int v = 0; v < V; v++)
    ty[v] = threadIdx.y + v * (K / V);

  for (int m = 0; m < M; m++) {
// Read w
#pragma unroll
    for (int v = 0; v < V; v++)
      W[ty[v]][tx] = w[ty[v] * output_nPlanes + tx];

    for (Int s = blockIdx.x * K; s < nHot; s += K * gridDim.x) {
#pragma unroll
      for (int v = 0; v < V; v++) {
	R0[v] = rules[2 * (s + ty[v])];
	R1[v] = rules[2 * (s + ty[v]) + 1];
      }
      __syncthreads();

// Read input, reset O[]
#pragma unroll
      for (int v = 0; v < V; v++) {
	I[ty[v]][tx] = inFeatures[R0[v] * input_stride + tx];
	O[v] = 0;
      }
      __syncthreads();

#pragma unroll
      for (int k = 0; k < K; k++)
#pragma unroll
	for (int v = 0; v < V; v++)
	  O[v] += I[ty[v]][k] * W[k][tx];

#pragma unroll
      for (int v = 0; v < V; v++)
	O[v] += outFeatures[R1[v] * output_stride + tx];
#pragma unroll
      for (int v = 0; v < V; v++)
	outFeatures[R1[v] * output_stride + tx] = O[v];
      __syncthreads();
    }
    w += K * output_nPlanes;
    inFeatures += K;
  }
}
template <typename T, Int K, Int V>
__global__ void
dConvolution_KMxKN_forwardB(T *inFeatures, T *outFeatures, T *w, Int *rules,
			    Int nHot, Int input_nPlanes, Int input_stride,
			    Int output_nPlanes, Int output_stride) {
  // Input x Weight -> Output
  // blockDim=(K,K/V,1), gridDim=(nBlocks,N,nGroups) Volkov-blocks
  // K is a multiple of V,

  // nHot x KM -> nHot x KN - parallel over N,nHot - loop over M

  Int M = input_nPlanes / K;
  // N = gridDim.y == output_nPlanes/K
  Int n = blockIdx.y;
  Int g = blockIdx.z;
  inFeatures += g * input_nPlanes;
  outFeatures += n * K + g * output_nPlanes;
  w += n * K + g * input_nPlanes * output_nPlanes;

  TACC O[V];
  __shared__ T W[K][K];
  __shared__ T I[K][K];
  Int R0[V];
  Int R1[V];
  const int tx = threadIdx.x;
  int ty[V];
#pragma unroll
  for (int v = 0; v < V; v++)
    ty[v] = threadIdx.y + v * (K / V);

  for (int m = 0; m < M; m++) {
// Read w
#pragma unroll
    for (int v = 0; v < V; v++)
      W[ty[v]][tx] = w[ty[v] * output_nPlanes + tx];

    for (Int s = blockIdx.x * K; s < nHot; s += K * gridDim.x) {
#pragma unroll
      for (int v = 0; v < V; v++) {
	if (s + ty[v] < nHot) {
	  R0[v] = rules[2 * (s + ty[v])];
	  R1[v] = rules[2 * (s + ty[v]) + 1];
	}
      }
      __syncthreads();

// Read input, reset O[]
#pragma unroll
      for (int v = 0; v < V; v++) {
	if (s + ty[v] < nHot)
	  I[ty[v]][tx] = inFeatures[R0[v] * input_stride + tx];
	O[v] = 0;
      }
      __syncthreads();

#pragma unroll
      for (int k = 0; k < K; k++)
#pragma unroll
	for (int v = 0; v < V; v++)
	  O[v] += I[ty[v]][k] * W[k][tx];

#pragma unroll
      for (int v = 0; v < V; v++)
	if (s + ty[v] < nHot)
	  O[v] += outFeatures[R1[v] * output_stride + tx];
#pragma unroll
      for (int v = 0; v < V; v++)
	if (s + ty[v] < nHot)
	  outFeatures[R1[v] * output_stride + tx] = O[v];
      __syncthreads();
    }
    w += K * output_nPlanes;
    inFeatures += K;
  }
}

#define FOO(T, K, V)                                                           \
  {                                                                            \
    if (input_nPlanes % K == 0 and output_nPlanes % K == 0) {                  \
      Int o = (nHot / K) * K;                                                  \
      if (o >= K)                                                              \
	dConvolution_KMxKN_forwardA<                                           \
	    T, K, V><<<dim3(std::min(o / K, (Int)512), output_nPlanes / K, nGroups),    \
		       dim3(K, K / V)>>>(inFeatures, outFeatures, w, rules, o, \
					 input_nPlanes, input_stride,          \
					 output_nPlanes, output_stride);       \
      if (nHot > o)                                                            \
	dConvolution_KMxKN_forwardB<                                           \
	    T, K, V><<<dim3(1, output_nPlanes / K, nGroups), dim3(K, K / V )>>>(         \
	    inFeatures, outFeatures, w, rules + 2 * o, nHot - o,               \
	    input_nPlanes, input_stride, output_nPlanes, output_stride);       \
      return;                                                                  \
    }                                                                          \
  }

template <typename T>
void dConvolution_forward(T *inFeatures, T *outFeatures, T *w, Int *rules,
			  Int nHot, Int input_nPlanes, Int input_stride,
			  Int output_nPlanes, Int output_stride, Int nGroups) {
  FOO(T, 64, 16)
  FOO(T, 32, 8)
  FOO(T, 16, 4)
  FOO(T, 8, 2)
  assert(false);
}
template <>
void dConvolution_forward<double>(double *inFeatures, double *outFeatures,
				  double *w, Int *rules, Int nHot,
				  Int input_nPlanes, Int input_stride,
				  Int output_nPlanes, Int output_stride, Int nGroups) {
  FOO(double, 32, 8)
  FOO(double, 16, 4)
  FOO(double, 8, 2)
  assert(false);
}
#undef FOO

// dOutput x W^T -> dInput and
// Input^T x dOutput -> dW
// blockDim=(K,K/V,1), gridDim=(nBlocks,M,nGroups)
template <typename T, Int K, Int V>
__global__ void
dConvolution_KMxKN_backward_dW_A(T *inFeatures, T *dInFeatures, T *dOutFeatures,
				 T *w, T *dw, Int *rules, Int nHot,
				 Int input_nPlanes, Int input_stride,
				 Int output_nPlanes, Int output_stride) {
  // M = gridDim.y == input_nPlanes / K
  Int N = output_nPlanes / K;
  Int m = blockIdx.y;
  Int g = blockIdx.z;
  inFeatures += m * K + g * input_nPlanes;
  dInFeatures += m * K + g * input_nPlanes;
  dOutFeatures += g * output_nPlanes;
  w += m * K * output_nPlanes+ g * input_nPlanes * output_nPlanes;
  dw += m * K * output_nPlanes+ g * input_nPlanes * output_nPlanes;

  TACC dI[V];
  TACC dW[V];
  __shared__ T I[K][K];
  __shared__ T dO[K][K];
  __shared__ T W[K][K];
  Int R0[V];
  Int R1[V];
  const int tx = threadIdx.x;
  int ty[V];
#pragma unroll
  for (int v = 0; v < V; v++)
    ty[v] = threadIdx.y + v * (K / V);

  for (int n = 0; n < N; n++) {
// Read w, reset dW
#pragma unroll
    for (int v = 0; v < V; v++) {
      W[ty[v]][tx] = w[ty[v] * output_nPlanes + tx];
      dW[v] = 0;
    }

    for (Int s = blockIdx.x * K; s < nHot; s += K * gridDim.x) {
#pragma unroll
      for (int v = 0; v < V; v++) {
	R0[v] = rules[2 * (s + ty[v])];
	R1[v] = rules[2 * (s + ty[v]) + 1];
	dI[v] = 0;
      }
      __syncthreads();
// Read input and dOutput
#pragma unroll
      for (int v = 0; v < V; v++) {
	I[ty[v]][tx] = inFeatures[R0[v] * input_stride + tx];
	dO[ty[v]][tx] = dOutFeatures[R1[v] * output_stride + tx];
      }
      __syncthreads();
#pragma unroll
      for (int k = 0; k < K; k++)
#pragma unroll
	for (int v = 0; v < V; v++) {
	  dI[v] += dO[ty[v]][k] * W[tx][k];
	  dW[v] += I[k][ty[v]] * dO[k][tx];
	}
#pragma unroll
      for (int v = 0; v < V; v++)
	dI[v] += dInFeatures[R0[v] * input_stride + tx];
#pragma unroll
      for (int v = 0; v < V; v++)
	dInFeatures[R0[v] * input_stride + tx] = dI[v];
      __syncthreads();
    }
#pragma unroll
    for (int v = 0; v < V; v++)
      atomicAdd(&dw[ty[v] * output_nPlanes + tx], dW[v]);
    w += K;
    dw += K;
    dOutFeatures += K;
  }
}

// dOutput x W^T -> dInput and
// Input^T x dOutput -> dW
// blockDim=(K,K/V,1), gridDim=(nBlocks,M,nGroups)
template <typename T, Int K, Int V>
__global__ void
dConvolution_KMxKN_backward_dW_B(T *inFeatures, T *dInFeatures, T *dOutFeatures,
				 T *w, T *dw, Int *rules, Int nHot,
				 Int input_nPlanes, Int input_stride,
				 Int output_nPlanes, Int output_stride) {
  // M = gridDim.y == input_nPlanes / K
  Int N = output_nPlanes / K;
  Int m = blockIdx.y;
  Int g = blockIdx.z;
  inFeatures += m * K + g * input_nPlanes;
  dInFeatures += m * K + g * input_nPlanes;
  dOutFeatures += g * output_nPlanes;
  w += m * K * output_nPlanes+ g * input_nPlanes * output_nPlanes;
  dw += m * K * output_nPlanes+ g * input_nPlanes * output_nPlanes;

  TACC dI[V];
  TACC dW[V];
  __shared__ T I[K][K];
  __shared__ T dO[K][K];
  __shared__ T W[K][K];
  Int R0[V];
  Int R1[V];
  const int tx = threadIdx.x;
  int ty[V];
#pragma unroll
  for (int v = 0; v < V; v++)
    ty[v] = threadIdx.y + v * (K / V);

  for (int n = 0; n < N; n++) {
// Read w, reset dW
#pragma unroll
    for (int v = 0; v < V; v++) {
      W[ty[v]][tx] = w[ty[v] * output_nPlanes + tx];
      dW[v] = 0;
    }

    for (Int s = blockIdx.x * K; s < nHot; s += K * gridDim.x) {
#pragma unroll
      for (int v = 0; v < V; v++) {
	if (s + ty[v] < nHot) {
	  R0[v] = rules[2 * (s + ty[v])];
	  R1[v] = rules[2 * (s + ty[v]) + 1];
	}
	dI[v] = 0;
      }
      __syncthreads();
// Read input and dOutput
#pragma unroll
      for (int v = 0; v < V; v++)
	if (s + ty[v] < nHot) {
	  I[ty[v]][tx] = inFeatures[R0[v] * input_stride + tx];
	  dO[ty[v]][tx] = dOutFeatures[R1[v] * output_stride + tx];
	} else {
	  I[ty[v]][tx] = 0;
	  dO[ty[v]][tx] = 0;
	}
      __syncthreads();
#pragma unroll
      for (int k = 0; k < K; k++)
#pragma unroll
	for (int v = 0; v < V; v++) {
	  dI[v] += dO[ty[v]][k] * W[tx][k];
	  dW[v] += I[k][ty[v]] * dO[k][tx];
	}
#pragma unroll
      for (int v = 0; v < V; v++)
	if (s + ty[v] < nHot)
	  dI[v] += dInFeatures[R0[v] * input_stride + tx];
#pragma unroll
      for (int v = 0; v < V; v++)
	if (s + ty[v] < nHot)
	  dInFeatures[R0[v] * input_stride + tx] = dI[v];
      __syncthreads();
    }
#pragma unroll
    for (int v = 0; v < V; v++)
      atomicAdd(&dw[ty[v] * output_nPlanes + tx], dW[v]);
    w += K;
    dw += K;
    dOutFeatures += K;
  }
}

#define FOO(T, K, V)                                                           \
  {                                                                            \
    if (input_nPlanes % K == 0 and output_nPlanes % K == 0) {                  \
      Int o = (nHot / K) * K;                                                  \
      if (o >= K)                                                              \
	dConvolution_KMxKN_backward_dW_A<                                      \
	    T, K, V><<<dim3(std::min(o / K, (Int)512), input_nPlanes / K, nGroups),     \
		       dim3(K, K / V)>>>(                                      \
	    inFeatures, dInFeatures, dOutFeatures, w, dw, rules, o,            \
	    input_nPlanes, input_stride, output_nPlanes, output_stride);       \
      if (nHot > o)                                                            \
	dConvolution_KMxKN_backward_dW_B<                                      \
	    T, K, V><<<dim3(1, input_nPlanes / K, nGroups), dim3(K, K / V)>>>(          \
	    inFeatures, dInFeatures, dOutFeatures, w, dw, rules + 2 * o,       \
	    nHot - o, input_nPlanes, input_stride, output_nPlanes,             \
	    output_stride);                                                    \
      return;                                                                  \
    }                                                                          \
  }

template <typename T>
void dConvolution_backward_dW(T *inFeatures, T *dInFeatures, T *dOutFeatures,
			      T *w, T *dw, Int *rules, Int nHot,
			      Int input_nPlanes, Int input_stride,
			      Int output_nPlanes, Int output_stride, Int nGroups) {
  FOO(T, 32, 8)
  FOO(T, 16, 4)
  FOO(T, 8, 2)
  assert(false);
}
#undef FOO

template <typename T, Int K, Int V>
__global__ void
dConvolution_KMxKN_forward2(T *inFeatures, T *outFeatures, T *w, Int *rules,
			    Int nHot, Int input_nPlanes, Int input_stride,
			    Int output_nPlanes, Int output_stride) {
  // Input x Weight -> Output
  // blockDim=(K,K/V,1), gridDim=(nBlocks,N,nGroups) Volkov-blocks
  // K is a multiple of V,

  // nHot x input_nplanes<=KM -> nHot x output_nPlanes<=KN
  // - parallel over N,nHot - loop over M

  Int M = (input_nPlanes + K - 1) / K;
  // N = gridDim.y ~ output_nPlanes/K
  Int n = blockIdx.y;
  Int g = blockIdx.z;
  inFeatures += g * input_nPlanes;
  outFeatures += n * K + g * output_nPlanes;
  w += n * K + g * input_nPlanes * output_nPlanes;
  Int KO = min(K, output_nPlanes - K * n);

  TACC O[V];
  __shared__ T W[K][K];
  __shared__ T I[K][K];
  __shared__ Int R[K * 2];
  const int tx = threadIdx.x;
  int ty[V];
#pragma unroll
  for (int v = 0; v < V; v++)
    ty[v] = threadIdx.y + v * (K / V);

  for (int m = 0; m < M; m++) {
    Int KI = min(K, input_nPlanes - K * m);

// Read w
#pragma unroll
    for (int v = 0; v < V; v++)
      if (ty[v] < KI and tx < KO)
	W[ty[v]][tx] = w[ty[v] * output_nPlanes + tx];

    for (Int s = blockIdx.x * K; s < nHot; s += K * gridDim.x) {
// Read rules for K input/output pairs
#pragma unroll
      for (int v = 0; v < V; v++) {
	if (ty[v] < 2) {
	  int q = ty[v] * K + tx;
	  if (s + q / 2 < nHot)
	    R[q] = rules[2 * s + q];
	}
      }
      __syncthreads();

// Read input, reset O[]
#pragma unroll
      for (int v = 0; v < V; v++) {
	if (tx < KI and s + ty[v] < nHot)
	  I[ty[v]][tx] = inFeatures[R[2 * ty[v]] * input_stride + tx];
	O[v] = 0;
      }
      __syncthreads();

#pragma unroll
      for (int k = 0; k < KI; k++)
#pragma unroll
	for (int v = 0; v < V; v++)
	  O[v] += I[ty[v]][k] * W[k][tx];
      __syncthreads();

#pragma unroll
      for (int v = 0; v < V; v++)
	if (tx < KO and s + ty[v] < nHot)
	  outFeatures[R[2 * ty[v] + 1] * output_stride + tx] += O[v];
      __syncthreads();
    }
    w += K * output_nPlanes;
    inFeatures += K;
  }
}

// dOutput x W^T -> dInput and
// Input^T x dOutput -> dW
// blockDim=(K,K/V,1), gridDim=(nBlocks,M,nGroups)
template <typename T, Int K, Int V>
__global__ void
dConvolution_KMxKN_backward_dW2(T *inFeatures, T *dInFeatures, T *dOutFeatures,
				T *w, T *dw, Int *rules, Int nHot,
				Int input_nPlanes, Int input_stride,
				Int output_nPlanes, Int output_stride) {
  // M = gridDim.y == input_nPlanes / K
  Int N = (output_nPlanes + K - 1) / K;
  Int m = blockIdx.y;
  Int g = blockIdx.z;
  inFeatures += m * K + g * input_nPlanes;
  dInFeatures += m * K + g * input_nPlanes;
  dOutFeatures += g * output_nPlanes;
  w += m * K * output_nPlanes+ g * input_nPlanes * output_nPlanes;
  dw += m * K * output_nPlanes+ g * input_nPlanes * output_nPlanes;
  Int KI = min(K, input_nPlanes - K * m);

  TACC dI[V];
  TACC dW[V];
  __shared__ T I[K][K];
  __shared__ T dO[K][K];
  __shared__ T W[K][K];
  __shared__ Int R[K * 2];
  const int tx = threadIdx.x;
  int ty[V];
#pragma unroll
  for (int v = 0; v < V; v++)
    ty[v] = threadIdx.y + v * (K / V);

  for (int n = 0; n < N; n++) {
    Int KO = min(K, output_nPlanes - K * n);

// Read w, reset dW
#pragma unroll
    for (int v = 0; v < V; v++) {
      if (ty[v] < KI and tx < KO)
	W[ty[v]][tx] = w[ty[v] * output_nPlanes + tx];
      dW[v] = 0;
    }

    for (Int s = blockIdx.x * K; s < nHot; s += K * gridDim.x) {
// Read rules for K input/output pairs, reset dI[]
#pragma unroll
      for (int v = 0; v < V; v++) {
	if (ty[v] < 2) {
	  int q = ty[v] * K + tx;
	  if (s + q / 2 < nHot)
	    R[q] = rules[2 * s + q];
	}
	dI[v] = 0;
      }
      __syncthreads();
// Read input and dOutput
#pragma unroll
      for (int v = 0; v < V; v++) {
	if (tx < KI and s + ty[v] < nHot)
	  I[ty[v]][tx] = inFeatures[R[2 * ty[v]] * input_stride + tx];
	else
	  I[ty[v]][tx] = 0;
	if (tx < KO and s + ty[v] < nHot)
	  dO[ty[v]][tx] = dOutFeatures[R[2 * ty[v] + 1] * output_stride + tx];
	else
	  dO[ty[v]][tx] = 0;
      }
      __syncthreads();
#pragma unroll
      for (int k = 0; k < KO; k++)
#pragma unroll
	for (int v = 0; v < V; v++)
	  dI[v] += dO[ty[v]][k] * W[tx][k];
#pragma unroll
      for (int k = 0; k < K; k++)
#pragma unroll
	for (int v = 0; v < V; v++)
	  dW[v] += I[k][ty[v]] * dO[k][tx];
      __syncthreads();
#pragma unroll
      for (int v = 0; v < V; v++)
	if (tx < KI and s + ty[v] < nHot)
	  dInFeatures[R[2 * ty[v]] * input_stride + tx] += dI[v];
      __syncthreads();
    }
#pragma unroll
    for (int v = 0; v < V; v++)
      if (ty[v] < KI and tx < KO)
	atomicAdd(&dw[ty[v] * output_nPlanes + tx], dW[v]);
    w += K;
    dw += K;
    dOutFeatures += K;
  }
}

template <typename T>
double dConvolution_forward2(T *inFeatures, T *outFeatures, T *w,
			     RuleBook _rules, Int input_nPlanes,
			     Int input_stride, Int output_nPlanes,
			     Int output_stride, Int nGroups) {
  Int c = input_nPlanes * output_nPlanes * nGroups;
  double flops = 0;
  if (input_nPlanes % 8 != 0 or output_nPlanes % 8 != 0) {
    const int K = 16;
    const int V = 4;
    RULEBOOKITERATOR(
	(dConvolution_KMxKN_forward2<
	    T, K,
	    V><<<dim3(128, (output_nPlanes + K - 1) / K, nGroups), dim3(K, K / V)>>>(
	    inFeatures, outFeatures, w, rbB, nHotB, input_nPlanes, input_stride,
	    output_nPlanes, output_stride));
	, w += c; flops += nHotB * c;)
  } else {
    RULEBOOKITERATOR(dConvolution_forward(inFeatures, outFeatures, w, rbB,
					  nHotB, input_nPlanes, input_stride,
					  output_nPlanes, output_stride, nGroups);
		     , w += c; flops += nHotB * c;)
  }
  return flops;
}

template <typename T>
void dConvolution_backward_dW2(T *inFeatures, T *dInFeatures, T *dOutFeatures,
			       T *w, T *dw, RuleBook _rules, Int input_nPlanes,
			       Int input_stride, Int output_nPlanes,
			       Int output_stride, Int nGroups) {
  Int c = input_nPlanes * output_nPlanes * nGroups;
  if (input_nPlanes % 8 != 0 or output_nPlanes % 8 != 0) {
    const int K = 16;
    const int V = 4;
    RULEBOOKITERATOR(
	(dConvolution_KMxKN_backward_dW2<
	    T, K,
	    V><<<dim3(128, (input_nPlanes + K - 1) / K, nGroups), dim3(K, K / V)>>>(
	    inFeatures, dInFeatures, dOutFeatures, w, dw, rbB, nHotB,
	    input_nPlanes, input_stride, output_nPlanes, output_stride));
	, w += c; dw += c;)
  } else {
    RULEBOOKITERATOR(dConvolution_backward_dW(inFeatures, dInFeatures,
					      dOutFeatures, w, dw, rbB, nHotB,
					      input_nPlanes, input_stride,
					      output_nPlanes, output_stride, nGroups);
		     , w += c; dw += c;)
  }
}
#undef TACC