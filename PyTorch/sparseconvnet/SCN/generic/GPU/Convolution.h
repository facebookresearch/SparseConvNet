// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef GPU_CONVOLUTION_H
#define GPU_CONVOLUTION_H
#include "../SparseConvNet.h"

template <typename T>
__global__ void Convolution_fp_bias(T *output_features, T *bias, uInt nPlanes,
                                    uInt output_stride, uInt nActive) {
  __shared__ T b[32];
  b[threadIdx.x] = bias[threadIdx.x];
  for (uInt row = blockIdx.x; row < nActive; row += 1 << 12) {
    output_features[row * output_stride + threadIdx.x] = b[threadIdx.x];
  }
}

template <typename T>
__global__ void dColumnSum(T *matrix, T *target, uInt nRows, uInt nColumns,
                           uInt nCOLUMNS) {
  uInt i = blockIdx.x * 32 + threadIdx.x;
  T t = 0;
  for (uInt j = blockIdx.y; j < nRows; j += 32)
    t += matrix[j * nCOLUMNS + i];
  atomicAdd(&target[i], t);
}
template <typename T>
void Convolution_bp_bias(T *matrix, T *target, uInt nRows, uInt nColumns,
                         uInt nCOLUMNS, cudaStream_t stream) {
  if (nColumns / 32 > 0)
    dColumnSum << <dim3(nColumns / 32, 32), 32, 0, stream>>>
        (matrix, target, nRows, nColumns, nCOLUMNS);
  if (nColumns % 32 > 0) {
    uInt o = nColumns / 32 * 32;
    dColumnSum << <dim3(1, 32), nColumns - o, 0, stream>>>
        (matrix + o, target + o, nRows, nColumns, nCOLUMNS);
  }
}

template <typename T, uInt K, uInt V>
__global__ void
dConvolution_KMxKN_forwardA(T *inFeatures, T *outFeatures, T *w, uInt *rules,
                            uInt nHot, uInt input_nPlanes, uInt input_stride,
                            uInt output_nPlanes, uInt output_stride) {
  // nHot must be a multiple of K!!

  // Input x Weight -> Output
  // blockDim=(K,K/V,1), gridDim=(nBlocks,N,1) Volkov-blocks
  // K is a multiple of V,

  // nHot x KM -> nHot x KN - parallel over N,nHot - loop over M

  uInt M = input_nPlanes / K;
  // N = gridDim.y == output_nPlanes/K
  uInt n = blockIdx.y;
  outFeatures += n * K;
  w += n * K;

  T O[V];
  __shared__ T W[K][K];
  __shared__ T I[K][K];
  uInt R0[V];
  uInt R1[V];
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

    for (uInt s = blockIdx.x * K; s < nHot; s += K * gridDim.x) {
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
template <typename T, uInt K, uInt V>
__global__ void
dConvolution_KMxKN_forwardB(T *inFeatures, T *outFeatures, T *w, uInt *rules,
                            uInt nHot, uInt input_nPlanes, uInt input_stride,
                            uInt output_nPlanes, uInt output_stride) {
  // Input x Weight -> Output
  // blockDim=(K,K/V,1), gridDim=(nBlocks,N,1) Volkov-blocks
  // K is a multiple of V,

  // nHot x KM -> nHot x KN - parallel over N,nHot - loop over M

  uInt M = input_nPlanes / K;
  // N = gridDim.y == output_nPlanes/K
  uInt n = blockIdx.y;
  outFeatures += n * K;
  w += n * K;

  T O[V];
  __shared__ T W[K][K];
  __shared__ T I[K][K];
  uInt R0[V];
  uInt R1[V];
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

    for (uInt s = blockIdx.x * K; s < nHot; s += K * gridDim.x) {
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

#define FOO(K, V)                                                              \
  {                                                                            \
    if (input_nPlanes % K == 0 and output_nPlanes % K == 0) {                  \
      uInt o = (nHot / K) * K;                                                 \
      if (o >= K)                                                              \
        dConvolution_KMxKN_forwardA<T, K, V> << <                              \
            dim3(std::min(o / K, (uInt)512), output_nPlanes / K),              \
            dim3(K, K / V), 0, stream>>>                                       \
            (inFeatures, outFeatures, w, rules, o, input_nPlanes,              \
             input_stride, output_nPlanes, output_stride);                     \
      if (nHot > o)                                                            \
        dConvolution_KMxKN_forwardB<T, K, V> << <dim3(1, output_nPlanes / K),  \
                                                 dim3(K, K / V), 0, stream>>>  \
            (inFeatures, outFeatures, w, rules + 2 * o, nHot - o,              \
             input_nPlanes, input_stride, output_nPlanes, output_stride);      \
      return;                                                                  \
    }                                                                          \
  }

template <typename T>
void dConvolution_forward(T *inFeatures, T *outFeatures, T *w, uInt *rules,
                          uInt nHot, uInt input_nPlanes, uInt input_stride,
                          uInt output_nPlanes, uInt output_stride,
                          cudaStream_t stream) {
  FOO(64, 16)
  FOO(32, 8)
  FOO(16, 4)
  FOO(8, 2)
  assert(false);
}
#undef FOO

// dOutput x W^T -> dInput and
// Input^T x dOutput -> dW
// blockDim=(K,K/V,1), gridDim=(nBlocks,M,1)
template <typename T, uInt K, uInt V>
__global__ void
dConvolution_KMxKN_backward_dW_A(T *inFeatures, T *dInFeatures, T *dOutFeatures,
                                 T *w, T *dw, uInt *rules, uInt nHot,
                                 uInt input_nPlanes, uInt input_stride,
                                 uInt output_nPlanes, uInt output_stride) {
  // M = gridDim.y == input_nPlanes / K
  uInt N = output_nPlanes / K;
  uInt m = blockIdx.y;
  inFeatures += m * K;
  dInFeatures += m * K;
  w += m * K * output_nPlanes;
  dw += m * K * output_nPlanes;

  T dI[V];
  T dW[V];
  __shared__ T I[K][K];
  __shared__ T dO[K][K];
  __shared__ T W[K][K];
  uInt R0[V];
  uInt R1[V];
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

    for (uInt s = blockIdx.x * K; s < nHot; s += K * gridDim.x) {
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
// blockDim=(K,K/V,1), gridDim=(nBlocks,M,1)
template <typename T, uInt K, uInt V>
__global__ void
dConvolution_KMxKN_backward_dW_B(T *inFeatures, T *dInFeatures, T *dOutFeatures,
                                 T *w, T *dw, uInt *rules, uInt nHot,
                                 uInt input_nPlanes, uInt input_stride,
                                 uInt output_nPlanes, uInt output_stride) {
  // M = gridDim.y == input_nPlanes / K
  uInt N = output_nPlanes / K;
  uInt m = blockIdx.y;
  inFeatures += m * K;
  dInFeatures += m * K;
  w += m * K * output_nPlanes;
  dw += m * K * output_nPlanes;

  T dI[V];
  T dW[V];
  __shared__ T I[K][K];
  __shared__ T dO[K][K];
  __shared__ T W[K][K];
  uInt R0[V];
  uInt R1[V];
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

    for (uInt s = blockIdx.x * K; s < nHot; s += K * gridDim.x) {
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

#define FOO(K, V)                                                              \
  {                                                                            \
    if (input_nPlanes % K == 0 and output_nPlanes % K == 0) {                  \
      uInt o = (nHot / K) * K;                                                 \
      if (o >= K)                                                              \
        dConvolution_KMxKN_backward_dW_A<T, K, V> << <                         \
            dim3(std::min(o / K, (uInt)512), input_nPlanes / K),               \
            dim3(K, K / V), 0, stream>>>                                       \
            (inFeatures, dInFeatures, dOutFeatures, w, dw, rules, o,           \
             input_nPlanes, input_stride, output_nPlanes, output_stride);      \
      if (nHot > o)                                                            \
        dConvolution_KMxKN_backward_dW_B<T, K, V> << <                         \
            dim3(1, input_nPlanes / K), dim3(K, K / V), 0, stream>>>           \
            (inFeatures, dInFeatures, dOutFeatures, w, dw, rules + 2 * o,      \
             nHot - o, input_nPlanes, input_stride, output_nPlanes,            \
             output_stride);                                                   \
      return;                                                                  \
    }                                                                          \
  }

template <typename T>
void dConvolution_backward_dW(T *inFeatures, T *dInFeatures, T *dOutFeatures,
                              T *w, T *dw, uInt *rules, uInt nHot,
                              uInt input_nPlanes, uInt input_stride,
                              uInt output_nPlanes, uInt output_stride,
                              cudaStream_t stream) {
  FOO(32, 8)
  FOO(16, 4)
  FOO(8, 2)
  assert(false);
}
#undef FOO

template <typename T, uInt K, uInt V>
__global__ void
dConvolution_KMxKN_forward2(T *inFeatures, T *outFeatures, T *w, uInt *rules,
                            uInt nHot, uInt input_nPlanes, uInt input_stride,
                            uInt output_nPlanes, uInt output_stride) {
  // Input x Weight -> Output
  // blockDim=(K,K/V,1), gridDim=(nBlocks,N,1) Volkov-blocks
  // K is a multiple of V,

  // nHot x input_nplanes<=KM -> nHot x output_nPlanes<=KN
  // - parallel over N,nHot - loop over M

  uInt M = (input_nPlanes + K - 1) / K;
  // N = gridDim.y ~ output_nPlanes/K
  uInt n = blockIdx.y;
  outFeatures += n * K;
  w += n * K;
  uInt KO = min(K, output_nPlanes - K * n);

  T O[V];
  __shared__ T W[K][K];
  __shared__ T I[K][K];
  __shared__ uInt R[K * 2];
  const int tx = threadIdx.x;
  int ty[V];
#pragma unroll
  for (int v = 0; v < V; v++)
    ty[v] = threadIdx.y + v * (K / V);

  for (int m = 0; m < M; m++) {
    uInt KI = min(K, input_nPlanes - K * m);

// Read w
#pragma unroll
    for (int v = 0; v < V; v++)
      if (ty[v] < KI and tx < KO)
        W[ty[v]][tx] = w[ty[v] * output_nPlanes + tx];

    for (uInt s = blockIdx.x * K; s < nHot; s += K * gridDim.x) {
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

template <typename T>
void dConvolution_forward2(T *inFeatures, T *outFeatures, T *w, uInt *rules,
                           uInt nHot, uInt input_nPlanes, uInt input_stride,
                           uInt output_nPlanes, uInt output_stride,
                           cudaStream_t stream) {
  if (input_nPlanes % 8 != 0 or output_nPlanes % 8 != 0) {
    const int K = 16;
    const int V = 4;
    dConvolution_KMxKN_forward2<T, K, V> << <
        dim3(128, (output_nPlanes + K - 1) / K), dim3(K, K / V), 0, stream>>>
        (inFeatures, outFeatures, w, rules, nHot, input_nPlanes, input_stride,
         output_nPlanes, output_stride);
    return;
  } else {
    dConvolution_forward(inFeatures, outFeatures, w, rules, nHot, input_nPlanes,
                         input_stride, output_nPlanes, output_stride, stream);
  }
}

// dOutput x W^T -> dInput and
// Input^T x dOutput -> dW
// blockDim=(K,K/V,1), gridDim=(nBlocks,M,1)
template <typename T, uInt K, uInt V>
__global__ void
dConvolution_KMxKN_backward_dW2(T *inFeatures, T *dInFeatures, T *dOutFeatures,
                                T *w, T *dw, uInt *rules, uInt nHot,
                                uInt input_nPlanes, uInt input_stride,
                                uInt output_nPlanes, uInt output_stride) {
  // M = gridDim.y == input_nPlanes / K
  uInt N = (output_nPlanes + K - 1) / K;
  uInt m = blockIdx.y;
  inFeatures += m * K;
  dInFeatures += m * K;
  w += m * K * output_nPlanes;
  dw += m * K * output_nPlanes;
  uInt KI = min(K, input_nPlanes - K * m);

  T dI[V];
  T dW[V];
  __shared__ T I[K][K];
  __shared__ T dO[K][K];
  __shared__ T W[K][K];
  __shared__ uInt R[K * 2];
  const int tx = threadIdx.x;
  int ty[V];
#pragma unroll
  for (int v = 0; v < V; v++)
    ty[v] = threadIdx.y + v * (K / V);

  for (int n = 0; n < N; n++) {
    uInt KO = min(K, output_nPlanes - K * n);

// Read w, reset dW
#pragma unroll
    for (int v = 0; v < V; v++) {
      if (ty[v] < KI and tx < KO)
        W[ty[v]][tx] = w[ty[v] * output_nPlanes + tx];
      dW[v] = 0;
    }

    for (uInt s = blockIdx.x * K; s < nHot; s += K * gridDim.x) {
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
void dConvolution_backward_dW2(T *inFeatures, T *dInFeatures, T *dOutFeatures,
                               T *w, T *dw, uInt *rules, uInt nHot,
                               uInt input_nPlanes, uInt input_stride,
                               uInt output_nPlanes, uInt output_stride,
                               cudaStream_t stream) {
  if (input_nPlanes % 8 != 0 or output_nPlanes % 8 != 0) {
    const int K = 16;
    const int V = 4;
    dConvolution_KMxKN_backward_dW2<T, K, V> << <
        dim3(128, (input_nPlanes + K - 1) / K), dim3(K, K / V), 0, stream>>>
        (inFeatures, dInFeatures, dOutFeatures, w, dw, rules, nHot,
         input_nPlanes, input_stride, output_nPlanes, output_stride);
    return;
  } else {
    dConvolution_backward_dW(inFeatures, dInFeatures, dOutFeatures, w, dw,
                             rules, nHot, input_nPlanes, input_stride,
                             output_nPlanes, output_stride, stream);
  }
}

#endif /* GPU_CONVOLUTION_H */
