// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// check if A+B is faster than just B
// check if loading affineBias into shared memory is faster than loading
// multiple times (if not try 64,16 backwards case)

template <typename T, Int K, Int V>
__global__ void dAffineReluTrivialConvolution_forwardA(
    T *inFeatures, T *outFeatures, T *affineWeight, T *affineBias,
    T *convWeight, Int input_nPlanes, Int input_stride, Int output_nPlanes,
    Int output_stride, Int nActive) {
  // nActive must be a multiple of K!!

  // Input x Weight -> Output
  // blockDim=(K,K/V,1), gridDim=(nBlocks,N,1) Volkov-blocks
  // K is a multiple of V,

  // nActive x KM -> nActive x KN - parallel over N,nActive - loop over M

  Int M = input_nPlanes / K;
  // N = gridDim.y == output_nPlanes/K
  Int n = blockIdx.y;
  outFeatures += n * K;
  convWeight += n * K;

  T O[V];
  __shared__ T I[K][K];
  __shared__ T AW[K];
  __shared__ T AB[K];
  __shared__ T CW[K][K];
  const Int tx = threadIdx.x;
  int ty[V];
#pragma unroll
  for (int v = 0; v < V; v++)
    ty[v] = threadIdx.y + v * (K / V);

  for (int m = 0; m < M; m++) {
    // Read affineWeight, affineBias and convWeight
    if (ty[0] == 0) {
      AW[tx] = affineWeight[tx];
      AB[tx] = affineBias[tx];
    }
#pragma unroll
    for (int v = 0; v < V; v++)
      CW[ty[v]][tx] = convWeight[ty[v] * output_nPlanes + tx];
    __syncthreads();

    for (Int s = blockIdx.x * K; s < nActive; s += K * gridDim.x) {
// Read input, do affine + relu, set O[]
#pragma unroll
      for (int v = 0; v < V; v++) {
        T i = inFeatures[(s + ty[v]) * input_stride + tx] * AW[tx] + AB[tx];
        I[ty[v]][tx] = (i > 0) ? i : 0;
        if (m == 0) {
          O[v] = 0;
        } else {
          O[v] = outFeatures[(s + ty[v]) * output_stride + tx];
        }
      }
      __syncthreads();

#pragma unroll
      for (int k = 0; k < K; k++)
#pragma unroll
        for (int v = 0; v < V; v++)
          O[v] += I[ty[v]][k] * CW[k][tx];
#pragma unroll
      for (int v = 0; v < V; v++)
        outFeatures[(s + ty[v]) * output_stride + tx] = O[v];
      __syncthreads();
    }
    affineWeight += K;
    affineBias += K;
    convWeight += K * output_nPlanes;
    inFeatures += K;
  }
}
template <typename T, Int K, Int V>
__global__ void dAffineReluTrivialConvolution_forwardB(
    T *inFeatures, T *outFeatures, T *affineWeight, T *affineBias,
    T *convWeight, Int input_nPlanes, Int input_stride, Int output_nPlanes,
    Int output_stride, Int nActive) {
  // Input x Weight -> Output
  // blockDim=(K,K/V,1), gridDim=(nBlocks,N,1) Volkov-blocks
  // K is a multiple of V,

  // nActive x KM -> nActive x KN - parallel over N,nActive - loop over M

  Int M = input_nPlanes / K;
  // N = gridDim.y == output_nPlanes/K
  Int n = blockIdx.y;
  outFeatures += n * K;
  convWeight += n * K;

  T O[V];
  __shared__ T I[K][K]; // zz try K+1 trick A+B+backwards
  __shared__ T AW[K];
  __shared__ T AB[K];
  __shared__ T CW[K][K];
  const Int tx = threadIdx.x;
  int ty[V];
#pragma unroll
  for (int v = 0; v < V; v++)
    ty[v] = threadIdx.y + v * (K / V);

  for (int m = 0; m < M; m++) {
    // Read affineWeight, affineBias and convWeight
    if (ty[0] == 0) {
      AW[tx] = affineWeight[tx];
      AB[tx] = affineBias[tx];
    }
#pragma unroll
    for (int v = 0; v < V; v++)
      CW[ty[v]][tx] = convWeight[ty[v] * output_nPlanes + tx];
    __syncthreads();

    for (Int s = blockIdx.x * K; s < nActive; s += K * gridDim.x) {
// Read input, do affine + relu, set O[]
#pragma unroll
      for (int v = 0; v < V; v++) {
        if (s + ty[v] < nActive) {
          T i = inFeatures[(s + ty[v]) * input_stride + tx] * AW[tx] + AB[tx];
          I[ty[v]][tx] = (i > 0) ? i : 0;
          if (m == 0) {
            O[v] = 0;
          } else {
            O[v] = outFeatures[(s + ty[v]) * output_stride + tx];
          }
        }
      }
      __syncthreads();

#pragma unroll
      for (int k = 0; k < K; k++)
#pragma unroll
        for (int v = 0; v < V; v++)
          O[v] += I[ty[v]][k] * CW[k][tx];
#pragma unroll
      for (int v = 0; v < V; v++)
        if (s + ty[v] < nActive)
          outFeatures[(s + ty[v]) * output_stride + tx] = O[v];
      __syncthreads();
    }
    affineWeight += K;
    affineBias += K;
    convWeight += K * output_nPlanes;
    inFeatures += K;
  }
}

#define FOO(T, K, V)                                                           \
  {                                                                            \
    if (input_nPlanes % K == 0 and output_nPlanes % K == 0) {                  \
      Int o = (nActive / K) * K;                                               \
      if (o > 0)                                                               \
        dAffineReluTrivialConvolution_forwardA<                                \
            T, K, V><<<dim3(std::min(o / K, (Int)512), output_nPlanes / K),    \
                       dim3(K, K / V)>>>(                                      \
            inFeatures, outFeatures, affineWeight, affineBias, convWeight,     \
            input_nPlanes, input_stride, output_nPlanes, output_stride, o);    \
      if (nActive > o)                                                         \
        dAffineReluTrivialConvolution_forwardB<                                \
            T, K, V><<<dim3(1, output_nPlanes / K), dim3(K, K / V)>>>(         \
            inFeatures + o * input_stride, outFeatures + o * output_stride,    \
            affineWeight, affineBias, convWeight, input_nPlanes, input_stride, \
            output_nPlanes, output_stride, nActive - o);                       \
      return;                                                                  \
    }                                                                          \
  }

template <typename T>
void dAffineReluTrivialConvolution_forward(T *inFeatures, T *outFeatures,
                                           T *affineWeight, T *affineBias,
                                           T *convWeight, Int input_nPlanes,
                                           Int input_stride, Int output_nPlanes,
                                           Int output_stride, Int nActive) {

  FOO(T, 64, 16)
  FOO(T, 32, 8)
  FOO(T, 16, 4)
  FOO(T, 8, 2)
  assert(false);
}
template <>
void dAffineReluTrivialConvolution_forward<double>(
    double *inFeatures, double *outFeatures, double *affineWeight,
    double *affineBias, double *convWeight, Int input_nPlanes, Int input_stride,
    Int output_nPlanes, Int output_stride, Int nActive) {

  FOO(double, 32, 8)
  FOO(double, 16, 4)
  FOO(double, 8, 2)
  assert(false);
}
#undef FOO

// dOutput x W^T -> dInput and
// Input^T x dOutput -> dW
// blockDim=(K,K/V,1), gridDim=(nBlocks,M,1)
template <typename T, Int K, Int V>
__global__ void dAffineReluTrivialConvolution_backward_dW_A(
    T *inFeatures, T *dInFeatures, T *dOutFeatures, T *affineWeight,
    T *dAffineWeight, T *affineBias, T *dAffineBias, T *convWeight,
    T *dConvWeight, Int input_nPlanes, Int input_stride, Int output_nPlanes,
    Int output_stride, Int nActive, bool additiveGrad) {
  // M = gridDim.y == input_nPlanes / K
  Int N = output_nPlanes / K;
  Int m = blockIdx.y;
  inFeatures += m * K;
  dInFeatures += m * K;
  convWeight += m * K * output_nPlanes;
  dConvWeight += m * K * output_nPlanes;
  affineWeight += m * K;
  dAffineWeight += m * K;
  affineBias += m * K;
  dAffineBias += m * K;

  T dI[V];
  T dCW[V];
  T i[V];
  T dAW = 0;
  T dAB = 0;
  __shared__ T I[K][K];
  __shared__ T dO[K][K];
  __shared__ T AW[K];
  __shared__ T AB[K];
  __shared__ T CW[K][K];
  const Int tx = threadIdx.x;
  int ty[V];
#pragma unroll
  for (int v = 0; v < V; v++)
    ty[v] = threadIdx.y + v * (K / V);
  if (ty[0] == 0) {
    AW[tx] = affineWeight[tx];
    AB[tx] = affineBias[tx];
  }

  for (int n = 0; n < N; n++) {
// Read w, reset dW
#pragma unroll
    for (int v = 0; v < V; v++) {
      CW[ty[v]][tx] = convWeight[ty[v] * output_nPlanes + tx];
      dCW[v] = 0;
    }
    __syncthreads();

    for (Int s = blockIdx.x * K; s < nActive; s += K * gridDim.x) {
#pragma unroll
      for (int v = 0; v < V; v++)
        dI[v] = 0;

      __syncthreads();
// Read input and dOutput
#pragma unroll
      for (int v = 0; v < V; v++) {
        T i_ = inFeatures[(s + ty[v]) * input_stride + tx];
        i[v] = i_;
        i_ = i_ * AW[tx] + AB[tx];
        I[ty[v]][tx] = (i_ > 0) ? i_ : 0;
        dO[ty[v]][tx] = dOutFeatures[(s + ty[v]) * output_stride + tx];
      }
      __syncthreads();
#pragma unroll
      for (int k = 0; k < K; k++)
#pragma unroll
        for (int v = 0; v < V; v++) {
          dI[v] += dO[ty[v]][k] * CW[tx][k];
          dCW[v] += I[k][ty[v]] * dO[k][tx];
        }
#pragma unroll
      for (int v = 0; v < V; v++) {
        dI[v] = (I[ty[v]][tx] > 0) ? dI[v] : 0;
        dAW += i[v] * dI[v];
        dAB += dI[v];
        if (additiveGrad)
          dInFeatures[(s + ty[v]) * input_stride + tx] += dI[v];
        else
          dInFeatures[(s + ty[v]) * input_stride + tx] = dI[v];
      }
      __syncthreads();
    }
#pragma unroll
    for (int v = 0; v < V; v++)
      atomicAdd(&dConvWeight[ty[v] * output_nPlanes + tx], dCW[v]);
    convWeight += K;
    dConvWeight += K;
    dOutFeatures += K;
    __syncthreads();
  }
  atomicAdd(&dAffineWeight[tx], dAW);
  atomicAdd(&dAffineBias[tx], dAB);
}

// dOutput x W^T -> dInput and
// Input^T x dOutput -> dW
// blockDim=(K,K/V,1), gridDim=(nBlocks,M,1)
template <typename T, Int K, Int V>
__global__ void dAffineReluTrivialConvolution_backward_dW_B(
    T *inFeatures, T *dInFeatures, T *dOutFeatures, T *affineWeight,
    T *dAffineWeight, T *affineBias, T *dAffineBias, T *convWeight,
    T *dConvWeight, Int input_nPlanes, Int input_stride, Int output_nPlanes,
    Int output_stride, Int nActive, bool additiveGrad) {
  // M = gridDim.y == input_nPlanes / K
  Int N = output_nPlanes / K;
  Int m = blockIdx.y;
  inFeatures += m * K;
  dInFeatures += m * K;
  convWeight += m * K * output_nPlanes;
  dConvWeight += m * K * output_nPlanes;
  affineWeight += m * K;
  dAffineWeight += m * K;
  affineBias += m * K;
  dAffineBias += m * K;

  T dI[V];
  T dCW[V];
  T i[V];
  T dAW = 0;
  T dAB = 0;
  __shared__ T I[K][K];
  __shared__ T dO[K][K];
  __shared__ T AW[K];
  __shared__ T AB[K];
  __shared__ T CW[K][K];
  const Int tx = threadIdx.x;
  int ty[V];
#pragma unroll
  for (int v = 0; v < V; v++)
    ty[v] = threadIdx.y + v * (K / V);
  if (ty[0] == 0) {
    AW[tx] = affineWeight[tx];
    AB[tx] = affineBias[tx];
  }

  for (int n = 0; n < N; n++) {
// Read w, reset dW
#pragma unroll
    for (int v = 0; v < V; v++) {
      CW[ty[v]][tx] = convWeight[ty[v] * output_nPlanes + tx];
      dCW[v] = 0;
    }
    __syncthreads();

    for (Int s = blockIdx.x * K; s < nActive; s += K * gridDim.x) {
#pragma unroll
      for (int v = 0; v < V; v++)
        dI[v] = 0;

      __syncthreads();
// Read input and dOutput
#pragma unroll
      for (int v = 0; v < V; v++)
        if (s + ty[v] < nActive) {
          T i_ = inFeatures[(s + ty[v]) * input_stride + tx];
          i[v] = i_;
          i_ = i_ * AW[tx] + AB[tx];
          I[ty[v]][tx] = (i_ > 0) ? i_ : 0;
          dO[ty[v]][tx] = dOutFeatures[(s + ty[v]) * output_stride + tx];
        } else {
          i[v] = 0;
          I[ty[v]][tx] = 0;
          dO[ty[v]][tx] = 0;
        }
      __syncthreads();
#pragma unroll
      for (int k = 0; k < K; k++)
#pragma unroll
        for (int v = 0; v < V; v++) {
          dI[v] += dO[ty[v]][k] * CW[tx][k];
          dCW[v] += I[k][ty[v]] * dO[k][tx];
        }
#pragma unroll
      for (int v = 0; v < V; v++)
        if (s + ty[v] < nActive) {
          dI[v] = (I[ty[v]][tx] > 0) ? dI[v] : 0;
          dAW += i[v] * dI[v];
          dAB += dI[v];
          if (additiveGrad)
            dInFeatures[(s + ty[v]) * input_stride + tx] += dI[v];
          else
            dInFeatures[(s + ty[v]) * input_stride + tx] = dI[v];
        }
      __syncthreads();
    }
#pragma unroll
    for (int v = 0; v < V; v++)
      atomicAdd(&dConvWeight[ty[v] * output_nPlanes + tx], dCW[v]);
    convWeight += K;
    dConvWeight += K;
    dOutFeatures += K;
    __syncthreads();
  }
  atomicAdd(&dAffineWeight[tx], dAW);
  atomicAdd(&dAffineBias[tx], dAB);
}

#define FOO(T, K, V)                                                           \
  {                                                                            \
    if (input_nPlanes % K == 0 and output_nPlanes % K == 0) {                  \
      Int o = (nActive / K) * K;                                               \
      if (o > 0)                                                               \
        dAffineReluTrivialConvolution_backward_dW_A<                           \
            T, K, V><<<dim3(std::min(o / K, (Int)512), input_nPlanes / K),     \
                       dim3(K, K / V)>>>(                                      \
            inFeatures, dInFeatures, dOutFeatures, affineWeight,               \
            dAffineWeight, affineBias, dAffineBias, convWeight, dConvWeight,   \
            input_nPlanes, input_stride, output_nPlanes, output_stride, o,     \
            additiveGrad);                                                     \
      if (nActive > o)                                                         \
        dAffineReluTrivialConvolution_backward_dW_B<                           \
            T, K, V><<<dim3(1, input_nPlanes / K), dim3(K, K / V)>>>(          \
            inFeatures + o * input_stride, dInFeatures + o * input_stride,     \
            dOutFeatures + o * output_stride, affineWeight, dAffineWeight,     \
            affineBias, dAffineBias, convWeight, dConvWeight, input_nPlanes,   \
            input_stride, output_nPlanes, output_stride, nActive - o,          \
            additiveGrad);                                                     \
      return;                                                                  \
    }                                                                          \
  }

template <typename T>
void dAffineReluTrivialConvolution_backward_dW(
    T *inFeatures, T *dInFeatures, T *dOutFeatures, T *affineWeight,
    T *dAffineWeight, T *affineBias, T *dAffineBias, T *convWeight,
    T *dConvWeight, Int input_nPlanes, Int input_stride, Int output_nPlanes,
    Int output_stride, Int nActive, bool additiveGrad) {
  FOO(T, 32, 8)
  FOO(T, 16, 4)
  FOO(T, 8, 2)
}
#undef FOO
