// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef GPU_AFFINERELUTRIVIALCONVOLUTION_H
#define GPU_AFFINERELUTRIVIALCONVOLUTION_H

// check if A+B is faster than just B
// check if loading affineBias into shared memory is faster than loading
// multiple times (if not try 64,16 backwards case)

template <typename T, uInt K, uInt V>
__global__ void dAffineReluTrivialConvolution_forwardA(
    T *inFeatures, T *outFeatures, T *affineWeight, T *affineBias,
    T *convWeight, uInt input_nPlanes, uInt input_stride, uInt output_nPlanes,
    uInt output_stride, uInt nActive) {
  // nActive must be a multiple of K!!

  // Input x Weight -> Output
  // blockDim=(K,K/V,1), gridDim=(nBlocks,N,1) Volkov-blocks
  // K is a multiple of V,

  // nActive x KM -> nActive x KN - parallel over N,nActive - loop over M

  uInt M = input_nPlanes / K;
  // N = gridDim.y == output_nPlanes/K
  uInt n = blockIdx.y;
  outFeatures += n * K;
  convWeight += n * K;

  T O[V];
  __shared__ T I[K][K];
  __shared__ T AW[K];
  __shared__ T AB[K];
  __shared__ T CW[K][K];
  const uInt tx = threadIdx.x;
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

    for (uInt s = blockIdx.x * K; s < nActive; s += K * gridDim.x) {
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
template <typename T, uInt K, uInt V>
__global__ void dAffineReluTrivialConvolution_forwardB(
    T *inFeatures, T *outFeatures, T *affineWeight, T *affineBias,
    T *convWeight, uInt input_nPlanes, uInt input_stride, uInt output_nPlanes,
    uInt output_stride, uInt nActive) {
  // Input x Weight -> Output
  // blockDim=(K,K/V,1), gridDim=(nBlocks,N,1) Volkov-blocks
  // K is a multiple of V,

  // nActive x KM -> nActive x KN - parallel over N,nActive - loop over M

  uInt M = input_nPlanes / K;
  // N = gridDim.y == output_nPlanes/K
  uInt n = blockIdx.y;
  outFeatures += n * K;
  convWeight += n * K;

  T O[V];
  __shared__ T I[K][K]; // zz try K+1 trick A+B+backwards
  __shared__ T AW[K];
  __shared__ T AB[K];
  __shared__ T CW[K][K];
  const uInt tx = threadIdx.x;
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

    for (uInt s = blockIdx.x * K; s < nActive; s += K * gridDim.x) {
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

template <typename T>
void dAffineReluTrivialConvolution_forward(T *inFeatures, T *outFeatures,
                                           T *affineWeight, T *affineBias,
                                           T *convWeight, uInt input_nPlanes,
                                           uInt input_stride,
                                           uInt output_nPlanes,
                                           uInt output_stride, uInt nActive) {
  {
    const uInt K = 64;
    const uInt V = 16;
    if (input_nPlanes % K == 0 and output_nPlanes % K == 0) {
      uInt o = (nActive / K) * K;
      if (o > 0)
        dAffineReluTrivialConvolution_forwardA<
            T, K, V><<<dim3(std::min(o / K, (uInt)512), output_nPlanes / K),
                       dim3(K, K / V), 0, THCState_getCurrentStream(state)>>>(
            inFeatures, outFeatures, affineWeight, affineBias, convWeight,
            input_nPlanes, input_stride, output_nPlanes, output_stride, o);
      if (nActive > o)
        dAffineReluTrivialConvolution_forwardB<
            T, K, V><<<dim3(1, output_nPlanes / K), dim3(K, K / V), 0,
                       THCState_getCurrentStream(state)>>>(
            inFeatures + o * input_stride, outFeatures + o * output_stride,
            affineWeight, affineBias, convWeight, input_nPlanes, input_stride,
            output_nPlanes, output_stride, nActive - o);
      return;
    }
  }
  {
    const uInt K = 32;
    const uInt V = 4;
    if (input_nPlanes % K == 0 and output_nPlanes % K == 0) {
      uInt o = (nActive / K) * K;
      if (o > 0)
        dAffineReluTrivialConvolution_forwardA<
            T, K, V><<<dim3(std::min(o / K, (uInt)512), output_nPlanes / K),
                       dim3(K, K / V), 0, THCState_getCurrentStream(state)>>>(
            inFeatures, outFeatures, affineWeight, affineBias, convWeight,
            input_nPlanes, input_stride, output_nPlanes, output_stride, o);
      if (nActive > o)
        dAffineReluTrivialConvolution_forwardB<
            T, K, V><<<dim3(1, output_nPlanes / K), dim3(K, K / V), 0,
                       THCState_getCurrentStream(state)>>>(
            inFeatures + o * input_stride, outFeatures + o * output_stride,
            affineWeight, affineBias, convWeight, input_nPlanes, input_stride,
            output_nPlanes, output_stride, nActive - o);
      return;
    }
  }
  {
    const uInt K = 16;
    const uInt V = 4;
    if (input_nPlanes % K == 0 and output_nPlanes % K == 0) {
      uInt o = (nActive / K) * K;
      if (o > 0)
        dAffineReluTrivialConvolution_forwardA<
            T, K, V><<<dim3(std::min(o / K, (uInt)512), output_nPlanes / K),
                       dim3(K, K / V), 0, THCState_getCurrentStream(state)>>>(
            inFeatures, outFeatures, affineWeight, affineBias, convWeight,
            input_nPlanes, input_stride, output_nPlanes, output_stride, o);
      if (nActive > o)
        dAffineReluTrivialConvolution_forwardB<
            T, K, V><<<dim3(1, output_nPlanes / K), dim3(K, K / V), 0,
                       THCState_getCurrentStream(state)>>>(
            inFeatures + o * input_stride, outFeatures + o * output_stride,
            affineWeight, affineBias, convWeight, input_nPlanes, input_stride,
            output_nPlanes, output_stride, nActive - o);
      return;
    }
  }
  {
    const uInt K = 8;
    const uInt V = 2;
    if (input_nPlanes % K == 0 and output_nPlanes % K == 0) {
      uInt o = (nActive / K) * K;
      if (o > 0)
        dAffineReluTrivialConvolution_forwardA<
            T, K, V><<<dim3(std::min(o / K, (uInt)512), output_nPlanes / K),
                       dim3(K, K / V), 0, THCState_getCurrentStream(state)>>>(
            inFeatures, outFeatures, affineWeight, affineBias, convWeight,
            input_nPlanes, input_stride, output_nPlanes, output_stride, o);
      if (nActive > o)
        dAffineReluTrivialConvolution_forwardB<
            T, K, V><<<dim3(1, output_nPlanes / K), dim3(K, K / V), 0,
                       THCState_getCurrentStream(state)>>>(
            inFeatures + o * input_stride, outFeatures + o * output_stride,
            affineWeight, affineBias, convWeight, input_nPlanes, input_stride,
            output_nPlanes, output_stride, nActive - o);
      return;
    }
  }
  assert(false);
}

// dOutput x W^T -> dInput and
// Input^T x dOutput -> dW
// blockDim=(K,K/V,1), gridDim=(nBlocks,M,1)
template <typename T, uInt K, uInt V>
__global__ void dAffineReluTrivialConvolution_backward_dW_A(
    T *inFeatures, T *dInFeatures, T *dOutFeatures, T *affineWeight,
    T *dAffineWeight, T *affineBias, T *dAffineBias, T *convWeight,
    T *dConvWeight, uInt input_nPlanes, uInt input_stride, uInt output_nPlanes,
    uInt output_stride, uInt nActive, bool additiveGrad) {
  // M = gridDim.y == input_nPlanes / K
  uInt N = output_nPlanes / K;
  uInt m = blockIdx.y;
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
  const uInt tx = threadIdx.x;
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

    for (uInt s = blockIdx.x * K; s < nActive; s += K * gridDim.x) {
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
template <typename T, uInt K, uInt V>
__global__ void dAffineReluTrivialConvolution_backward_dW_B(
    T *inFeatures, T *dInFeatures, T *dOutFeatures, T *affineWeight,
    T *dAffineWeight, T *affineBias, T *dAffineBias, T *convWeight,
    T *dConvWeight, uInt input_nPlanes, uInt input_stride, uInt output_nPlanes,
    uInt output_stride, uInt nActive, bool additiveGrad) {
  // M = gridDim.y == input_nPlanes / K
  uInt N = output_nPlanes / K;
  uInt m = blockIdx.y;
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
  const uInt tx = threadIdx.x;
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

    for (uInt s = blockIdx.x * K; s < nActive; s += K * gridDim.x) {
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

template <typename T>
void dAffineReluTrivialConvolution_backward_dW(
    T *inFeatures, T *dInFeatures, T *dOutFeatures, T *affineWeight,
    T *dAffineWeight, T *affineBias, T *dAffineBias, T *convWeight,
    T *dConvWeight, uInt input_nPlanes, uInt input_stride, uInt output_nPlanes,
    uInt output_stride, uInt nActive, bool additiveGrad) {
  {
    const uInt K = 32;
    const uInt V = 8;
    if (input_nPlanes % K == 0 and output_nPlanes % K == 0) {
      uInt o = (nActive / K) * K;
      if (o > 0)
        dAffineReluTrivialConvolution_backward_dW_A<
            T, K, V><<<dim3(std::min(o / K, (uInt)512), input_nPlanes / K),
                       dim3(K, K / V), 0, THCState_getCurrentStream(state)>>>(
            inFeatures, dInFeatures, dOutFeatures, affineWeight, dAffineWeight,
            affineBias, dAffineBias, convWeight, dConvWeight, input_nPlanes,
            input_stride, output_nPlanes, output_stride, o, additiveGrad);
      if (nActive > o)
        dAffineReluTrivialConvolution_backward_dW_B<
            T, K, V><<<dim3(1, input_nPlanes / K), dim3(K, K / V), 0,
                       THCState_getCurrentStream(state)>>>(
            inFeatures + o * input_stride, dInFeatures + o * input_stride,
            dOutFeatures + o * output_stride, affineWeight, dAffineWeight,
            affineBias, dAffineBias, convWeight, dConvWeight, input_nPlanes,
            input_stride, output_nPlanes, output_stride, nActive - o,
            additiveGrad);
      return;
    }
  }
  {
    const uInt K = 16;
    const uInt V = 4;
    if (input_nPlanes % K == 0 and output_nPlanes % K == 0) {
      uInt o = (nActive / K) * K;
      if (o > 0)
        dAffineReluTrivialConvolution_backward_dW_A<
            T, K, V><<<dim3(std::min(o / K, (uInt)512), input_nPlanes / K),
                       dim3(K, K / V), 0, THCState_getCurrentStream(state)>>>(
            inFeatures, dInFeatures, dOutFeatures, affineWeight, dAffineWeight,
            affineBias, dAffineBias, convWeight, dConvWeight, input_nPlanes,
            input_stride, output_nPlanes, output_stride, o, additiveGrad);
      if (nActive > o)
        dAffineReluTrivialConvolution_backward_dW_B<
            T, K, V><<<dim3(1, input_nPlanes / K), dim3(K, K / V), 0,
                       THCState_getCurrentStream(state)>>>(
            inFeatures + o * input_stride, dInFeatures + o * input_stride,
            dOutFeatures + o * output_stride, affineWeight, dAffineWeight,
            affineBias, dAffineBias, convWeight, dConvWeight, input_nPlanes,
            input_stride, output_nPlanes, output_stride, nActive - o,
            additiveGrad);
      return;
    }
  }
  {
    const uInt K = 8;
    const uInt V = 2;
    if (input_nPlanes % K == 0 and output_nPlanes % K == 0) {
      uInt o = (nActive / K) * K;
      if (o > 0)
        dAffineReluTrivialConvolution_backward_dW_A<
            T, K, V><<<dim3(std::min(o / K, (uInt)512), input_nPlanes / K),
                       dim3(K, K / V), 0, THCState_getCurrentStream(state)>>>(
            inFeatures, dInFeatures, dOutFeatures, affineWeight, dAffineWeight,
            affineBias, dAffineBias, convWeight, dConvWeight, input_nPlanes,
            input_stride, output_nPlanes, output_stride, o, additiveGrad);
      if (nActive > o)
        dAffineReluTrivialConvolution_backward_dW_B<
            T, K, V><<<dim3(1, input_nPlanes / K), dim3(K, K / V), 0,
                       THCState_getCurrentStream(state)>>>(
            inFeatures + o * input_stride, dInFeatures + o * input_stride,
            dOutFeatures + o * output_stride, affineWeight, dAffineWeight,
            affineBias, dAffineBias, convWeight, dConvWeight, input_nPlanes,
            input_stride, output_nPlanes, output_stride, nActive - o,
            additiveGrad);
      return;
    }
  }
}

#endif
