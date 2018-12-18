// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

template <typename T, Int NTX, Int NTY>
__global__ void BatchwiseMultiplicativeDropout_fp(T *input_features,
                                                  T *output_features, T *noise,
                                                  Int nActive, Int nPlanes,
                                                  Int input_stride,
                                                  Int output_stride, T alpha) {
  __shared__ T nz[NTX];
  for (Int plane = threadIdx.x + blockIdx.x * NTX; plane < nPlanes;
       plane += gridDim.x * NTX) {
    if (threadIdx.y == 0)
      nz[threadIdx.x] = noise[plane];
    __syncthreads();
    for (Int row = threadIdx.y + blockIdx.y * NTY; row < nActive;
         row += gridDim.y * NTY) {
      Int i = row * input_stride + plane;
      Int o = row * output_stride + plane;
      output_features[o] = input_features[i] * nz[threadIdx.x] *
                           ((input_features[i] > 0) ? 1 : alpha);
    }
    __syncthreads();
  }
}
template <typename T, Int NTX, Int NTY>
__global__ void BatchwiseMultiplicativeDropout_bp(
    T *input_features, T *d_input_features, T *d_output_features, T *noise,
    Int nActive, Int nPlanes, Int input_stride, Int output_stride, T alpha) {
  __shared__ T nz[NTX];
  for (Int plane = threadIdx.x + blockIdx.x * NTX; plane < nPlanes;
       plane += gridDim.x * NTX) {
    if (threadIdx.y == 0)
      nz[threadIdx.x] = noise[plane];
    __syncthreads();
    for (Int row = threadIdx.y + blockIdx.y * NTY; row < nActive;
         row += gridDim.y * NTY) {
      Int i = row * input_stride + plane;
      Int o = row * output_stride + plane;
      d_input_features[i] = d_output_features[o] * nz[threadIdx.x] *
                            ((input_features[i] > 0) ? 1 : alpha);
    }
    __syncthreads();
  }
}

#define SPARSECONVNET_FOO(NTX, NTY)                                            \
  {                                                                            \
    if (nPlanes % NTX == 0) {                                                  \
      BatchwiseMultiplicativeDropout_fp<T, NTX, NTY><<<                        \
          dim3(std::min((Int)16, nPlanes / NTX), 16), dim3(NTX, NTY)>>>(       \
          input_features, output_features, noise, nActive, nPlanes, nPlanes,   \
          nPlanes, alpha);                                                     \
      return;                                                                  \
    }                                                                          \
  }

template <typename T>
void bmd_f(T *input_features, T *output_features, T *noise, Int nActive,
           Int nPlanes, T alpha) {
  SPARSECONVNET_FOO(32, 32)
  SPARSECONVNET_FOO(24, 32)
  SPARSECONVNET_FOO(16, 64)
  SPARSECONVNET_FOO(12, 64)
  SPARSECONVNET_FOO(8, 64)
  SPARSECONVNET_FOO(4, 64)
  SPARSECONVNET_FOO(1, 64)
}
#undef SPARSECONVNET_FOO

#define SPARSECONVNET_FOO(NTX, NTY)                                            \
  {                                                                            \
    if (nPlanes % NTX == 0) {                                                  \
      BatchwiseMultiplicativeDropout_bp<T, NTX, NTY><<<                        \
          dim3(std::min((Int)16, nPlanes / NTX), 16), dim3(NTX, NTY)>>>(       \
          input_features, d_input_features, d_output_features, noise, nActive, \
          nPlanes, nPlanes, nPlanes, alpha);                                   \
      return;                                                                  \
    }                                                                          \
  }

template <typename T>
void bmd_b(T *input_features, T *d_input_features, T *d_output_features,
           T *noise, Int nActive, Int nPlanes, T alpha) {
  SPARSECONVNET_FOO(32, 32)
  SPARSECONVNET_FOO(24, 32)
  SPARSECONVNET_FOO(16, 64)
  SPARSECONVNET_FOO(12, 64)
  SPARSECONVNET_FOO(8, 64)
  SPARSECONVNET_FOO(4, 64)
  SPARSECONVNET_FOO(1, 64)
}

#undef SPARSECONVNET_FOO
