// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef CUDA_BATCHWISEMULTIPLICATIVEDROPOUT_H
#define CUDA_BATCHWISEMULTIPLICATIVEDROPOUT_H
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
__global__ void
BatchwiseMultiplicativeDropout_bp(T *input_features, T *d_input_features,
                                  T *d_output_features, T *noise, Int nActive,
                                  Int nPlanes, Int input_stride,
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
      d_input_features[i] = d_output_features[o] * nz[threadIdx.x] *
                            ((input_features[i] > 0) ? 1 : alpha);
    }
    __syncthreads();
  }
}
#endif /* CUDA_BATCHWISEMULTIPLICATIVEDROPOUT_H */
