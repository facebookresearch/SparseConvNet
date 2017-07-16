// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef GPU_AVERAGEPOOLING_H
#define GPU_AVERAGEPOOLING_H

// NTX must be >=2 so r is filled properly
template <typename T, uInt NTX, uInt NTY>
__global__ void AveragePooling_fp(T *input_features, T *output_features,
                                  uInt nPlanes, uInt input_stride,
                                  uInt output_stride, uInt *rules, uInt nHot,
                                  T alpha) {
  __shared__ uInt r[NTY * 2];
  for (uInt n = blockIdx.x * NTY; n < nHot; n += gridDim.x * NTY) {
    {
      uInt i = threadIdx.x + NTX * threadIdx.y;
      if (i < NTY * 2 and i < 2 * (n - nHot))
        r[i] = rules[2 * n + i];
    }
    __syncthreads();
    if (n + threadIdx.y < nHot) {
      uInt i = r[2 * threadIdx.y] * input_stride;
      uInt o = r[2 * threadIdx.y + 1] * output_stride;
      for (uInt plane = threadIdx.x; plane < nPlanes; plane += NTX)
        atomicAdd(&output_features[o + plane],
                  alpha * input_features[i + plane]);
    }
    __syncthreads();
  }
}

template <typename T>
void AveragePooling_ForwardPass(cudaStream_t stream, T *input_features,
                                T *output_features, uInt nPlanes,
                                uInt input_stride, uInt output_stride,
                                uInt *rules, uInt nHot, uInt filterVolume) {
  AveragePooling_fp<T, 32, 32><<<32, dim3(32, 32), 0, stream>>>(
      input_features, output_features, nPlanes, input_stride, output_stride,
      rules, nHot, 1.0 / filterVolume);
}
template <typename T, uInt NTX, uInt NTY>
__global__ void AveragePooling_bp(T *d_input_features, T *d_output_features,
                                  uInt nPlanes, uInt input_stride,
                                  uInt output_stride, uInt *rules, uInt nHot,
                                  T alpha) {
  __shared__ uInt r[NTY * 2];
  for (uInt n = blockIdx.x * NTY; n < nHot; n += gridDim.x * NTY) {
    {
      uInt i = threadIdx.x + NTX * threadIdx.y;
      if (i < NTY * 2 and i < 2 * (n - nHot))
        r[i] = rules[2 * n + i];
    }
    __syncthreads();
    if (n + threadIdx.y < nHot) {
      uInt i = r[2 * threadIdx.y] * input_stride;
      uInt o = r[2 * threadIdx.y + 1] * output_stride;
      for (uInt plane = threadIdx.x; plane < nPlanes; plane += NTX)
        d_input_features[i + plane] += alpha * d_output_features[o + plane];
    }
    __syncthreads();
  }
}

template <typename T>
void AveragePooling_BackwardPass(cudaStream_t stream, T *d_input_features,
                                 T *d_output_features, uInt nPlanes,
                                 uInt input_stride, uInt output_stride,
                                 uInt *rules, uInt nHot, uInt filterVolume) {
  AveragePooling_bp<T, 32, 32><<<32, dim3(32, 32), 0, stream>>>(
      d_input_features, d_output_features, nPlanes, input_stride, output_stride,
      rules, nHot, 1.0 / filterVolume);
}
#endif /* GPU_AVERAGEPOOLING_H */
