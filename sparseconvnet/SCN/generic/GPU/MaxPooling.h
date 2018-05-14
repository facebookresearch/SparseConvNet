// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef GPU_MAXPOOLING_H
#define GPU_MAXPOOLING_H

// NTX must be >=2 so r is filled properly
template <typename T, uInt NTX, uInt NTY>
__global__ void MaxPooling_fp(T *input_features, T *output_features,
                              uInt nPlanes, uInt input_stride,
                              uInt output_stride, uInt *rules, uInt nHot) {
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
      for (uInt plane = threadIdx.x; plane < nPlanes; plane += NTX) {
        T inp = input_features[i + plane];
        if (output_features[o + plane] < inp)
          output_features[o + plane] = inp;
      }
    }
    __syncthreads();
  }
}

template <typename T>
void MaxPooling_ForwardPass(cudaStream_t stream, T *input_features,
                            T *output_features, uInt nPlanes, uInt input_stride,
                            uInt output_stride, uInt *rules, uInt nHot) {
  MaxPooling_fp<T, 32, 32> << <32, dim3(32, 32), 0, stream>>>
      (input_features, output_features, nPlanes, input_stride, output_stride,
       rules, nHot);
}
template <typename T, uInt NTX, uInt NTY>
__global__ void MaxPooling_bp(T *input_features, T *d_input_features,
                              T *output_features, T *d_output_features,
                              uInt nPlanes, uInt input_stride,
                              uInt output_stride, uInt *rules, uInt nHot) {
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
        if (output_features[o + plane] == input_features[i + plane])
          d_input_features[i + plane] += d_output_features[o + plane];
    }
    __syncthreads();
  }
}

template <typename T>
void MaxPooling_BackwardPass(cudaStream_t stream, T *input_features,
                             T *d_input_features, T *output_features,
                             T *d_output_features, uInt nPlanes,
                             uInt input_stride, uInt output_stride, uInt *rules,
                             uInt nHot) {
  MaxPooling_bp<T, 32, 32> << <32, dim3(32, 32), 0, stream>>>
      (input_features, d_input_features, output_features, d_output_features,
       nPlanes, input_stride, output_stride, rules, nHot);
}
#endif /* GPU_MAXPOOLING_H */
