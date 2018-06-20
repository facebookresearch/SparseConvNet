// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "BatchwiseMultiplicativeDropout.h"

#define SPARSECONVNET_FOO(NTX, NTY)                                            \
  {                                                                            \
    if (nPlanes % NTX == 0) {                                                  \
      BatchwiseMultiplicativeDropout_fp<                                       \
          T, NTX,                                                              \
          NTY><<<dim3(std::min(16L, nPlanes / NTX), 16), dim3(NTX, NTY)>>>(    \
          input_features.data<T>(), output_features.data<T>(),                 \
          noise.data<T>(), nActive, nPlanes, nPlanes, nPlanes, alpha);         \
      return;                                                                  \
    }                                                                          \
  }

template <typename T>
void cuda_BatchwiseMultiplicativeDropout_updateOutput(
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor output_features, /*cuda float*/ at::Tensor noise,
    float alpha) {
  output_features.resize_as_(input_features);
  auto nActive = input_features.size(0);
  auto nPlanes = input_features.size(1);
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
      BatchwiseMultiplicativeDropout_bp<                                       \
          T, NTX,                                                              \
          NTY><<<dim3(std::min(16L, nPlanes / NTX), 16), dim3(NTX, NTY)>>>(    \
          input_features.data<T>(), d_input_features.data<T>(),                \
          d_output_features.data<T>(), noise.data<T>(), nActive, nPlanes,      \
          nPlanes, nPlanes, alpha);                                            \
      return;                                                                  \
    }                                                                          \
  }
template <typename T>
void cuda_BatchwiseMultiplicativeDropout_updateGradInput(
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor d_input_features,
    /*cuda float*/ at::Tensor d_output_features,
    /*cuda float*/ at::Tensor noise, float alpha) {
  d_input_features.resize_as_(d_output_features);
  auto nActive = input_features.size(0);
  auto nPlanes = input_features.size(1);

  SPARSECONVNET_FOO(32, 32)
  SPARSECONVNET_FOO(24, 32)
  SPARSECONVNET_FOO(16, 64)
  SPARSECONVNET_FOO(12, 64)
  SPARSECONVNET_FOO(8, 64)
  SPARSECONVNET_FOO(4, 64)
  SPARSECONVNET_FOO(1, 64)
}
#undef SPARSECONVNET_FOO
