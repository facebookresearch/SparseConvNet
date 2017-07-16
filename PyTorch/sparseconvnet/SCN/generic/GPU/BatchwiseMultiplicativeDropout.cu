// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/GPU/BatchwiseMultiplicativeDropout.cu"
#else
#include "BatchwiseMultiplicativeDropout.h"

#define SPARSECONVNET_FOO(NTX, NTY)                                            \
  {                                                                            \
    if (nPlanes % NTX == 0) {                                                  \
      BatchwiseMultiplicativeDropout_fp<real, NTX, NTY> << <                   \
          dim3(std::min(16L, nPlanes / NTX), 16), dim3(NTX, NTY), 0,           \
          THCState_getCurrentStream(state)>>>                                  \
          (THCTensor_(data)(state, input_features),                            \
           THCTensor_(data)(state, output_features),                           \
           THCTensor_(data)(state, noise), nActive, nPlanes, nPlanes, nPlanes, \
           alpha);                                                             \
      return;                                                                  \
    }                                                                          \
  }

extern "C" void scn_R_(BatchwiseMultiplicativeDropout_updateOutput)(
    THCTensor *input_features, THCTensor *output_features, THCTensor *noise,
    float alpha) {
  if (input_features != output_features)
    THCTensor_(resizeAs)(state, output_features, input_features);
  auto nActive = input_features->size[0];
  auto nPlanes = input_features->size[1];
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
      BatchwiseMultiplicativeDropout_bp<real, NTX, NTY> << <                   \
          dim3(std::min(16L, nPlanes / NTX), 16), dim3(NTX, NTY), 0,           \
          THCState_getCurrentStream(state)>>>                                  \
          (THCTensor_(data)(state, input_features),                            \
           THCTensor_(data)(state, d_input_features),                          \
           THCTensor_(data)(state, d_output_features),                         \
           THCTensor_(data)(state, noise), nActive, nPlanes, nPlanes, nPlanes, \
           alpha);                                                             \
      return;                                                                  \
    }                                                                          \
  }
extern "C" void scn_R_(BatchwiseMultiplicativeDropout_updateGradInput)(
    THCTensor *input_features, THCTensor *d_input_features,
    THCTensor *d_output_features, THCTensor *noise, float alpha) {
  if (d_input_features != d_output_features)
    THCTensor_(resizeAs)(state, d_input_features, d_output_features);
  auto nActive = input_features->size[0];
  auto nPlanes = input_features->size[1];

  SPARSECONVNET_FOO(32, 32)
  SPARSECONVNET_FOO(24, 32)
  SPARSECONVNET_FOO(16, 64)
  SPARSECONVNET_FOO(12, 64)
  SPARSECONVNET_FOO(8, 64)
  SPARSECONVNET_FOO(4, 64)
  SPARSECONVNET_FOO(1, 64)
}
#undef SPARSECONVNET_FOO

#endif
