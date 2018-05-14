// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/GPU/NetworkInNetwork.cu"
#else
#include "Convolution.h"

#include <algorithm>

extern "C" double
scn_R_(NetworkInNetwork_updateOutput)(THCTensor *input_features_,
                                      THCTensor *output_features_,
                                      THCTensor *weight_, THCTensor *bias_) {
  auto nActive = input_features_->size[0];
  auto input_nPlanes = weight_->size[0];
  auto output_nPlanes = weight_->size[1];
  THCTensor_(resize2d)(state, output_features_, nActive, output_nPlanes);
  auto input_features = THCTensor_(data)(state, input_features_);
  auto output_features = THCTensor_(data)(state, output_features_);
  auto weight = THCTensor_(data)(state, weight_);

  if (bias_ != nullptr) {
    auto bias = THCTensor_(data)(state, bias_);
    for (uInt i = 0; i < output_nPlanes; i += 32) {
      uInt blockDim = min(32L, output_nPlanes - i);
      uInt gridDim = min(4096L, nActive);
      Convolution_fp_bias<<<gridDim, blockDim, 0,
                            THCState_getCurrentStream(state)>>>(
          output_features + i, bias + i, output_nPlanes, output_nPlanes,
          nActive);
    }
    // Do GEMM (note: gemm assumes column-major matrices)
    // buffer          is l*m (row-major)
    // weight          is m*r (row-major)
    // output_features is l*r (row-major)
    // buffer * weights + bias -> output_features
    THBLAS_GEMM(state, 'n', 'n',
                output_nPlanes, // r
                nActive,        // l
                input_nPlanes,  // m
                1,              // alpha
                weight,
                output_nPlanes, // r
                input_features,
                input_nPlanes, // m
                1,             // beta
                output_features,
                output_nPlanes // r
                );
  } else {
    THCTensor_(zero)(state, output_features_);
    THBLAS_GEMM(state, 'n', 'n',
                output_nPlanes, // r
                nActive,        // l
                input_nPlanes,  // m
                1,              // alpha
                weight,
                output_nPlanes, // r
                input_features,
                input_nPlanes, // m
                0,             // beta
                output_features,
                output_nPlanes // r
                );
  }
  return nActive * input_nPlanes * output_nPlanes;
}

extern "C" void
scn_R_(NetworkInNetwork_updateGradInput)(THCTensor *d_input_features_,
                                         THCTensor *d_output_features_,
                                         THCTensor *weight_) {
  auto nActive = d_output_features_->size[0];
  auto input_nPlanes = weight_->size[0];
  auto output_nPlanes = weight_->size[1];
  THCTensor_(resize2d)(state, d_input_features_, nActive, input_nPlanes);
  THCTensor_(zero)(state, d_input_features_);
  auto d_input_features = THCTensor_(data)(state, d_input_features_);
  auto d_output_features = THCTensor_(data)(state, d_output_features_);
  auto weight = THCTensor_(data)(state, weight_);
  // Do GEMM (note: gemm assumes column-major matrices)
  // d_output_features is l*m (row-major)
  // weights           is r*m (row-major)
  // d_buffer          is l*r (row-major)
  // d_output_features * T(weight) -> d_buffer
  THBLAS_GEMM(state, 't', 'n',
              input_nPlanes,  // r
              nActive,        // l
              output_nPlanes, // m
              1,              // alpha
              weight,
              output_nPlanes, // m
              d_output_features,
              output_nPlanes, // m
              0,              // beta
              d_input_features,
              input_nPlanes // r
              );
}

extern "C" void scn_R_(NetworkInNetwork_accGradParameters)(
    THCTensor *input_features_, THCTensor *d_output_features_,
    THCTensor *d_weight_, THCTensor *d_bias_) {
  auto nActive = input_features_->size[0];
  auto input_nPlanes = d_weight_->size[0];
  auto output_nPlanes = d_weight_->size[1];
  auto input_features = THCTensor_(data)(state, input_features_);
  auto d_output_features = THCTensor_(data)(state, d_output_features_);
  auto d_weight = THCTensor_(data)(state, d_weight_);
  // Do GEMM (note: gemm assumes column-major matrices)
  // buffer            is m*l (row-major)
  // d_output_features is m*r (row-major)
  // weights           is l*r (row-major)
  // T(buffer) * d_output_features -> d_weight
  THBLAS_GEMM(state, 'n', 't',
              output_nPlanes, // r
              input_nPlanes,  // l
              nActive,        // m
              1,              // alpha
              d_output_features,
              output_nPlanes, // r
              input_features,
              input_nPlanes, // l
              1,             // beta
              d_weight,
              output_nPlanes // r
              );

  if (d_bias_) {
    auto d_bias = THCTensor_(data)(state, d_bias_);
    Convolution_bp_bias(d_output_features, d_bias, output_nPlanes,
                        output_nPlanes, nActive,
                        THCState_getCurrentStream(state));
  }
}

#endif
