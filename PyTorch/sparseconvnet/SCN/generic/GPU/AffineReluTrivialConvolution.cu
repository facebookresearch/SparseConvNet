// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/GPU/AffineReluTrivialConvolution.cu"
#else
#include "AffineReluTrivialConvolution.h"

#include <algorithm>
#include <iostream>

extern "C" void scn_R_(AffineReluTrivialConvolution_updateOutput)(
    THCTensor *input_features, THCTensor *output_features,
    THCTensor *affineWeight, THCTensor *affineBias, THCTensor *convWeight) {

  THCTensor_(resize2d)(state, output_features, input_features->size[0],
                       convWeight->size[1]);
  dAffineReluTrivialConvolution_forward<real>(
      THCTensor_(data)(state, input_features),
      THCTensor_(data)(state, output_features),
      THCTensor_(data)(state, affineWeight),
      THCTensor_(data)(state, affineBias), THCTensor_(data)(state, convWeight),
      convWeight->size[0], input_features->stride[0], convWeight->size[1],
      output_features->size[1], input_features->size[0]);
}

extern "C" void scn_R_(AffineReluTrivialConvolution_backward)(
    THCTensor *input_features, THCTensor *d_input_features,
    THCTensor *d_output_features, THCTensor *affineWeight,
    THCTensor *d_affineWeight, THCTensor *affineBias, THCTensor *d_affineBias,
    THCTensor *convWeight, THCTensor *d_convWeight, bool additiveGrad) {

  THCTensor_(resizeAs)(state, d_input_features, input_features);
  dAffineReluTrivialConvolution_backward_dW<real>(
      THCTensor_(data)(state, input_features),
      THCTensor_(data)(state, d_input_features),
      THCTensor_(data)(state, d_output_features),
      THCTensor_(data)(state, affineWeight),
      THCTensor_(data)(state, d_affineWeight),
      THCTensor_(data)(state, affineBias),
      THCTensor_(data)(state, d_affineBias),
      THCTensor_(data)(state, convWeight),
      THCTensor_(data)(state, d_convWeight), convWeight->size[0],
      input_features->stride[0], convWeight->size[1],
      d_output_features->stride[0], input_features->size[0], additiveGrad);
}

#endif
