// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/CPU/AffineReluTrivialConvolution.cpp"
#else
#include "AffineReluTrivialConvolution.h"

extern "C" void scn_R_(AffineReluTrivialConvolution_updateOutput)(
    THTensor *input_features, THTensor *output_features, THTensor *affineWeight,
    THTensor *affineBias, THTensor *convWeight) {
  THTensor_(resize2d)(output_features, input_features->size[0],
                      convWeight->size[1]);
  AffineReluTrivialConvolution_ForwardPass(
      THTensor_(data)(input_features), convWeight->size[0],
      input_features->stride[0], THTensor_(data)(output_features),
      convWeight->size[1], output_features->stride[0],
      THTensor_(data)(affineWeight), THTensor_(data)(affineBias),
      THTensor_(data)(convWeight), input_features->size[0]);
}

extern "C" void scn_R_(AffineReluTrivialConvolution_backward)(
    THTensor *input_features, THTensor *d_input_features,
    THTensor *d_output_features, THTensor *affineWeight,
    THTensor *d_affineWeight, THTensor *affineBias, THTensor *d_affineBias,
    THTensor *convWeight, THTensor *d_convWeight, bool additiveGrad) {

  THTensor_(resizeAs)(d_input_features, input_features);
  AffineReluTrivialConvolution_BackwardPass(
      THTensor_(data)(input_features), THTensor_(data)(d_input_features),
      convWeight->size[0], input_features->stride[0],
      THTensor_(data)(d_output_features), convWeight->size[1],
      d_output_features->stride[0], THTensor_(data)(affineWeight),
      THTensor_(data)(d_affineWeight), THTensor_(data)(affineBias),
      THTensor_(data)(d_affineBias), THTensor_(data)(convWeight),
      THTensor_(data)(d_convWeight), input_features->size[0], additiveGrad);
}

#endif
