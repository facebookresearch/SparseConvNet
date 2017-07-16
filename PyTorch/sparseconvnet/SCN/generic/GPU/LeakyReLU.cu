// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/GPU/LeakyReLU.cu"
#else
#include "LeakyReLU.h"

extern "C" void scn_R_(LeakyReLU_updateOutput)(THCTensor *input_features,
                                               THCTensor *output_features,
                                               float alpha) {
  if (input_features != output_features)
    THCTensor_(resizeAs)(state, output_features, input_features);
  auto n = THCTensor_(nElement)(state, input_features);
  LeakyReLU_fp<real> << <16, 1024, 0, THCState_getCurrentStream(state)>>>
      (THCTensor_(data)(state, input_features),
       THCTensor_(data)(state, output_features), n, alpha);
}

extern "C" void scn_R_(LeakyReLU_updateGradInput)(THCTensor *input_features,
                                                  THCTensor *d_input_features,
                                                  THCTensor *d_output_features,
                                                  float alpha) {
  if (d_input_features != d_output_features)
    THCTensor_(resizeAs)(state, d_input_features, d_output_features);
  auto n = THCTensor_(nElement)(state, d_input_features);
  LeakyReLU_bp<real> << <16, 1024, 0, THCState_getCurrentStream(state)>>>
      (THCTensor_(data)(state, input_features),
       THCTensor_(data)(state, d_input_features),
       THCTensor_(data)(state, d_output_features), n, alpha);
}

#endif
