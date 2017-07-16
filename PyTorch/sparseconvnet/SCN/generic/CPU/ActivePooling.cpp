// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef TH_GENERIC_FILE_
#define TH_GENERIC_FILE_ "generic/CPU/ActivePooling.cpp"
#else
#include "ActivePooling.h"

extern "C" void scn_DR_(ActivePooling_updateOutput)(
    THLongTensor *inputSize, void **m, THTensor *input_features,
    THTensor *output_features, void *rulesBuffer, bool average) {

  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  uInt nPlanes = input_features->size[1];
  auto _rules = _m.getActivePoolingRuleBook(inputSize);
  uInt batchSize = _rules[1][0];
  uInt maxActive = _rules[1][1];
  THTensor_(resize2d)(output_features, batchSize, nPlanes);
  THTensor_(zero)(output_features);

  ActivePooling_ForwardPass<real>(THTensor_(data)(input_features),
                                  THTensor_(data)(output_features), batchSize,
                                  maxActive, nPlanes, _rules, average);
}
extern "C" void scn_DR_(ActivePooling_updateGradInput)(
    THLongTensor *inputSize, void **m, THTensor *input_features,
    THTensor *d_input_features, THTensor *d_output_features, void *rulesBuffer,
    bool average) {

  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  uInt nPlanes = input_features->size[1];
  auto _rules = _m.getActivePoolingRuleBook(inputSize);
  uInt batchSize = _rules[1][0];
  uInt maxActive = _rules[1][1];
  THTensor_(resizeAs)(d_input_features, input_features);
  THTensor_(zero)(d_input_features);

  ActivePooling_BackwardPass<real>(
      THTensor_(data)(d_input_features), THTensor_(data)(d_output_features),
      batchSize, maxActive, nPlanes, _rules, average);
}
#endif
