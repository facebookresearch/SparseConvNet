// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef TH_GENERIC_FILE_
#define TH_GENERIC_FILE_ "generic/CPU/Convolution.cpp"
#else
#include "Convolution.h"

extern "C" double scn_DR_(Convolution_updateOutput)(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THTensor *input_features,
    THTensor *output_features, THTensor *weight, THTensor *bias,
    long filterVolume, void *rulesBuffer) {

  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  auto _rules =
      _m.getRuleBook(inputSize, outputSize, filterSize, filterStride, true);
  uInt nActive = _m.getNActive(outputSize);
  THTensor_(resize2d)(output_features, nActive, weight->size[1]);
  if (not bias)
    THTensor_(zero)(output_features);

  auto iF = THTensor_(data)(input_features);
  auto oF = THTensor_(data)(output_features);
  auto ip = input_features->size[1];
  auto op = output_features->size[1];
  auto w = THTensor_(data)(weight);
  auto b = THOptionalTensorData(bias);
  Convolution_ForwardPass(iF, ip, ip, oF, op, op, w, b, _rules, nActive,
                          THBlas_(gemm));
  double flops = 0;
  for (auto &r : _rules)
    flops += r.size() / 2 * ip * op;
  return flops;
}

extern "C" void scn_DR_(Convolution_backward)(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THTensor *input_features,
    THTensor *d_input_features, THTensor *d_output_features, THTensor *weight,
    THTensor *d_weight, THTensor *d_bias, long filterVolume,
    void *rulesBuffer) {

  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  auto _rules =
      _m.getRuleBook(inputSize, outputSize, filterSize, filterStride, true);
  uInt nActive = _m.getNActive(outputSize);
  THTensor_(resizeAs)(d_input_features, input_features);
  THTensor_(zero)(d_input_features);

  auto iF = THTensor_(data)(input_features);
  auto diF = THTensor_(data)(d_input_features);
  auto doF = THTensor_(data)(d_output_features);
  auto ip = input_features->size[1];
  auto op = d_output_features->size[1];
  auto w = THTensor_(data)(weight);
  auto dw = THTensor_(data)(d_weight);
  auto db = THOptionalTensorData(d_bias);

  Convolution_BackwardPass(iF, diF, ip, ip, doF, op, op, w, dw, db, _rules,
                           nActive, THBlas_(gemm));
}

extern "C" double scn_DR_(ValidConvolution_updateOutput)(
    THLongTensor *inputSize, THLongTensor *filterSize, void **m,
    THTensor *input_features, THTensor *output_features, THTensor *weight,
    THTensor *bias, long filterVolume, void *rulesBuffer) {

  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  auto _rules = _m.getValidRuleBook(inputSize, filterSize, true);
  uInt nActive = input_features->size[0];
  THTensor_(resize2d)(output_features, nActive, weight->size[1]);
  if (not bias)
    THTensor_(zero)(output_features);

  auto iF = THTensor_(data)(input_features);
  auto oF = THTensor_(data)(output_features);
  auto ip = input_features->size[1];
  auto op = output_features->size[1];
  auto w = THTensor_(data)(weight);
  auto b = THOptionalTensorData(bias);

  Convolution_ForwardPass(iF, ip, ip, oF, op, op, w, b, _rules, nActive,
                          THBlas_(gemm));
  double flops = 0;
  for (auto &r : _rules)
    flops += r.size() / 2 * ip * op;

  return flops;
}

extern "C" void scn_DR_(ValidConvolution_backward)(
    THLongTensor *inputSize, THLongTensor *filterSize, void **m,
    THTensor *input_features, THTensor *d_input_features,
    THTensor *d_output_features, THTensor *weight, THTensor *d_weight,
    THTensor *d_bias, long filterVolume, void *rulesBuffer) {

  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  auto _rules = _m.getValidRuleBook(inputSize, filterSize, true);
  uInt nActive = input_features->size[0];
  THTensor_(resizeAs)(d_input_features, input_features);
  THTensor_(zero)(d_input_features);

  auto iF = THTensor_(data)(input_features);
  auto diF = THTensor_(data)(d_input_features);
  auto doF = THTensor_(data)(d_output_features);
  auto ip = input_features->size[1];
  auto op = d_output_features->size[1];
  auto w = THTensor_(data)(weight);
  auto dw = THTensor_(data)(d_weight);
  auto db = THOptionalTensorData(d_bias);

  Convolution_BackwardPass(iF, diF, ip, ip, doF, op, op, w, dw, db, _rules,
                           nActive, THBlas_(gemm));
}

#endif
