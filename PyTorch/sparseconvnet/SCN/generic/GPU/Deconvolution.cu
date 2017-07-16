// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef TH_GENERIC_FILE_
#define TH_GENERIC_FILE_ "generic/GPU/Deconvolution.cu"
#else
#include "Convolution.h"
#include "Deconvolution.h"

#include <algorithm>

extern "C" double scn_DR_(Deconvolution_updateOutput)(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCTensor *input_features,
    THCTensor *output_features, THCTensor *weight, THCTensor *bias,
    long filterVolume, THCITensor *rulesBuffer) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  auto _rules =
      _m.getRuleBook(outputSize, inputSize, filterSize, filterStride, true);
  uInt nActive = _m.getNActive(outputSize);
  THCTensor_(resize2d)(state, output_features, nActive, weight->size[1]);
  if (not bias)
    THCTensor_(zero)(state, output_features);

  auto iF = THCTensor_(data)(state, input_features);
  auto oF = THCTensor_(data)(state, output_features);
  auto ip = input_features->size[1];
  auto op = output_features->size[1];
  auto w = THCTensor_(data)(state, weight);
  double flops = 0;

  if (bias) {
    auto b = THCTensor_(data)(state, bias);
    for (uInt i = 0; i < op; i += 32) {
      uInt blockDim = min(32L, op - i);
      uInt gridDim = min(4096, nActive);
      Convolution_fp_bias
              << <gridDim, blockDim, 0, THCState_getCurrentStream(state)>>>
          (oF + i, b + i, op, op, nActive);
    }
  }
  uInt c = ip * op;
  RULEBOOKITERATOR(
      dDeconvolution_forward2<real>(iF, oF, w, rbB, nHotB, ip, ip, op, op,
                                    THCState_getCurrentStream(state));
      , w += c; flops += nHotB * c;)
  return flops;
}

extern "C" void scn_DR_(Deconvolution_backward)(
    THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize,
    THLongTensor *filterStride, void **m, THCTensor *input_features,
    THCTensor *d_input_features, THCTensor *d_output_features,
    THCTensor *weight, THCTensor *d_weight, THCTensor *d_bias,
    long filterVolume, THCITensor *rulesBuffer) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  auto _rules =
      _m.getRuleBook(outputSize, inputSize, filterSize, filterStride, true);
  uInt nActive = _m.getNActive(outputSize);
  THCTensor_(resizeAs)(state, d_input_features, input_features);
  THCTensor_(zero)(state, d_input_features);

  auto iF = THCTensor_(data)(state, input_features);
  auto diF = THCTensor_(data)(state, d_input_features);
  auto doF = THCTensor_(data)(state, d_output_features);
  auto ip = input_features->size[1];
  auto op = d_output_features->size[1];
  auto w = THCTensor_(data)(state, weight);
  auto dw = THCTensor_(data)(state, d_weight);
  uInt c = ip * op;
  RULEBOOKITERATOR(dDeconvolution_backward_dW2<real>(
                       iF, diF, doF, w, dw, rbB, nHotB, ip, ip, op, op,
                       THCState_getCurrentStream(state));

                   , w += c; dw += c;)

  if (d_bias) {
    auto db = THCTensor_(data)(state, d_bias);
    Convolution_bp_bias(doF, db, op, op, nActive,
                        THCState_getCurrentStream(state));
  }
}

#endif
