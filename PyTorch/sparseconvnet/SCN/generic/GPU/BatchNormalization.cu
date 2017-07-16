// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/GPU/BatchNormalization.cu"
#else
#include "BatchNormalization.h"

#define BN_F_MACRO(N)                                                          \
  if (nPlanes % N == 0) {                                                      \
    BatchNormalization_ForwardPass<real, N, 64>(                               \
        THCTensor_(data)(state, input_features),                               \
        THCTensor_(data)(state, output_features), nPlanes, input_stride,       \
        output_stride, nActive, THCTensor_(data)(state, saveMean),             \
        THCTensor_(data)(state, saveInvStd),                                   \
        THCTensor_(data)(state, runningMean),                                  \
        THCTensor_(data)(state, runningVar),                                   \
        weight ? THCTensor_(data)(state, weight) : 0,                          \
        bias ? THCTensor_(data)(state, bias) : 0, eps, momentum, train,        \
        leakiness);                                                            \
  }

extern "C" void scn_R_(BatchNormalization_updateOutput)(
    THCTensor *input_features, THCTensor *output_features, THCTensor *saveMean,
    THCTensor *saveInvStd, THCTensor *runningMean, THCTensor *runningVar,
    THCTensor *weight, THCTensor *bias, real eps, real momentum, bool train,
    real leakiness) {

  THCTensor_(resizeAs)(state, output_features, input_features);
  auto nActive = input_features->size[0];
  auto nPlanes = input_features->size[1];
  auto input_stride = input_features->stride[0];
  auto output_stride = output_features->stride[0];

  BN_F_MACRO(16)
  else BN_F_MACRO(12) else BN_F_MACRO(8) else BN_F_MACRO(4) else BN_F_MACRO(1)
}

extern "C" void scn_R_(BatchNormalizationInTensor_updateOutput)(
    THCTensor *input_features, THCTensor *output_features, THCTensor *saveMean,
    THCTensor *saveInvStd, THCTensor *runningMean, THCTensor *runningVar,
    THCTensor *weight, THCTensor *bias, real eps, real momentum, bool train,
    real leakiness) {

  auto nActive = input_features->size[0];
  auto nPlanes = input_features->size[1];
  auto input_stride = input_features->stride[0];
  auto output_stride = output_features->stride[0];

  BN_F_MACRO(16)
  else BN_F_MACRO(12) else BN_F_MACRO(8) else BN_F_MACRO(4) else BN_F_MACRO(1)
}

#undef BN_F_MACRO

#define BN_B_MACRO(N)                                                          \
  if (nPlanes % N == 0) {                                                      \
    BatchNormalization_BackwardPass<real, N, 64>(                              \
        THCTensor_(data)(state, input_features),                               \
        THCTensor_(data)(state, d_input_features),                             \
        THCTensor_(data)(state, output_features),                              \
        THCTensor_(data)(state, d_output_features), nPlanes, input_stride,     \
        output_stride, nActive, THCTensor_(data)(state, saveMean),             \
        THCTensor_(data)(state, saveInvStd),                                   \
        THCTensor_(data)(state, runningMean),                                  \
        THCTensor_(data)(state, runningVar),                                   \
        weight ? THCTensor_(data)(state, weight) : 0,                          \
        bias ? THCTensor_(data)(state, bias) : 0,                              \
        d_weight ? THCTensor_(data)(state, d_weight) : 0,                      \
        d_bias ? THCTensor_(data)(state, d_bias) : 0, leakiness);              \
  }

extern "C" void scn_R_(BatchNormalization_backward)(
    THCTensor *input_features, THCTensor *d_input_features,
    THCTensor *output_features, THCTensor *d_output_features,
    THCTensor *saveMean, THCTensor *saveInvStd, THCTensor *runningMean,
    THCTensor *runningVar, THCTensor *weight, THCTensor *bias,
    THCTensor *d_weight, THCTensor *d_bias, real leakiness) {

  THCTensor_(resizeAs)(state, d_input_features, d_output_features);
  auto nActive = input_features->size[0];
  auto nPlanes = input_features->size[1];
  auto input_stride = input_features->stride[0];
  auto output_stride = output_features->stride[0];

  BN_B_MACRO(16)
  else BN_B_MACRO(12) else BN_B_MACRO(8) else BN_B_MACRO(4) else BN_B_MACRO(1)
}
#endif
