// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/CPU/BatchNormalization.cpp"
#else
#include "BatchNormalization.h"

extern "C" void scn_R_(BatchNormalization_updateOutput)(
    THTensor *input_features, THTensor *output_features, THTensor *saveMean,
    THTensor *saveInvStd, THTensor *runningMean, THTensor *runningVar,
    THTensor *weight, THTensor *bias, real eps, real momentum, bool train,
    real leakiness) {

  THTensor_(resizeAs)(output_features, input_features);
  auto nActive = input_features->size[0];
  auto nPlanes = input_features->size[1];
  auto input_stride = input_features->stride[0];
  auto output_stride = output_features->stride[0];
  BatchNormalization_ForwardPass<real>(
      THTensor_(data)(input_features), THTensor_(data)(output_features),
      nPlanes, input_stride, output_stride, nActive, THTensor_(data)(saveMean),
      THTensor_(data)(saveInvStd), THTensor_(data)(runningMean),
      THTensor_(data)(runningVar), THOptionalTensorData(weight),
      THOptionalTensorData(bias), eps, momentum, train, leakiness);
}

extern "C" void scn_R_(BatchNormalizationInTensor_updateOutput)(
    THTensor *input_features, THTensor *output_features, THTensor *saveMean,
    THTensor *saveInvStd, THTensor *runningMean, THTensor *runningVar,
    THTensor *weight, THTensor *bias, real eps, real momentum, bool train,
    real leakiness) {

  auto nActive = input_features->size[0];
  auto nPlanes = input_features->size[1];
  auto input_stride = input_features->stride[0];
  auto output_stride = output_features->stride[0];

  BatchNormalization_ForwardPass<real>(
      THTensor_(data)(input_features), THTensor_(data)(output_features),
      nPlanes, input_stride, output_stride, nActive, THTensor_(data)(saveMean),
      THTensor_(data)(saveInvStd), THTensor_(data)(runningMean),
      THTensor_(data)(runningVar), THOptionalTensorData(weight),
      THOptionalTensorData(bias), eps, momentum, train, leakiness);
}

extern "C" void scn_R_(BatchNormalization_backward)(
    THTensor *input_features, THTensor *d_input_features,
    THTensor *output_features, THTensor *d_output_features, THTensor *saveMean,
    THTensor *saveInvStd, THTensor *runningMean, THTensor *runningVar,
    THTensor *weight, THTensor *bias, THTensor *d_weight, THTensor *d_bias,
    real leakiness) {

  THTensor_(resizeAs)(d_input_features, input_features);
  auto nActive = input_features->size[0];
  auto nPlanes = input_features->size[1];
  auto input_stride = input_features->stride[0];
  auto output_stride = output_features->stride[0];
  BatchNormalization_BackwardPass<real>(
      THTensor_(data)(input_features), THTensor_(data)(d_input_features),
      THTensor_(data)(output_features), THTensor_(data)(d_output_features),
      nPlanes, input_stride, output_stride, nActive, THTensor_(data)(saveMean),
      THTensor_(data)(saveInvStd), THTensor_(data)(runningMean),
      THTensor_(data)(runningVar), THOptionalTensorData(weight),
      THOptionalTensorData(bias), THOptionalTensorData(d_weight),
      THOptionalTensorData(d_bias), leakiness);
}
#endif
