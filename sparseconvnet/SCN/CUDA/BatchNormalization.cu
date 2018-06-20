// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "BatchNormalization.h"

#define BN_F_MACRO(N)                                                          \
  if (nPlanes % N == 0) {                                                      \
    BatchNormalization_ForwardPass<T, N, 64>(                                  \
        input_features.data<T>(), output_features.data<T>(), nPlanes,          \
        input_stride, output_stride, nActive, saveMean.data<T>(),              \
        saveInvStd.data<T>(), runningMean.data<T>(), runningVar.data<T>(),     \
        OptionalTensorData<T>(weight), OptionalTensorData<T>(bias), eps, momentum,   \
        train, leakiness);                                                     \
  }

template <typename T>
void cuda_BatchNormalization_updateOutput(
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor output_features,
    /*cuda float*/ at::Tensor saveMean,
    /*cuda float*/ at::Tensor saveInvStd, /*cuda float*/ at::Tensor runningMean,
    /*cuda float*/ at::Tensor runningVar,
    /*cuda float*/ at::Tensor weight, /*cuda float*/ at::Tensor bias, T eps,
    T momentum, bool train, T leakiness) {

  output_features.resize_as_(input_features);
  if (input_features.ndimension() == 2) {
    auto nActive = input_features.size(0);
    auto nPlanes = input_features.size(1);
    auto input_stride = input_features.stride(0);
    auto output_stride = output_features.stride(0);
    BN_F_MACRO(16)
    else BN_F_MACRO(12) else BN_F_MACRO(8) else BN_F_MACRO(4) else BN_F_MACRO(1)
  }
}

template <typename T>
void cuda_BatchNormalizationInTensor_updateOutput(
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor output_features,
    /*cuda float*/ at::Tensor saveMean,
    /*cuda float*/ at::Tensor saveInvStd, /*cuda float*/ at::Tensor runningMean,
    /*cuda float*/ at::Tensor runningVar,
    /*cuda float*/ at::Tensor weight, /*cuda float*/ at::Tensor bias, T eps,
    T momentum, bool train, T leakiness) {
  if (input_features.ndimension() == 2) {
    auto nActive = input_features.size(0);
    auto nPlanes = input_features.size(1);
    auto input_stride = input_features.stride(0);
    auto output_stride = output_features.stride(0);
    BN_F_MACRO(16)
    else BN_F_MACRO(12) else BN_F_MACRO(8) else BN_F_MACRO(4) else BN_F_MACRO(1)
  }
}

#undef BN_F_MACRO

#define BN_B_MACRO(N)                                                          \
  if (nPlanes % N == 0) {                                                      \
    BatchNormalization_BackwardPass<T, N, 64>(                                 \
        input_features.data<T>(), d_input_features.data<T>(),                  \
        output_features.data<T>(), d_output_features.data<T>(), nPlanes,       \
        input_stride, output_stride, nActive, saveMean.data<T>(),              \
        saveInvStd.data<T>(), runningMean.data<T>(), runningVar.data<T>(),     \
        OptionalTensorData<T>(weight), OptionalTensorData<T>(bias),                  \
        OptionalTensorData<T>(d_weight), OptionalTensorData<T>(d_bias), leakiness);  \
  }

template <typename T>
void cuda_BatchNormalization_backward(
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor d_input_features,
    /*cuda float*/ at::Tensor output_features,
    /*cuda float*/ at::Tensor d_output_features,
    /*cuda float*/ at::Tensor saveMean, /*cuda float*/ at::Tensor saveInvStd,
    /*cuda float*/ at::Tensor runningMean,
    /*cuda float*/ at::Tensor runningVar, /*cuda float*/ at::Tensor weight,
    /*cuda float*/ at::Tensor bias,
    /*cuda float*/ at::Tensor d_weight, /*cuda float*/ at::Tensor d_bias,
    T leakiness) {

  d_input_features.resize_as_(d_output_features);
  if (input_features.ndimension() == 2) {
    auto nActive = input_features.size(0);
    auto nPlanes = input_features.size(1);
    auto input_stride = input_features.stride(0);
    auto output_stride = output_features.stride(0);
    BN_B_MACRO(16)
    else BN_B_MACRO(12) else BN_B_MACRO(8) else BN_B_MACRO(4) else BN_B_MACRO(1)
  }
}
