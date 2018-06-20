// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "BatchNormalization.h"

template <typename T>
void cpu_BatchNormalization_updateOutput(
    /*float*/ at::Tensor input_features, /*float*/ at::Tensor output_features,
    /*float*/ at::Tensor saveMean,
    /*float*/ at::Tensor saveInvStd, /*float*/ at::Tensor runningMean,
    /*float*/ at::Tensor runningVar,
    /*float*/ at::Tensor weight, /*float*/ at::Tensor bias, T eps, T momentum,
    bool train, T leakiness) {
  output_features.resize_as_(input_features);
  if (input_features.ndimension() == 2) {
    auto nActive = input_features.size(0);
    auto nPlanes = input_features.size(1);
    auto input_stride = input_features.stride(0);
    auto output_stride = output_features.stride(0);
    BatchNormalization_ForwardPass<T>(
        input_features.data<T>(), output_features.data<T>(), nPlanes,
        input_stride, output_stride, nActive, saveMean.data<T>(),
        saveInvStd.data<T>(), runningMean.data<T>(), runningVar.data<T>(),
        OptionalTensorData<T>(weight), OptionalTensorData<T>(bias), eps,
        momentum, train, leakiness);
  }
}

template <typename T>
void cpu_BatchNormalizationInTensor_updateOutput(
    /*float*/ at::Tensor input_features, /*float*/ at::Tensor output_features,
    /*float*/ at::Tensor saveMean,
    /*float*/ at::Tensor saveInvStd, /*float*/ at::Tensor runningMean,
    /*float*/ at::Tensor runningVar,
    /*float*/ at::Tensor weight, /*float*/ at::Tensor bias, T eps, T momentum,
    bool train, T leakiness) {

  if (input_features.ndimension() == 2) {
    auto nActive = input_features.size(0);
    auto nPlanes = input_features.size(1);
    auto input_stride = input_features.stride(0);
    auto output_stride = output_features.stride(0);

    BatchNormalization_ForwardPass<T>(
        input_features.data<T>(), output_features.data<T>(), nPlanes,
        input_stride, output_stride, nActive, saveMean.data<T>(),
        saveInvStd.data<T>(), runningMean.data<T>(), runningVar.data<T>(),
        OptionalTensorData<T>(weight), OptionalTensorData<T>(bias), eps,
        momentum, train, leakiness);
  }
}

template <typename T>
void cpu_BatchNormalization_backward(
    /*float*/ at::Tensor input_features, /*float*/ at::Tensor d_input_features,
    /*float*/ at::Tensor output_features,
    /*float*/ at::Tensor d_output_features, /*float*/ at::Tensor saveMean,
    /*float*/ at::Tensor saveInvStd, /*float*/ at::Tensor runningMean,
    /*float*/ at::Tensor runningVar,
    /*float*/ at::Tensor weight, /*float*/ at::Tensor bias,
    /*float*/ at::Tensor d_weight, /*float*/ at::Tensor d_bias, T leakiness) {

  d_input_features.resize_as_(input_features);
  if (input_features.ndimension() == 2) {
    auto nActive = input_features.size(0);
    auto nPlanes = input_features.size(1);
    auto input_stride = input_features.stride(0);
    auto output_stride = output_features.stride(0);
    BatchNormalization_BackwardPass<T>(
        input_features.data<T>(), d_input_features.data<T>(),
        output_features.data<T>(), d_output_features.data<T>(), nPlanes,
        input_stride, output_stride, nActive, saveMean.data<T>(),
        saveInvStd.data<T>(), runningMean.data<T>(), runningVar.data<T>(),
        OptionalTensorData<T>(weight), OptionalTensorData<T>(bias),
        OptionalTensorData<T>(d_weight), OptionalTensorData<T>(d_bias),
        leakiness);
  }
}
