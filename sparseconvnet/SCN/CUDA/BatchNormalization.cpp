// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

template <typename T>
void bn_f(T *iF, T *oF, Int nPlanes, Int input_stride, Int output_stride,
          Int nActive, T *saveMean, T *saveInvStd, T *runningMean,
          T *runningVar, T *weight, T *bias, T eps, T momentum, bool train,
          T leakiness);

template <typename T>
void bn_b(T *input_features, T *d_input_features, T *output_features,
          T *d_output_features, Int nPlanes, Int input_stride,
          Int output_stride, Int nActive, T *saveMean, T *saveInvStd,
          T *runningMean, T *runningVar, T *weight, T *bias, T *d_weight,
          T *d_bias, T leakiness);

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
    bn_f(input_features.data_ptr<T>(), output_features.data_ptr<T>(), nPlanes,
         input_stride, output_stride, nActive, saveMean.data_ptr<T>(),
         saveInvStd.data_ptr<T>(), runningMean.data_ptr<T>(), runningVar.data_ptr<T>(),
         OptionalTensorData<T>(weight), OptionalTensorData<T>(bias), eps,
         momentum, train, leakiness);
  }
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
    bn_b(input_features.data_ptr<T>(), d_input_features.data_ptr<T>(),
         output_features.data_ptr<T>(), d_output_features.data_ptr<T>(), nPlanes,
         input_stride, output_stride, nActive, saveMean.data_ptr<T>(),
         saveInvStd.data_ptr<T>(), runningMean.data_ptr<T>(), runningVar.data_ptr<T>(),
         OptionalTensorData<T>(weight), OptionalTensorData<T>(bias),
         OptionalTensorData<T>(d_weight), OptionalTensorData<T>(d_bias),
         leakiness);
  }
}
