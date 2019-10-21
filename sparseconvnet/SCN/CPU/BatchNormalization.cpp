// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <vector>

// in/output_stride is normally the same as nPlanes; allow other values to act
// on a subset of columns, i.e. an inplace DenseNet blocks

template <typename T>
void BatchNormalization_ForwardPass(T *input_features, T *output_features,
                                    Int nPlanes, Int input_stride,
                                    Int output_stride, Int nActive, T *saveMean,
                                    T *saveInvStd, T *runningMean,
                                    T *runningVar, T *weight, T *bias, T eps,
                                    T momentum, bool train, T leakiness) {
  if (train) {
    std::memset(saveMean, 0, nPlanes * sizeof(T));
    std::memset(saveInvStd, 0, nPlanes * sizeof(T));
    for (Int row = 0; row < nActive; row++) {
      Int ci = row * input_stride;
      for (Int plane = 0; plane < nPlanes; plane++, ci++) {
        T ifci = input_features[ci];
        saveMean[plane] += ifci;
        saveInvStd[plane] += ifci * ifci; // accumulate sum-squares
                                          // before inverse square
                                          // rooting
      }
    }
    for (Int plane = 0; plane < nPlanes; plane++) {
      saveMean[plane] /= nActive;
      runningMean[plane] =
          momentum * runningMean[plane] + (1 - momentum) * saveMean[plane];
      saveInvStd[plane] -= saveMean[plane] * saveMean[plane] * nActive;
      runningVar[plane] = momentum * runningVar[plane] +
                          (1 - momentum) * saveInvStd[plane] / (nActive - 1);
      saveInvStd[plane] = powf(saveInvStd[plane] / nActive + eps, -0.5);
    }
  } else {
    for (Int plane = 0; plane < nPlanes; plane++) {
      saveMean[plane] = runningMean[plane];
      saveInvStd[plane] = powf(runningVar[plane] + eps, -0.5);
    }
  }
  std::vector<T> w(nPlanes);
  std::vector<T> b(nPlanes);
  for (Int plane = 0; plane < nPlanes; plane++) {
    w[plane] = saveInvStd[plane] * (weight ? weight[plane] : 1);
    b[plane] = -saveMean[plane] * w[plane] + (bias ? bias[plane] : 0);
  }
  for (Int row = 0; row < nActive; row++) {
    Int ci = row * input_stride;
    Int co = row * output_stride;
    for (Int plane = 0; plane < nPlanes; plane++, ci++, co++) {
      T out = input_features[ci] * w[plane] + b[plane];
      const T r = (out > 0) ? 1 : leakiness;
      output_features[co] = out * r;
    }
  }
}

template <typename T>
void BatchNormalization_BackwardPass(T *input_features, T *d_input_features,
                                     T *output_features, T *d_output_features,
                                     Int nPlanes, Int input_stride,
                                     Int output_stride, Int nActive,
                                     T *saveMean, T *saveInvStd, T *runningMean,
                                     T *runningVar, T *weight, T *bias,
                                     T *d_weight, T *d_bias, T leakiness) {
  std::vector<T> gradMean(nPlanes);
  std::vector<T> dotp(nPlanes);
  std::vector<T> k(nPlanes);
  for (Int row = 0; row < nActive; row++) {
    Int ci = row * input_stride;
    Int co = row * output_stride;
    for (Int plane = 0; plane < nPlanes; plane++, ci++, co++) {
      T d = d_output_features[co];
      const T r = (output_features[co] > 0) ? 1 : leakiness;
      d *= r;
      d_output_features[co] = d;
      gradMean[plane] += d;
      dotp[plane] += (input_features[ci] - saveMean[plane]) * d;
    }
  }
  for (Int plane = 0; plane < nPlanes; plane++) {
    if (d_bias)
      d_bias[plane] = gradMean[plane]; // sum of grads, really, until ...
    gradMean[plane] /= nActive;        // ...now
    k[plane] = dotp[plane] * saveInvStd[plane] * saveInvStd[plane] / nActive;
  }
  for (Int row = 0; row < nActive; row++) {
    Int ci = row * input_stride;
    Int co = row * output_stride;
    for (Int plane = 0; plane < nPlanes; plane++, ci++, co++) {
      d_input_features[ci] =
          (d_output_features[co] - gradMean[plane] -
           (input_features[ci] - saveMean[plane]) * k[plane]) *
          saveInvStd[plane] * (weight ? weight[plane] : 1);
    }
  }
  if (d_weight)
    for (Int plane = 0; plane < nPlanes; plane++) {
      d_weight[plane] = dotp[plane] * saveInvStd[plane];
    }
}

template <typename T>
void cpu_BatchNormalization_updateOutput(
    /*float*/ at::Tensor &input_features, /*float*/ at::Tensor &output_features,
    /*float*/ at::Tensor &saveMean,
    /*float*/ at::Tensor &saveInvStd, /*float*/ at::Tensor &runningMean,
    /*float*/ at::Tensor &runningVar,
    /*float*/ at::Tensor &weight, /*float*/ at::Tensor &bias, T eps, T momentum,
    bool train, T leakiness) {
  output_features.resize_as_(input_features);
  if (input_features.ndimension() == 2) {
    auto nActive = input_features.size(0);
    auto nPlanes = input_features.size(1);
    auto input_stride = input_features.stride(0);
    auto output_stride = output_features.stride(0);
    BatchNormalization_ForwardPass<T>(
        input_features.data_ptr<T>(), output_features.data_ptr<T>(), nPlanes,
        input_stride, output_stride, nActive, saveMean.data_ptr<T>(),
        saveInvStd.data_ptr<T>(), runningMean.data_ptr<T>(), runningVar.data_ptr<T>(),
        OptionalTensorData<T>(weight), OptionalTensorData<T>(bias), eps,
        momentum, train, leakiness);
  }
}

template <typename T>
void cpu_BatchNormalization_backward(
    /*float*/ at::Tensor &input_features,
    /*float*/ at::Tensor &d_input_features,
    /*float*/ at::Tensor &output_features,
    /*float*/ at::Tensor &d_output_features, /*float*/ at::Tensor &saveMean,
    /*float*/ at::Tensor &saveInvStd, /*float*/ at::Tensor &runningMean,
    /*float*/ at::Tensor &runningVar,
    /*float*/ at::Tensor &weight, /*float*/ at::Tensor &bias,
    /*float*/ at::Tensor &d_weight, /*float*/ at::Tensor &d_bias, T leakiness) {

  d_input_features.resize_as_(input_features);
  if (input_features.ndimension() == 2) {
    auto nActive = input_features.size(0);
    auto nPlanes = input_features.size(1);
    auto input_stride = input_features.stride(0);
    auto output_stride = output_features.stride(0);
    BatchNormalization_BackwardPass<T>(
        input_features.data_ptr<T>(), d_input_features.data_ptr<T>(),
        output_features.data_ptr<T>(), d_output_features.data_ptr<T>(), nPlanes,
        input_stride, output_stride, nActive, saveMean.data_ptr<T>(),
        saveInvStd.data_ptr<T>(), runningMean.data_ptr<T>(), runningVar.data_ptr<T>(),
        OptionalTensorData<T>(weight), OptionalTensorData<T>(bias),
        OptionalTensorData<T>(d_weight), OptionalTensorData<T>(d_bias),
        leakiness);
  }
}
