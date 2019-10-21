// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

template <typename T>
void cuda_AveragePooling_ForwardPass(T *input_features, T *output_features,
                                     Int nPlanes, Int input_stride,
                                     Int output_stride, RuleBook _rules,
                                     Int filterVolume);

template <typename T>
void cuda_AveragePooling_BackwardPass(T *d_input_features, T *d_output_features,
                                      Int nPlanes, Int input_stride,
                                      Int output_stride, RuleBook _rules,
                                      Int filterVolume);

template <typename T, Int Dimension>
void cuda_AveragePooling_updateOutput(
    /*long*/ at::Tensor &inputSize, /*long*/ at::Tensor &outputSize,
    /*long*/ at::Tensor &poolSize,
    /*long*/ at::Tensor &poolStride, Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor &input_features,
    /*cuda float*/ at::Tensor &output_features, long nFeaturesToDrop) {

  Int nPlanes = input_features.size(1) - nFeaturesToDrop;
  const auto &_rules =
      m.getRuleBook(inputSize, outputSize, poolSize, poolStride, true);
  Int nActive = m.getNActive(outputSize);
  output_features.resize_({nActive, input_features.size(1) - nFeaturesToDrop});
  output_features.zero_();

  auto iF = input_features.data_ptr<T>() + nFeaturesToDrop;
  auto oF = output_features.data_ptr<T>();
  cuda_AveragePooling_ForwardPass<T>(iF, oF, nPlanes, input_features.size(1),
                                     output_features.size(1), _rules,
                                     _rules.size());
}

template <typename T, Int Dimension>
void cuda_AveragePooling_updateGradInput(
    /*long*/ at::Tensor &inputSize, /*long*/ at::Tensor &outputSize,
    /*long*/ at::Tensor &poolSize,
    /*long*/ at::Tensor &poolStride, Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor &input_features,
    /*cuda float*/ at::Tensor &d_input_features,
    /*cuda float*/ at::Tensor &d_output_features, long nFeaturesToDrop) {

  Int nPlanes = input_features.size(1) - nFeaturesToDrop;
  const auto &_rules =
      m.getRuleBook(inputSize, outputSize, poolSize, poolStride, true);
  d_input_features.resize_as_(input_features);
  d_input_features.zero_();

  auto diF = d_input_features.data_ptr<T>() + nFeaturesToDrop;
  auto doF = d_output_features.data_ptr<T>();
  cuda_AveragePooling_BackwardPass<T>(diF, doF, nPlanes, input_features.size(1),
                                      d_output_features.size(1), _rules,
                                      _rules.size());
}

template <typename T>
void cuda_CopyFeaturesHelper_ForwardPass(T *input_features, T *output_features,
                                         Int *rules, Int nPlanes, Int nHot);

template <typename T>
void cuda_CopyFeaturesHelper_BackwardPass(T *d_input_features,
                                          T *d_output_features, Int *rules,
                                          Int nPlanes, Int nHot);

template <typename T>
void cuda_CopyFeaturesHelper_updateOutput(at::Tensor &rules, at::Tensor &context,
                                          at::Tensor &Context) {

  Int nPlanes = context.size(1);
  Int nHot = rules.size(0) / 2;
  cuda_CopyFeaturesHelper_ForwardPass<T>(context.data_ptr<T>(), Context.data_ptr<T>(),
                                         rules.data_ptr<Int>(), nPlanes, nHot);
}

template <typename T>
void cuda_CopyFeaturesHelper_updateGradInput(at::Tensor &rules,
                                             at::Tensor &dcontext,
                                             at::Tensor &dContext) {

  Int nPlanes = dcontext.size(1);
  Int nHot = rules.size(0) / 2;
  cuda_CopyFeaturesHelper_BackwardPass<T>(
      dcontext.data_ptr<T>(), dContext.data_ptr<T>(), rules.data_ptr<Int>(), nPlanes, nHot);
}
