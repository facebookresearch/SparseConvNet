// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

template <typename T>
void cuda_UnPooling_ForwardPass(T *input_features, T *output_features,
                                Int nPlanes, Int input_stride,
                                Int output_stride, RuleBook _rules);
template <typename T>
void cuda_UnPooling_BackwardPass(T *d_input_features, T *d_output_features,
                                 Int nPlanes, Int input_stride,
                                 Int output_stride, RuleBook _rules);

template <typename T, Int Dimension>
void cuda_UnPooling_updateOutput(
    /*long*/ at::Tensor &inputSize, /*long*/ at::Tensor &outputSize,
    /*long*/ at::Tensor &poolSize,
    /*long*/ at::Tensor &poolStride, Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor &input_features,
    /*cuda float*/ at::Tensor &output_features, long nFeaturesToDrop) {

  Int nPlanes = input_features.size(1) - nFeaturesToDrop;
  const auto &_rules =
      m.getRuleBook(outputSize, inputSize, poolSize, poolStride, true);
  Int nActive = m.getNActive(outputSize);
  output_features.resize_({nActive, input_features.size(1) - nFeaturesToDrop});
  output_features.zero_();

  auto iF = input_features.data_ptr<T>() + nFeaturesToDrop;
  auto oF = output_features.data_ptr<T>();

  cuda_UnPooling_ForwardPass<T>(iF, oF, nPlanes, input_features.size(1),
                                output_features.size(1), _rules);
}

template <typename T, Int Dimension>
void cuda_UnPooling_updateGradInput(
    /*long*/ at::Tensor &inputSize, /*long*/ at::Tensor &outputSize,
    /*long*/ at::Tensor &poolSize,
    /*long*/ at::Tensor &poolStride, Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor &d_input_features,
    /*cuda float*/ at::Tensor &d_output_features, long nFeaturesToDrop) {

  Int nPlanes = d_input_features.size(1) - nFeaturesToDrop;
  const auto &_rules =
      m.getRuleBook(outputSize, inputSize, poolSize, poolStride, true);

  auto diF = d_input_features.data_ptr<T>() + nFeaturesToDrop;
  auto doF = d_output_features.data_ptr<T>();

  cuda_UnPooling_BackwardPass<T>(diF, doF, nPlanes, d_input_features.size(1),
                                 d_output_features.size(1), _rules);
}
