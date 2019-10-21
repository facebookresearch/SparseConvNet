// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

template <typename T>
void ActivePooling_ForwardPass(T *input_features, T *output_features,
                               Int batchSize, Int maxActive, Int nPlanes,
                               const Int *rules, bool average);

template <typename T>
void ActivePooling_BackwardPass(T *d_input_features, T *d_output_features,
                                Int batchSize, Int maxActive, Int nPlanes,
                                const Int *rules, bool average);

template <typename T, Int Dimension>
void cuda_ActivePooling_updateOutput(
    /*long*/ at::Tensor &inputSize, Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor &input_features,
    /*cuda float*/ at::Tensor &output_features, bool average) {

  Int nPlanes = input_features.size(1);
  const auto &_rules = m.getActivePoolingRuleBook(inputSize);
  Int batchSize = _rules[1][0];
  Int maxActive = _rules[1][1];
  output_features.resize_({batchSize, nPlanes});
  output_features.zero_();

  auto iF = input_features.data_ptr<T>();
  auto oF = output_features.data_ptr<T>();
  ActivePooling_ForwardPass<T>(iF, oF, batchSize, maxActive, nPlanes,
                               &_rules[0][0], average);
}
template <typename T, Int Dimension>
void cuda_ActivePooling_updateGradInput(
    /*long*/ at::Tensor &inputSize, Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor &input_features,
    /*cuda float*/ at::Tensor &d_input_features,
    /*cuda float*/ at::Tensor &d_output_features, bool average) {

  Int nPlanes = input_features.size(1);
  const auto &_rules = m.getActivePoolingRuleBook(inputSize);
  Int batchSize = _rules[1][0];
  Int maxActive = _rules[1][1];
  d_input_features.resize_as_(input_features);
  d_input_features.zero_();

  auto diF = d_input_features.data_ptr<T>();
  auto doF = d_output_features.data_ptr<T>();

  ActivePooling_BackwardPass<T>(diF, doF, batchSize, maxActive, nPlanes,
                                &_rules[0][0], average);
}
