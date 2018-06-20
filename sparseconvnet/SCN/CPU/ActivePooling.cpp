// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "ActivePooling.h"

template <typename T, Int Dimension>
void cpu_ActivePooling_updateOutput(
    /*long*/ at::Tensor inputSize, Metadata<Dimension> &m,
    /*float*/ at::Tensor input_features,
    /*float*/ at::Tensor output_features, bool average) {

  Int nPlanes = input_features.size(1);
  auto _rules = m.getActivePoolingRuleBook(inputSize);
  Int batchSize = _rules[1][0];
  Int maxActive = _rules[1][1];
  output_features.resize_({batchSize, nPlanes});
  output_features.zero_();

  ActivePooling_ForwardPass<T>(input_features.data<T>(),
                               output_features.data<T>(), batchSize, maxActive,
                               nPlanes, _rules, average);
}

template <typename T, Int Dimension>
void cpu_ActivePooling_updateGradInput(
    /*long*/ at::Tensor inputSize, Metadata<Dimension> &m,
    /*float*/ at::Tensor input_features,
    /*float*/ at::Tensor d_input_features,
    /*float*/ at::Tensor d_output_features, bool average) {

  Int nPlanes = input_features.size(1);
  auto _rules = m.getActivePoolingRuleBook(inputSize);
  Int batchSize = _rules[1][0];
  Int maxActive = _rules[1][1];
  d_input_features.resize_as_(input_features);
  d_input_features.zero_();

  ActivePooling_BackwardPass<T>(d_input_features.data<T>(),
                                d_output_features.data<T>(), batchSize,
                                maxActive, nPlanes, _rules, average);
}
