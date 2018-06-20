// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "ActivePooling.h"

template <typename T, Int Dimension>
void cuda_ActivePooling_updateOutput(
    /*long*/ at::Tensor inputSize, Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor output_features, bool average) {

  Int nPlanes = input_features.size(1);
  auto _rules = m.getActivePoolingRuleBook(inputSize);
  Int batchSize = _rules[1][0];
  Int maxActive = _rules[1][1];
  output_features.resize_({batchSize, nPlanes});
  output_features.zero_();

  auto rulesBuffer = at::CUDA(at_kINT).tensor({1 << 22});
  Int *rb = rulesBuffer.data<Int>();
  Int rowBatchSize = std::min((Int)32768, (1 << 22) / (maxActive + 1));
  assert(rowBatchSize > 0);

  auto iF = input_features.data<T>();
  auto oF = output_features.data<T>();
  for (Int o = 0; o < batchSize; o += rowBatchSize) {
    Int batchSize_ = std::min(rowBatchSize, (Int(batchSize - o)));
    cudaMemcpy(rb, &_rules[0][o * (maxActive + 1)],
               sizeof(Int) * (maxActive + 1) * batchSize_,
               cudaMemcpyHostToDevice);
    ActivePooling_ForwardPass<T>(iF, oF + o * nPlanes, batchSize_, maxActive,
                                 nPlanes, rb, average);
  }
}
template <typename T, Int Dimension>
void cuda_ActivePooling_updateGradInput(
    /*long*/ at::Tensor inputSize, Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor input_features,
    /*cuda float*/ at::Tensor d_input_features,
    /*cuda float*/ at::Tensor d_output_features, bool average) {

  Int nPlanes = input_features.size(1);
  auto _rules = m.getActivePoolingRuleBook(inputSize);
  Int batchSize = _rules[1][0];
  Int maxActive = _rules[1][1];
  d_input_features.resize_as_(input_features);
  d_input_features.zero_();

  auto rulesBuffer = at::CUDA(at_kINT).tensor({1 << 22});
  Int *rb = rulesBuffer.data<Int>();
  Int rowBatchSize = std::min((Int)32768, (1 << 22) / (maxActive + 1));
  assert(rowBatchSize > 0);

  auto diF = d_input_features.data<T>();
  auto doF = d_output_features.data<T>();
  for (Int o = 0; o < batchSize; o += rowBatchSize) {
    Int batchSize_ = std::min(rowBatchSize, (Int(batchSize - o)));
    cudaMemcpy(rb, &_rules[0][o * (maxActive + 1)],
               sizeof(Int) * (maxActive + 1) * batchSize_,
               cudaMemcpyHostToDevice);
    ActivePooling_BackwardPass<T>(diF, doF + o * nPlanes, batchSize_, maxActive,
                                  nPlanes, rb, average);
  }
}
