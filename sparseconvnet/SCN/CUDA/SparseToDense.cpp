// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

template <typename T>
void cuda_SparseToDense_ForwardPass(T *input_features, T *output_features,
                                    Int nPlanes, Int spatialVolume,
                                    RuleBook _rules);
template <typename T>
void cuda_SparseToDense_BackwardPass(T *d_input_features, T *d_output_features,
                                     Int nPlanes, Int spatialVolume,
                                     RuleBook _rules);

template <typename T, Int Dimension>
void cuda_SparseToDense_updateOutput(
    /*long*/ at::Tensor &inputSize, Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor &input_features,
    /*cuda float*/ at::Tensor &output_features, long nPlanes) {

  {
    std::array<long, Dimension + 2> sz;
    sz[0] = m.grids.begin()->second.size(); // batch size
    sz[1] = nPlanes;
    long *in_sz = inputSize.data_ptr<long>();
    for (Int i = 0; i < Dimension; ++i)
      sz[i + 2] = in_sz[i];
    output_features.resize_(sz);
    output_features.zero_();
  }
  if (input_features.ndimension() == 2) {
    const auto &_rules = m.getSparseToDenseRuleBook(inputSize, true);
    Int _nPlanes = input_features.size(1);
    auto iF = input_features.data_ptr<T>();
    auto oF = output_features.data_ptr<T>();
    long spatialVolume = inputSize.prod().data_ptr<long>()[0];
    cuda_SparseToDense_ForwardPass<T>(iF, oF, _nPlanes, spatialVolume, _rules);
  }
}
template <typename T, Int Dimension>
void cuda_SparseToDense_updateGradInput(
    /*long*/ at::Tensor &inputSize, Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor &input_features,
    /*cuda float*/ at::Tensor &d_input_features,
    /*cuda float*/ at::Tensor &d_output_features) {

  d_input_features.resize_as_(input_features);
  d_input_features.zero_();

  if (input_features.ndimension() == 2) {
    const auto &_rules = m.getSparseToDenseRuleBook(inputSize, true);
    long spatialVolume = inputSize.prod().data_ptr<long>()[0];
    Int _nPlanes = d_input_features.size(1);
    auto diF = d_input_features.data_ptr<T>();
    auto doF = d_output_features.data_ptr<T>();
    cuda_SparseToDense_BackwardPass<T>(diF, doF, _nPlanes, spatialVolume,
                                       _rules);
  }
}
