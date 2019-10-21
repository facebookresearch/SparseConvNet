// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

template <typename T>
void InputLayer_fp(T *input_features, T *output_features, Int nRows,
                   Int maxActive, Int nPlanes, Int *rules_cpu, Int *rules_gpu,
                   bool average);

template <typename T>
void InputLayer_bp(T *d_input_features, T *d_output_features, Int nRows,
                   Int maxActive, Int nPlanes, Int *rules_cpu, Int *rules_gpu,
                   bool average);

template <typename T, Int Dimension>
void cuda_InputLayer_updateOutput(Metadata<Dimension> &m,
                                  /*long*/ at::Tensor &spatialSize,
                                  /*long*/ at::Tensor &input_coords,
                                  /*cuda float*/ at::Tensor &input_features,
                                  /*cuda float*/ at::Tensor &output_features,
                                  long batchSize, long mode) {

  m.inputLayer(spatialSize, input_coords, batchSize, mode);
  Int nPlanes = input_features.size(1);
  auto &rules = m.inputLayerRuleBook;
  Int maxActive = rules[0][1];
  Int nRows = rules[0][3];
  if (mode == 0) {
    output_features.resize_as_(input_features);
    output_features.copy_(input_features);
  } else {
    output_features.resize_({*m.inputNActive, nPlanes});
    output_features.zero_();
    auto rulesBuffer = at::empty({(int)rules[1].size()}, at::CUDA(at_kINT));
    auto iF = input_features.data_ptr<T>();
    auto oF = output_features.data_ptr<T>();
    Int *rb = rulesBuffer.data_ptr<Int>();
    InputLayer_fp<T>(iF, oF, nRows, maxActive, nPlanes, &rules[1][0], rb,
                     mode == 4);
  }
}
template <typename T, Int Dimension>
void cuda_InputLayer_updateGradInput(
    Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor &d_input_features,
    /*cuda float*/ at::Tensor &d_output_features) {

  auto &rules = m.inputLayerRuleBook;
  Int nPlanes = d_output_features.size(1);
  auto mode = rules[0][0];
  Int maxActive = rules[0][1];
  Int nRows = rules[0][3];
  if (mode == 0) {
    d_input_features.resize_as_(d_output_features);
    d_input_features.copy_(d_output_features);
  } else {
    d_input_features.resize_({rules[0][2], nPlanes});
    d_input_features.zero_();
    auto rulesBuffer = at::empty({(int)rules[1].size()}, at::CUDA(at_kINT));
    auto diF = d_input_features.data_ptr<T>();
    auto doF = d_output_features.data_ptr<T>();
    Int *rb = rulesBuffer.data_ptr<Int>();
    InputLayer_bp(diF, doF, nRows, maxActive, nPlanes, &rules[1][0], rb,
                  mode == 4);
  }
}

template <typename T, Int Dimension>
void cuda_OutputLayer_updateOutput(Metadata<Dimension> &m,
                                   /*cuda float*/ at::Tensor &input_features,
                                   /*cuda float*/ at::Tensor &output_features) {

  auto &rules = m.inputLayerRuleBook;
  Int nPlanes = input_features.size(1);
  auto mode = rules[0][0];
  auto maxActive = rules[0][1];
  auto nRows = rules[0][3];
  if (mode == 0) {
    output_features.resize_as_(input_features);
    output_features.copy_(input_features);
  } else {
    output_features.resize_({rules[0][2], nPlanes});
    output_features.zero_();
    auto rulesBuffer = at::empty({(int)rules[1].size()}, at::CUDA(at_kINT));
    auto iF = input_features.data_ptr<T>();
    auto oF = output_features.data_ptr<T>();
    Int *rb = rulesBuffer.data_ptr<Int>();
    InputLayer_bp(oF, iF, nRows, maxActive, nPlanes, &rules[1][0], rb, false);
  }
}
template <typename T, Int Dimension>
void cuda_OutputLayer_updateGradInput(
    Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor &d_input_features,
    /*cuda float*/ at::Tensor &d_output_features) {

  auto &rules = m.inputLayerRuleBook;
  Int nPlanes = d_output_features.size(1);
  auto mode = rules[0][0];
  auto maxActive = rules[0][1];
  auto nRows = rules[0][3];
  if (mode == 0) {
    d_input_features.resize_as_(d_output_features);
    d_input_features.copy_(d_output_features);
  } else {
    d_input_features.resize_({nRows, nPlanes});
    d_input_features.zero_();
    auto rulesBuffer = at::empty({(int)rules[1].size()}, at::CUDA(at_kINT));
    auto diF = d_input_features.data_ptr<T>();
    auto doF = d_output_features.data_ptr<T>();
    Int *rb = rulesBuffer.data_ptr<Int>();
    InputLayer_fp<T>(doF, diF, nRows, maxActive, nPlanes, &rules[1][0], rb,
                     false);
  }
}

template <typename T, Int Dimension>
void cuda_BLInputLayer_updateOutput(Metadata<Dimension> &m,
                                    /*long*/ at::Tensor &spatialSize,
                                    /*long*/ at::Tensor &input_coords,
                                    /*cuda float*/ at::Tensor &input_features,
                                    /*cuda float*/ at::Tensor &output_features,
                                    long mode) {

  m.blLayer(spatialSize, input_coords, mode);
  Int nPlanes = input_features.size(2);
  output_features.resize_({*m.inputNActive, nPlanes});
  output_features.zero_();
  auto &rules = m.blLayerRuleBook;
  Int maxActive = rules[0][1];
  Int nRows = rules[0][4];

  if (mode == 0) {
    output_features.resize_as_(input_features);
    output_features.copy_(input_features);
    output_features.resize_({*m.inputNActive, nPlanes});
  } else {
    auto rulesBuffer = at::empty({(int)rules[1].size()}, at::CUDA(at_kINT));
    auto iF = input_features.data_ptr<T>();
    auto oF = output_features.data_ptr<T>();
    Int *rb = rulesBuffer.data_ptr<Int>();
    InputLayer_fp<T>(iF, oF, nRows, maxActive, nPlanes, &rules[1][0], rb,
                     mode == 4);
  }
}
template <typename T, Int Dimension>
void cuda_BLInputLayer_updateGradInput(
    Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor &d_input_features,
    /*cuda float*/ at::Tensor &d_output_features) {

  auto &rules = m.blLayerRuleBook;
  Int nPlanes = d_output_features.size(1);
  Int mode = rules[0][0];
  Int maxActive = rules[0][1];
  Int nRows = rules[0][4];

  if (mode == 0) {
    d_input_features.resize_as_(d_output_features);
    d_input_features.copy_(d_output_features);
    d_input_features.resize_({rules[0][2], rules[0][3], nPlanes});
  } else {
    d_input_features.resize_({rules[0][2], rules[0][3], nPlanes});
    d_input_features.zero_();
    auto rulesBuffer = at::empty({(int)rules[1].size()}, at::CUDA(at_kINT));
    auto diF = d_input_features.data_ptr<T>();
    auto doF = d_output_features.data_ptr<T>();
    Int *rb = rulesBuffer.data_ptr<Int>();
    InputLayer_bp(diF, doF, nRows, maxActive, nPlanes, &rules[1][0], rb,
                  mode == 4);
  }
}

template <typename T, Int Dimension>
void cuda_BLOutputLayer_updateOutput(
    Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor &input_features,
    /*cuda float*/ at::Tensor &output_features) {

  auto &rules = m.blLayerRuleBook;
  Int nPlanes = input_features.size(1);
  auto mode = rules[0][0];
  Int maxActive = rules[0][1];
  Int nRows = rules[0][4];
  if (mode == 0) {
    output_features.resize_as_(input_features);
    output_features.copy_(input_features);
    output_features.resize_({rules[0][2], rules[0][3], nPlanes});
  } else {
    output_features.resize_({rules[0][2], rules[0][3], nPlanes});
    output_features.zero_();
    auto rulesBuffer = at::empty({(int)rules[1].size()}, at::CUDA(at_kINT));
    auto iF = input_features.data_ptr<T>();
    auto oF = output_features.data_ptr<T>();
    Int *rb = rulesBuffer.data_ptr<Int>();
    InputLayer_bp(oF, iF, nRows, maxActive, nPlanes, &rules[1][0], rb, false);
  }
}
template <typename T, Int Dimension>
void cuda_BLOutputLayer_updateGradInput(
    Metadata<Dimension> &m,
    /*cuda float*/ at::Tensor &d_input_features,
    /*cuda float*/ at::Tensor &d_output_features) {

  auto &rules = m.blLayerRuleBook;
  Int nPlanes = d_output_features.size(2);
  Int mode = rules[0][0];
  Int maxActive = rules[0][1];
  Int nRows = rules[0][4];
  if (mode == 0) {
    d_input_features.resize_as_(d_output_features);
    d_input_features.copy_(d_output_features);
    d_input_features.resize_({nRows, nPlanes});
  } else {
    d_input_features.resize_({nRows, nPlanes});
    d_input_features.zero_();
    auto rulesBuffer = at::empty({(int)rules[1].size()}, at::CUDA(at_kINT));
    auto diF = d_input_features.data_ptr<T>();
    auto doF = d_output_features.data_ptr<T>();
    Int *rb = rulesBuffer.data_ptr<Int>();
    InputLayer_fp<T>(doF, diF, nRows, maxActive, nPlanes, &rules[1][0], rb,
                     false);
  }
}
