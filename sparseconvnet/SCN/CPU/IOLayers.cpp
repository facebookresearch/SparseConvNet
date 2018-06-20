// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "IOLayers.h"

template <typename T, Int Dimension>
void cpu_InputLayer_updateOutput(Metadata<Dimension> &m,
                                 /*long*/ at::Tensor spatialSize,
                                 /*long*/ at::Tensor input_coords,
                                 /*float*/ at::Tensor input_features,
                                 /*float*/ at::Tensor output_features,
                                 long batchSize, long mode) {

  m.inputLayer(spatialSize, input_coords, batchSize, mode);
  auto nPlanes = input_features.size(1);
  auto &rules = m.inputLayerRuleBook;
  auto maxActive = rules[0][1];
  auto nRows = rules[0][3];
  if (mode == 0) {
    output_features.resize_as_(input_features);
    output_features.copy_(input_features);
  } else {
    output_features.resize_({*m.inputNActive, nPlanes});
    output_features.zero_();
    InputLayer_ForwardPass<T>(input_features.data<T>(),
                                 output_features.data<T>(), nRows,
                                 maxActive, nPlanes, &rules[1][0], mode == 4);
  }
}
template <typename T, Int Dimension>
void cpu_InputLayer_updateGradInput(Metadata<Dimension> &m,
                                    /*float*/ at::Tensor d_input_features,
                                    /*float*/ at::Tensor d_output_features) {

  auto &rules = m.inputLayerRuleBook;
  auto nPlanes = d_output_features.size(1);
  auto mode = rules[0][0];
  auto maxActive = rules[0][1];
  auto nRows = rules[0][3];
  if (mode == 0) {
    d_input_features.resize_as_(d_output_features);
    d_input_features.copy_(d_output_features);
  } else {
    d_input_features.resize_({rules[0][2], nPlanes});
    d_input_features.zero_();
    InputLayer_BackwardPass<T>(d_input_features.data<T>(),
                                  d_output_features.data<T>(), nRows,
                                  maxActive, nPlanes, &rules[1][0], mode == 4);
  }
}

template <typename T, Int Dimension>
void cpu_OutputLayer_updateOutput(Metadata<Dimension> &m,
                                  /*float*/ at::Tensor input_features,
                                  /*float*/ at::Tensor output_features) {

  auto &rules = m.inputLayerRuleBook;
  auto nPlanes = input_features.size(1);
  auto mode = rules[0][0];
  auto maxActive = rules[0][1];
  auto nRows = rules[0][3];
  if (mode == 0) {
    output_features.resize_as_(input_features);
    output_features.copy_(input_features);
  } else {
    output_features.resize_({rules[0][2], nPlanes});
    output_features.zero_();
    InputLayer_BackwardPass<T>(output_features.data<T>(),
                                  input_features.data<T>(), nRows,
                                  maxActive, nPlanes, &rules[1][0], false);
  }
}
template <typename T, Int Dimension>
void cpu_OutputLayer_updateGradInput(Metadata<Dimension> &m,
                                     /*float*/ at::Tensor d_input_features,
                                     /*float*/ at::Tensor d_output_features) {

  auto &rules = m.inputLayerRuleBook;
  auto nPlanes = d_output_features.size(1);
  auto mode = rules[0][0];
  auto maxActive = rules[0][1];
  auto nRows = rules[0][3];
  if (mode == 0) {
    d_input_features.resize_as_(d_output_features);
    d_input_features.copy_(d_output_features);
  } else {
    d_input_features.resize_({nRows, nPlanes});
    d_input_features.zero_();
    InputLayer_ForwardPass<T>(d_output_features.data<T>(),
                                 d_input_features.data<T>(), nRows,
                                 maxActive, nPlanes, &rules[1][0], false);
  }
}

template <typename T, Int Dimension>
void cpu_BLInputLayer_updateOutput(Metadata<Dimension> &m,
                                   /*long*/ at::Tensor spatialSize,
                                   /*long*/ at::Tensor input_coords,
                                   /*float*/ at::Tensor input_features,
                                   /*float*/ at::Tensor output_features,
                                   long mode) {

  m.blLayer(spatialSize, input_coords, mode);
  auto nPlanes = input_features.size(2);
  auto &rules = m.blLayerRuleBook;
  auto maxActive = rules[0][1];
  auto nRows = rules[0][4];
  if (mode == 0) {
    output_features.resize_as_(input_features);
    output_features.copy_(input_features);
    output_features.resize_({*m.inputNActive, nPlanes});
  } else {
    output_features.resize_({*m.inputNActive, nPlanes});
    output_features.zero_();
    InputLayer_ForwardPass<T>(input_features.data<T>(),
                                 output_features.data<T>(), nRows,
                                 maxActive, nPlanes, &rules[1][0], mode == 4);
  }
}
template <typename T, Int Dimension>
void cpu_BLInputLayer_updateGradInput(Metadata<Dimension> &m,
                                      /*float*/ at::Tensor d_input_features,
                                      /*float*/ at::Tensor d_output_features) {

  auto &rules = m.blLayerRuleBook;
  auto nPlanes = d_output_features.size(1);
  auto mode = rules[0][0];
  auto maxActive = rules[0][1];
  auto nRows = rules[0][4];

  if (mode == 0) {
    d_input_features.resize_as_(d_output_features);
    d_input_features.copy_(d_output_features);
    d_input_features.resize_({rules[0][2], rules[0][3], nPlanes});
  } else {
    d_input_features.resize_({rules[0][2], rules[0][3], nPlanes});
    d_input_features.zero_();
    InputLayer_BackwardPass<T>(d_input_features.data<T>(),
                                  d_output_features.data<T>(), nRows,
                                  maxActive, nPlanes, &rules[1][0], mode == 4);
  }
}

template <typename T, Int Dimension>
void cpu_BLOutputLayer_updateOutput(Metadata<Dimension> &m,
                                    /*float*/ at::Tensor input_features,
                                    /*float*/ at::Tensor output_features) {

  auto &rules = m.blLayerRuleBook;
  auto nPlanes = input_features.size(1);
  auto mode = rules[0][0];
  auto maxActive = rules[0][1];
  auto nRows = rules[0][4];
  if (mode == 0) {
    output_features.resize_as_(input_features);
    output_features.copy_(input_features);
    output_features.resize_({rules[0][2], rules[0][3], nPlanes});
  } else {
    output_features.resize_({rules[0][2], rules[0][3], nPlanes});
    output_features.zero_();
    InputLayer_BackwardPass<T>(output_features.data<T>(),
                                  input_features.data<T>(), nRows,
                                  maxActive, nPlanes, &rules[1][0], false);
  }
}
template <typename T, Int Dimension>
void cpu_BLOutputLayer_updateGradInput(Metadata<Dimension> &m,
                                       /*float*/ at::Tensor d_input_features,
                                       /*float*/ at::Tensor d_output_features) {

  auto &rules = m.blLayerRuleBook;
  auto nPlanes = d_output_features.size(2);
  auto mode = rules[0][0];
  auto maxActive = rules[0][1];
  auto nRows = rules[0][4];
  if (mode == 0) {
    d_input_features.resize_as_(d_output_features);
    d_input_features.copy_(d_output_features);
    d_input_features.resize_({nRows, nPlanes});
  } else {
    d_input_features.resize_({nRows, nPlanes});
    d_input_features.zero_();
    InputLayer_ForwardPass<T>(d_output_features.data<T>(),
                                 d_input_features.data<T>(), nRows,
                                 maxActive, nPlanes, &rules[1][0], false);
  }
}
