// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef TH_GENERIC_FILE_
#define TH_GENERIC_FILE_ "generic/CPU/InputLayer.cpp"
#else
#include "InputLayer.h"

extern "C" void scn_DR_(InputLayer_updateOutput)(
    void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
    THTensor *input_features, THTensor *output_features, long batchSize,
    long mode, void *rulesBuffer) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  _m.inputLayer(spatialSize, input_coords, batchSize, mode);
  auto nPlanes = input_features->size[1];
  THTensor_(resize2d)(output_features, *_m.inputNActive, nPlanes);
  THTensor_(zero)(output_features);
  auto &rules = _m.inputLayerRuleBook;
  auto maxActive = rules[0][1];
  auto nRows = rules[0][3];
  InputLayer_ForwardPass<real>(THTensor_(data)(input_features),
                               THTensor_(data)(output_features), nRows,
                               maxActive, nPlanes, &rules[1][0], mode == 4);
}
extern "C" void scn_DR_(InputLayer_updateGradInput)(void **m,
                                                    THTensor *d_input_features,
                                                    THTensor *d_output_features,
                                                    void *rulesBuffer) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  auto &rules = _m.inputLayerRuleBook;
  auto nPlanes = d_output_features->size[1];
  THTensor_(resize2d)(d_input_features, rules[0][2], nPlanes);
  THTensor_(zero)(d_input_features);
  auto mode = rules[0][0];
  auto maxActive = rules[0][1];
  auto nRows = rules[0][3];
  InputLayer_BackwardPass<real>(THTensor_(data)(d_input_features),
                                THTensor_(data)(d_output_features), nRows,
                                maxActive, nPlanes, &rules[1][0], mode == 4);
}

extern "C" void scn_DR_(BLInputLayer_updateOutput)(
    void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
    THTensor *input_features, THTensor *output_features, long mode,
    void *rulesBuffer) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  _m.blLayer(spatialSize, input_coords, mode);
  auto nPlanes = input_features->size[2];
  THTensor_(resize2d)(output_features, *_m.inputNActive, nPlanes);
  THTensor_(zero)(output_features);
  auto &rules = _m.blLayerRuleBook;
  auto maxActive = rules[0][1];
  auto nRows = rules[0][4];
  InputLayer_ForwardPass<real>(THTensor_(data)(input_features),
                               THTensor_(data)(output_features), nRows,
                               maxActive, nPlanes, &rules[1][0], mode == 4);
}
extern "C" void
scn_DR_(BLInputLayer_updateGradInput)(void **m, THTensor *d_input_features,
                                      THTensor *d_output_features,
                                      void *rulesBuffer) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  auto &rules = _m.blLayerRuleBook;
  auto nPlanes = d_output_features->size[1];
  THTensor_(resize3d)(d_input_features, rules[0][2], rules[0][3], nPlanes);
  THTensor_(zero)(d_input_features);
  auto mode = rules[0][0];
  auto maxActive = rules[0][1];
  auto nRows = rules[0][4];

  InputLayer_BackwardPass<real>(THTensor_(data)(d_input_features),
                                THTensor_(data)(d_output_features), nRows,
                                maxActive, nPlanes, &rules[1][0], mode == 4);
}

extern "C" void scn_DR_(BLOutputLayer_updateOutput)(
    void **m,
    THTensor *input_features, THTensor *output_features,
    void *rulesBuffer) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  auto &rules = _m.blLayerRuleBook;
  auto nPlanes = input_features->size[1];
  THTensor_(resize3d)(output_features, rules[0][2], rules[0][3], nPlanes);
  THTensor_(zero)(output_features);
  auto mode = rules[0][0];
  auto maxActive = rules[0][1];
  auto nRows = rules[0][4];
  InputLayer_BackwardPass<real>(THTensor_(data)(output_features),
                                THTensor_(data)(input_features), nRows,
                                maxActive, nPlanes, &rules[1][0], false);
}
extern "C" void
scn_DR_(BLOutputLayer_updateGradInput)(void **m, THTensor *d_input_features,
                                       THTensor *d_output_features,
                                       void *rulesBuffer) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  auto &rules = _m.blLayerRuleBook;
  auto nPlanes = d_output_features->size[2];
  auto mode = rules[0][0];
  auto maxActive = rules[0][1];
  auto nRows = rules[0][4];
  THTensor_(resize2d)(d_input_features, nRows, nPlanes);
  THTensor_(zero)(d_input_features);

  InputLayer_ForwardPass<real>(THTensor_(data)(d_output_features),
                               THTensor_(data)(d_input_features), nRows,
                               maxActive, nPlanes, &rules[1][0], false);
}
#endif
