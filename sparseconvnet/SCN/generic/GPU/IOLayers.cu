// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef TH_GENERIC_FILE_
#define TH_GENERIC_FILE_ "generic/GPU/IOLayers.cu"
#else
#include "IOLayers.h"

extern "C" void scn_DR_(InputLayer_updateOutput)(
    void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
    THCTensor *input_features, THCTensor *output_features, long batchSize,
    long mode) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  _m.inputLayer(spatialSize, input_coords, batchSize, mode);
  uInt nPlanes = input_features->size[1];
  auto &rules = _m.inputLayerRuleBook;
  uInt maxActive = rules[0][1];
  uInt nRows = rules[0][3];
  if (mode == 0) {
    THCTensor_(resizeAs)(state, output_features, input_features);
    THCTensor_(copy)(state, output_features, input_features);
  } else {
    THCTensor_(resize2d)(state, output_features, *_m.inputNActive, nPlanes);
    THCTensor_(zero)(state, output_features);
    auto rulesBuffer = THCITensor_(new)(state);
    THCITensor_(resize1d)(state, rulesBuffer, sizeof(uInt) * rules[1].size());
    auto iF = THCTensor_(data)(state, input_features);
    auto oF = THCTensor_(data)(state, output_features);
    auto rb = (uInt *)THCITensor_(data)(state, rulesBuffer);
    cudaMemcpy(rb, &rules[1][0], sizeof(uInt) * rules[1].size(),
               cudaMemcpyHostToDevice);
    InputLayer_fp<real><<<std::min(nRows, 32768U), std::min(nPlanes, 32U), 0,
                          THCState_getCurrentStream(state)>>>(
        iF, oF, nRows, maxActive, nPlanes, rb, mode == 4);
    THCITensor_(free)(state, rulesBuffer);
  }
}
extern "C" void
scn_DR_(InputLayer_updateGradInput)(void **m, THCTensor *d_input_features,
                                    THCTensor *d_output_features) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  auto &rules = _m.inputLayerRuleBook;
  uInt nPlanes = d_output_features->size[1];
  auto mode = rules[0][0];
  uInt maxActive = rules[0][1];
  uInt nRows = rules[0][3];
  if (mode == 0) {
    THCTensor_(resizeAs)(state, d_input_features, d_output_features);
    THCTensor_(copy)(state, d_input_features, d_output_features);
  } else {
    THCTensor_(resize2d)(state, d_input_features, rules[0][2], nPlanes);
    THCTensor_(zero)(state, d_input_features);
    auto rulesBuffer = THCITensor_(new)(state);
    THCITensor_(resize1d)(state, rulesBuffer, sizeof(uInt) * rules[1].size());
    auto diF = THCTensor_(data)(state, d_input_features);
    auto doF = THCTensor_(data)(state, d_output_features);
    auto rb = (uInt *)THCITensor_(data)(state, rulesBuffer);
    cudaMemcpy(rb, &rules[1][0], sizeof(uInt) * rules[1].size(),
               cudaMemcpyHostToDevice);
    InputLayer_bp<real><<<std::min(nRows, 32768U), std::min(nPlanes, 32U), 0,
                          THCState_getCurrentStream(state)>>>(
        diF, doF, nRows, maxActive, nPlanes, rb, mode == 4);
    THCITensor_(free)(state, rulesBuffer);
  }
}

extern "C" void scn_DR_(OutputLayer_updateOutput)(void **m,
                                                  THCTensor *input_features,
                                                  THCTensor *output_features) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  auto &rules = _m.inputLayerRuleBook;
  uInt nPlanes = input_features->size[1];
  auto mode = rules[0][0];
  auto maxActive = rules[0][1];
  auto nRows = rules[0][3];
  if (mode == 0) {
    THCTensor_(resizeAs)(state, output_features, input_features);
    THCTensor_(copy)(state, output_features, input_features);
  } else {
    THCTensor_(resize2d)(state, output_features, rules[0][2], nPlanes);
    THCTensor_(zero)(state, output_features);
    auto rulesBuffer = THCITensor_(new)(state);
    THCITensor_(resize1d)(state, rulesBuffer, sizeof(uInt) * rules[1].size());
    auto iF = THCTensor_(data)(state, input_features);
    auto oF = THCTensor_(data)(state, output_features);
    auto rb = (uInt *)THCITensor_(data)(state, rulesBuffer);
    cudaMemcpy(rb, &rules[1][0], sizeof(uInt) * rules[1].size(),
               cudaMemcpyHostToDevice);
    InputLayer_bp<real><<<std::min(nRows, 32768U), std::min(nPlanes, 32U), 0,
                          THCState_getCurrentStream(state)>>>(
        oF, iF, nRows, maxActive, nPlanes, rb, false);
    THCITensor_(free)(state, rulesBuffer);
  }
}
extern "C" void
scn_DR_(OutputLayer_updateGradInput)(void **m, THCTensor *d_input_features,
                                     THCTensor *d_output_features) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  auto &rules = _m.inputLayerRuleBook;
  uInt nPlanes = d_output_features->size[1];
  auto mode = rules[0][0];
  auto maxActive = rules[0][1];
  auto nRows = rules[0][3];
  if (mode == 0) {
    THCTensor_(resizeAs)(state, d_input_features, d_output_features);
    THCTensor_(copy)(state, d_input_features, d_output_features);
  } else {
    THCTensor_(resize2d)(state, d_input_features, nRows, nPlanes);
    THCTensor_(zero)(state, d_input_features);
    auto rulesBuffer = THCITensor_(new)(state);
    THCITensor_(resize1d)(state, rulesBuffer, sizeof(uInt) * rules[1].size());
    auto diF = THCTensor_(data)(state, d_input_features);
    auto doF = THCTensor_(data)(state, d_output_features);
    auto rb = (uInt *)THCITensor_(data)(state, rulesBuffer);
    cudaMemcpy(rb, &rules[1][0], sizeof(uInt) * rules[1].size(),
               cudaMemcpyHostToDevice);
    InputLayer_fp<real><<<std::min(nRows, 32768U), std::min(nPlanes, 32U), 0,
                          THCState_getCurrentStream(state)>>>(
        doF, diF, nRows, maxActive, nPlanes, rb, false);
    THCITensor_(free)(state, rulesBuffer);
  }
}

extern "C" void scn_DR_(BLInputLayer_updateOutput)(
    void **m, THLongTensor *spatialSize, THLongTensor *input_coords,
    THCTensor *input_features, THCTensor *output_features, long mode) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  _m.blLayer(spatialSize, input_coords, mode);
  uInt nPlanes = input_features->size[2];
  THCTensor_(resize2d)(state, output_features, *_m.inputNActive, nPlanes);
  THCTensor_(zero)(state, output_features);
  auto &rules = _m.blLayerRuleBook;
  uInt maxActive = rules[0][1];
  uInt nRows = rules[0][4];

  if (mode == 0) {
    THCTensor_(resizeAs)(state, output_features, input_features);
    THCTensor_(copy)(state, output_features, input_features);
    THCTensor_(resize2d)(state, output_features, *_m.inputNActive, nPlanes);
  } else {
    auto rulesBuffer = THCITensor_(new)(state);
    THCITensor_(resize1d)(state, rulesBuffer, sizeof(uInt) * rules[1].size());
    auto iF = THCTensor_(data)(state, input_features);
    auto oF = THCTensor_(data)(state, output_features);
    auto rb = (uInt *)THCITensor_(data)(state, rulesBuffer);
    cudaMemcpy(rb, &rules[1][0], sizeof(uInt) * rules[1].size(),
               cudaMemcpyHostToDevice);
    InputLayer_fp<real><<<std::min(nRows, 32768U), std::min(nPlanes, 32U), 0,
                          THCState_getCurrentStream(state)>>>(
        iF, oF, nRows, maxActive, nPlanes, rb, mode == 4);
    THCITensor_(free)(state, rulesBuffer);
  }
}
extern "C" void
scn_DR_(BLInputLayer_updateGradInput)(void **m, THCTensor *d_input_features,
                                      THCTensor *d_output_features) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  auto &rules = _m.blLayerRuleBook;
  uInt nPlanes = d_output_features->size[1];
  uInt mode = rules[0][0];
  uInt maxActive = rules[0][1];
  uInt nRows = rules[0][4];

  if (mode == 0) {
    THCTensor_(resizeAs)(state, d_input_features, d_output_features);
    THCTensor_(copy)(state, d_input_features, d_output_features);
    THCTensor_(resize3d)(state, d_input_features, rules[0][2], rules[0][3],
                         nPlanes);
  } else {
    THCTensor_(resize3d)(state, d_input_features, rules[0][2], rules[0][3],
                         nPlanes);
    THCTensor_(zero)(state, d_input_features);
    auto rulesBuffer = THCITensor_(new)(state);
    THCITensor_(resize1d)(state, rulesBuffer, sizeof(uInt) * rules[1].size());
    auto diF = THCTensor_(data)(state, d_input_features);
    auto doF = THCTensor_(data)(state, d_output_features);
    auto rb = (uInt *)THCITensor_(data)(state, rulesBuffer);
    cudaMemcpy(rb, &rules[1][0], sizeof(uInt) * rules[1].size(),
               cudaMemcpyHostToDevice);
    InputLayer_bp<real><<<std::min(nRows, 32768U), std::min(nPlanes, 32U), 0,
                          THCState_getCurrentStream(state)>>>(
        diF, doF, nRows, maxActive, nPlanes, rb, mode == 4);
  THCITensor_(free)(state, rulesBuffer);
  }
}

extern "C" void scn_DR_(BLOutputLayer_updateOutput)(void **m,
                                                    THCTensor *input_features,
                                                    THCTensor *output_features) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  auto &rules = _m.blLayerRuleBook;
  uInt nPlanes = input_features->size[1];
  auto mode = rules[0][0];
  uInt maxActive = rules[0][1];
  uInt nRows = rules[0][4];
  if (mode == 0) {
    THCTensor_(resizeAs)(state, output_features, input_features);
    THCTensor_(copy)(state, output_features, input_features);
    THCTensor_(resize3d)(state, output_features, rules[0][2], rules[0][3],
                         nPlanes);
  } else {
    THCTensor_(resize3d)(state, output_features, rules[0][2], rules[0][3],
                         nPlanes);
    THCTensor_(zero)(state, output_features);
    auto rulesBuffer = THCITensor_(new)(state);
    THCITensor_(resize1d)(state, rulesBuffer, sizeof(uInt) * rules[1].size());
    auto iF = THCTensor_(data)(state, input_features);
    auto oF = THCTensor_(data)(state, output_features);
    auto rb = (uInt *)THCITensor_(data)(state, rulesBuffer);
    cudaMemcpy(rb, &rules[1][0], sizeof(uInt) * rules[1].size(),
               cudaMemcpyHostToDevice);
    InputLayer_bp<real><<<std::min(nRows, 32768U), std::min(nPlanes, 32U), 0,
                          THCState_getCurrentStream(state)>>>(
        oF, iF, nRows, maxActive, nPlanes, rb, false);
  THCITensor_(free)(state, rulesBuffer);
  }
}
extern "C" void
scn_DR_(BLOutputLayer_updateGradInput)(void **m, THCTensor *d_input_features,
                                       THCTensor *d_output_features) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  auto &rules = _m.blLayerRuleBook;
  uInt nPlanes = d_output_features->size[2];
  uInt mode = rules[0][0];
  uInt maxActive = rules[0][1];
  uInt nRows = rules[0][4];
  if (mode == 0) {
    THCTensor_(resizeAs)(state, d_input_features, d_output_features);
    THCTensor_(copy)(state, d_input_features, d_output_features);
    THCTensor_(resize2d)(state, d_input_features, nRows, nPlanes);
  } else {
    THCTensor_(resize2d)(state, d_input_features, nRows, nPlanes);
    THCTensor_(zero)(state, d_input_features);
    auto rulesBuffer = THCITensor_(new)(state);
    THCITensor_(resize1d)(state, rulesBuffer, sizeof(uInt) * rules[1].size());
    auto diF = THCTensor_(data)(state, d_input_features);
    auto doF = THCTensor_(data)(state, d_output_features);
    auto rb = (uInt *)THCITensor_(data)(state, rulesBuffer);
    cudaMemcpy(rb, &rules[1][0], sizeof(uInt) * rules[1].size(),
               cudaMemcpyHostToDevice);
    InputLayer_fp<real><<<std::min(nRows, 32768U), std::min(nPlanes, 32U), 0,
                          THCState_getCurrentStream(state)>>>(
        doF, diF, nRows, maxActive, nPlanes, rb, false);
   THCITensor_(free)(state, rulesBuffer);
  }
}
#endif
