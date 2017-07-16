// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef TH_GENERIC_FILE_
#define TH_GENERIC_FILE_ "generic/CPU/SparseToDense.cpp"
#else
#include "SparseToDense.h"

extern "C" void scn_DR_(SparseToDense_updateOutput)(THLongTensor *inputSize,
                                                    void **m,
                                                    THTensor *input_features,
                                                    THTensor *output_features,
                                                    void *rulesBuffer) {

  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m) {
    long sz[Dimension + 2];
    sz[0] = _m.inputSGs->size();
    sz[1] = input_features->size[1];
    for (int i = 0; i < Dimension; i++) {
      auto x = THLongTensor_data(inputSize)[i];
      sz[i + 2] = x;
    }
    THTensor_(resizeNd)(output_features, Dimension + 2, sz, NULL);
    THTensor_(zero)(output_features);
  }
  auto _rules = _m.getSparseToDenseRuleBook(inputSize, true);
  auto spatialVolume = _rules.size();
  uInt nPlanes = input_features->size[1];
  auto iF = THTensor_(data)(input_features);
  auto oF = THTensor_(data)(output_features);

  for (auto &r : _rules) {
    uInt nHot = r.size() / 2;
    SparseToDense_ForwardPass<real>(iF, oF, nPlanes, spatialVolume, &r[0],
                                    nHot);
    oF += spatialVolume;
  }
}
extern "C" void scn_DR_(SparseToDense_updateGradInput)(
    THLongTensor *inputSize, void **m, THTensor *input_features,
    THTensor *d_input_features, THTensor *d_output_features,
    void *rulesBuffer) {

  THTensor_(resizeAs)(d_input_features, input_features);
  THTensor_(zero)(d_input_features);

  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  auto _rules = _m.getSparseToDenseRuleBook(inputSize, true);
  auto spatialVolume = _rules.size();
  uInt nPlanes = d_input_features->size[1];
  auto diF = THTensor_(data)(d_input_features);
  auto doF = THTensor_(data)(d_output_features);

  for (auto &r : _rules) {
    uInt nHot = r.size() / 2;
    SparseToDense_BackwardPass<real>(diF, doF, nPlanes, spatialVolume, &r[0],
                                     nHot);
    doF += spatialVolume;
  }
}
#endif
