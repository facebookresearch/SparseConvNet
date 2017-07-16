// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef TH_GENERIC_FILE_
#define TH_GENERIC_FILE_ "generic/GPU/SparseToDense.cu"
#else
#include "SparseToDense.h"

extern "C" void scn_DR_(SparseToDense_updateOutput)(THLongTensor *inputSize,
                                                    void **m,
                                                    THCTensor *input_features,
                                                    THCTensor *output_features,
                                                    THCITensor *rulesBuffer) {

  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m) {
    long sz[Dimension + 2];
    sz[0] = _m.inputSGs->size();
    sz[1] = input_features->size[1];
    for (int i = 0; i < Dimension; i++) {
      auto x = THLongTensor_data(inputSize)[i];
      sz[i + 2] = x;
    }
    THCTensor_(resizeNd)(state, output_features, Dimension + 2, sz, NULL);
    THCTensor_(zero)(state, output_features);
  }
  auto _rules = _m.getSparseToDenseRuleBook(inputSize, true);
  auto spatialVolume = _rules.size();
  uInt nPlanes = input_features->size[1];
  auto iF = THCTensor_(data)(state, input_features);
  auto oF = THCTensor_(data)(state, output_features);
  RULEBOOKITERATOR(
      SparseToDense_ForwardPass<real>(THCState_getCurrentStream(state), iF, oF,
                                      nPlanes, spatialVolume, rbB, nHotB);
      , oF++;) // todo check ++ or +=spatialVolume????zzz
}
extern "C" void scn_DR_(SparseToDense_updateGradInput)(
    THLongTensor *inputSize, void **m, THCTensor *input_features,
    THCTensor *d_input_features, THCTensor *d_output_features,
    THCITensor *rulesBuffer) {

  THCTensor_(resizeAs)(state, d_input_features, input_features);
  THCTensor_(zero)(state, d_input_features);

  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  auto _rules = _m.getSparseToDenseRuleBook(inputSize, true);
  auto spatialVolume = _rules.size();
  uInt nPlanes = d_input_features->size[1];
  auto diF = THCTensor_(data)(state, d_input_features);
  auto doF = THCTensor_(data)(state, d_output_features);
  RULEBOOKITERATOR(
      SparseToDense_BackwardPass<real>(THCState_getCurrentStream(state), diF,
                                       doF, nPlanes, spatialVolume, rbB, nHotB);
      , doF++;)
}
#endif
