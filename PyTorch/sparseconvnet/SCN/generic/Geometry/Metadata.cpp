// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Geometry/Metadata.cpp"
#else

#include "Metadata.h"
#include <cstring>

extern "C" void scn_D_(setInputSpatialSize)(void **m,
                                            THLongTensor *spatialSize) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  _m.setInputSpatialSize(spatialSize);
}

extern "C" void scn_D_(batchAddSample)(void **m) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  assert(_m.inputSGs && "Call setInputSpatialSize first, please!");
  _m.inputSGs->resize(_m.inputSGs->size() + 1);
  _m.inputSG = &_m.inputSGs->back();
}
extern "C" void scn_D_(setInputSpatialLocation)(void **m,
                                                THFloatTensor *features,
                                                THLongTensor *location,
                                                THFloatTensor *vec,
                                                bool overwrite) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  auto p = LongTensorToPoint<Dimension>(location);
  auto &mp = _m.inputSG->mp;
  auto &nActive = *_m.inputNActive;
  auto iter = mp.find(p);
  auto nPlanes = vec->size[0];
  if (iter == mp.end()) {
    iter = mp.insert(std::make_pair(p, nActive++)).first;
    THFloatTensor_resize2d(features, nActive, nPlanes);
    std::memcpy(THFloatTensor_data(features) + (nActive - 1) * nPlanes,
                THFloatTensor_data(vec), sizeof(float) * nPlanes);
  } else if (overwrite) {
    std::memcpy(THFloatTensor_data(features) + iter->second * nPlanes,
                THFloatTensor_data(vec), sizeof(float) * nPlanes);
  }
}
extern "C" void
    scn_D_(createMetadataForDenseToSparse)(void **m, THLongTensor *spatialSize_,
                                           THLongTensor *pad_,
                                           THLongTensor *nz_, long batchSize) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  _m.setInputSpatialSize(spatialSize_);
  _m.inputSGs->resize(batchSize);
  auto &nActive = *_m.inputNActive;
  nActive = nz_->size[0];

  auto nz = THLongTensor_data(nz_);
  auto pad = THLongTensor_data(pad_);
  auto spatialSize = THLongTensor_data(spatialSize_);

  std::vector<uInt> br(batchSize + 1);
  if (batchSize == 1) {
    br[1] = nActive;
  } else {
    long b = 0;
    for (uInt i = 0; i < nActive; i++) {
      long B = nz[i * (Dimension + 1)];
      for (; b < B;)
        br[++b] = i;
    }
    for (; b < batchSize;)
      br[++b] = nActive;
  }
  uInt b;
#pragma omp parallel for private(b)
  for (b = 0; b < batchSize; b++) {
    auto &sg = _m.inputSGs->at(b);
    for (uInt i = br[b]; i < br[b + 1]; i++) {
      Point<Dimension> x;
      for (uInt j = 0; j < Dimension; j++) {
        x[j] = nz[i * (Dimension + 1) + j + 1] +
               pad[b * Dimension + j]; // 0-indexed
      }
      sg.mp[x] = i;
    }
  }
}

// tensor is size[0] x .. x size[Dimension-1] x size[Dimension]
// size[0] x .. x size[Dimension-1] == spatial volume
// size[Dimension] == #feature planes
extern "C" void scn_D_(addSampleFromThresholdedTensor)(
    void **m, THFloatTensor *features_, THFloatTensor *tensor_,
    THLongTensor *offset_, THLongTensor *spatialSize_, float threshold) {

  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  auto &nActive = *_m.inputNActive;
  auto &SGs = *_m.inputSGs;
  SGs.resize(SGs.size() + 1);
  auto &sg = SGs.back();

  auto tensor = THFloatTensor_data(tensor_);
  auto offset = THLongTensor_data(offset_);
  auto spatialSize = THLongTensor_data(spatialSize_);
  long *size = tensor_->size;
  auto nPlanes = size[Dimension];
  long volume = 1;
  for (int i = 0; i < Dimension; ++i)
    volume *= size[i];
  THFloatTensor_resize2d(features_, nActive + volume, nPlanes);
  // Increment pointers as we work through the data
  auto features = THFloatTensor_data(features_) + nActive * nPlanes;

  // Active locations
  Point<Dimension> point;
  for (uInt i = 0; i < Dimension; i++)
    point[i] = offset[i];
  for (uInt ctr = 0; ctr < volume; ctr++) {
    bool active = false;
    for (uInt i = 0; i < nPlanes; i++) {
      if (fabs(tensor[i]) > threshold) {
        active = true;
        break;
      }
    }
    for (uInt i = 0; i < Dimension; i++) {
      if (point[i] < 0 or point[i] >= spatialSize[i]) {
        active = false;
        break;
      }
    }
    if (active) {
      sg.mp[point] = nActive++;
      std::memcpy(features, tensor, sizeof(float) * nPlanes);
      features += nPlanes;
    }
    tensor += nPlanes;
    incrementPointInCube<Dimension>(point, size, offset);
  }
  THFloatTensor_resize2d(features_, nActive, nPlanes);
}

// 3x3 valid convolutions, 3x3/2x2 pooling or strided convolutions
extern "C" void scn_D_(generateRuleBooks3s2)(void **m) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  long sz[Dimension], str[Dimension], inS[Dimension], outS[Dimension];
  Point<Dimension> p1;
  Point<2 * Dimension> p2;
  Point<3 * Dimension> p3;
  for (int i = 0; i < Dimension; ++i) {
    p1[i] = p2[i] = p3[i] = inS[i] = _m.inputSpatialSize[i];
    p2[i + Dimension] = p3[i + Dimension] = sz[i] = 3;
    p3[i + 2 * Dimension] = str[i] = 2;
  }
  while (true) {
    auto &SGs = _m.grids[p1];
    auto &rb = _m.validRuleBooks[p2];
    if (rb.empty())
      ValidConvolution_SgsToRules(SGs, rb, sz);
    for (int i = 0; i < Dimension; ++i)
      if (p1[i] < 3 or p1[i] % 2 != 1)
        return;
      else
        p1[i] = outS[i] = (inS[i] - 1) / 2;
    auto &SGs2 = _m.grids[p1];
    auto &rb2 = _m.ruleBooks[p3];
    if (rb2.empty())
      _m.nActive[p1] = Convolution_InputSgsToRulesAndOutputSgs(
          SGs, SGs2, rb2, sz, str, inS, outS);
    for (int i = 0; i < Dimension; ++i)
      p2[i] = p3[i] = inS[i] = outS[i];
  }
}

// 3x3 valid convolutions, 2x2 pooling or strided convolutions
extern "C" void scn_D_(generateRuleBooks2s2)(void **m) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  long s2[Dimension], s3[Dimension], inS[Dimension], outS[Dimension];
  Point<Dimension> p1;
  Point<2 * Dimension> p2;
  Point<3 * Dimension> p3;
  for (int i = 0; i < Dimension; ++i) {
    p1[i] = p2[i] = p3[i] = inS[i] = _m.inputSpatialSize[i];
    p2[i + Dimension] = s3[i] = 3;
    p3[i + Dimension] = p3[i + 2 * Dimension] = s2[i] = 2;
  }
  while (true) {
    auto &SGs = _m.grids[p1];
    auto &rb = _m.validRuleBooks[p2];
    ValidConvolution_SgsToRules(SGs, rb, s3);
    for (int i = 0; i < Dimension; ++i)
      if (p1[i] < 2 or p1[i] % 2 != 0)
        return;
      else
        p1[i] = outS[i] = inS[i] / 2;
    auto &SGs2 = _m.grids[p1];
    auto &rb2 = _m.ruleBooks[p3];
    if (rb2.empty())
      _m.nActive[p1] = Convolution_InputSgsToRulesAndOutputSgs(
          SGs, SGs2, rb2, s2, s2, inS, outS);
    for (int i = 0; i < Dimension; ++i)
      p2[i] = p3[i] = inS[i] = outS[i];
  }
}
extern "C" void scn_D_(freeMetadata)(void **m) {
  SCN_DELETE(Metadata<Dimension>, m)
}

#endif
