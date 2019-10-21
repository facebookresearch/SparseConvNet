// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "Metadata.h"

#include "ActivePoolingRules.h"
#include "ConvolutionRules.h"
#include "FullConvolutionRules.h"
#include "IOLayersRules.h"
#include "PermutohedralSubmanifoldConvolutionRules.h"
#include "RandomizedStrideRules.h"
#include "SubmanifoldConvolutionRules.h"

template <Int dimension> SparseGrid<dimension>::SparseGrid() : ctr(0) {
  // Sparsehash needs a key to be set aside and never used
  Point<dimension> empty_key;
  for (Int i = 0; i < dimension; ++i)
    empty_key[i] = std::numeric_limits<Int>::min();
  mp.set_empty_key(empty_key);
}

template <typename T> T *OptionalTensorData(at::Tensor &tensor) {
  return tensor.numel() ? tensor.data_ptr<T>() : nullptr;
}

template <Int dimension>
void addPointToSparseGridMapAndFeatures(SparseGridMap<dimension> &mp,
                                        Point<dimension> p, Int &nActive,
                                        long nPlanes,
                                        /*float*/ at::Tensor &features,
                                        float *vec, bool overwrite) {

  auto mapVal = mp.insert(std::make_pair(p, nActive));
  if (mapVal.second) {
    nActive++;
    features.resize_({(int)nActive, nPlanes});
    std::memcpy(features.data_ptr<float>() + (nActive - 1) * nPlanes, vec,
                sizeof(float) * nPlanes);
  } else if (overwrite) {
    std::memcpy(features.data_ptr<float>() + mapVal.first->second * nPlanes, vec,
                sizeof(float) * nPlanes);
  }
}

template <Int dimension>
Metadata<dimension>::Metadata()
    : re(std::chrono::system_clock::now().time_since_epoch().count()) {}

template <Int dimension> void Metadata<dimension>::clear() {
  nActive.clear();
  grids.clear();
  activePoolingRuleBooks.clear();
  inputLayerRuleBook.clear();
  submanifoldRuleBooks.clear();
  ruleBooks.clear();
  fullConvolutionRuleBook.clear();
  sparseToDenseRuleBooks.clear();
  inputSGs = nullptr;
  inputSG = nullptr;
  inputNActive = nullptr;
  inputLayerRuleBook.clear();
  blLayerRuleBook.clear();
}
template <Int dimension>
Int Metadata<dimension>::getNActive(/*long*/ at::Tensor &spatialSize) {
  return nActive[LongTensorToPoint<dimension>(spatialSize)];
};
template <Int dimension>
SparseGrids<dimension> &
Metadata<dimension>::getSparseGrid(/*long*/ at::Tensor &spatialSize) {
  return grids[LongTensorToPoint<dimension>(spatialSize)];
};
template <Int dimension>
void Metadata<dimension>::setInputSpatialSize(
    /*long*/ at::Tensor &spatialSize) {
  inputSpatialSize = LongTensorToPoint<dimension>(spatialSize);
  inputSGs = &grids[inputSpatialSize];
  inputNActive = &nActive[inputSpatialSize];
}
template <Int dimension> void Metadata<dimension>::batchAddSample() {
  assert(inputSGs && "Call setInputSpatialSize first, please!");
  inputSGs->resize(inputSGs->size() + 1);
  inputSG = &inputSGs->back();
}
template <Int dimension>
void Metadata<dimension>::setInputSpatialLocation(
    /*float*/ at::Tensor &features,
    /*long*/ at::Tensor &location,
    /*float*/ at::Tensor &vec, bool overwrite) {
  auto p = LongTensorToPoint<dimension>(location);
  SparseGridMap<dimension> &mp = inputSG->mp;
  Int &nActive = *inputNActive;
  auto nPlanes = vec.size(0);
  addPointToSparseGridMapAndFeatures<dimension>(
      mp, p, nActive, nPlanes, features, vec.data_ptr<float>(), overwrite);
}
template <Int dimension>
void Metadata<dimension>::setInputSpatialLocations(
    /*float*/ at::Tensor &features,
    /*long*/ at::Tensor &locations,
    /*float*/ at::Tensor &vecs, bool overwrite) {
  /* assert(locations.ndimension() == 2 and "locations must be 2
   * dimensional!"); */
  /* assert(vecs.ndimension() == 2 and "vecs must be 2 dimensional!"); */
  /* assert(locations.size(0) == vecs.size(0) and */
  /*        "Location.size(0) and vecs.size(0) must be equal!"); */
  /* assert((locations.size(1) == dimension or */
  /*         locations.size(1) == 1 + dimension) and */
  /*        "locations.size(0) must be either dimension or dimension+1"); */
  Point<dimension> p;
  Int &nActive = *inputNActive;
  auto nPlanes = vecs.size(1);
  long *l = locations.data_ptr<long>();
  float *v = vecs.data_ptr<float>();

  if (locations.size(1) == dimension) {
    // add points to current sample
    assert(inputSG);
    SparseGridMap<dimension> &mp = inputSG->mp;
    for (Int idx = 0; idx < locations.size(0); ++idx) {
      for (Int d = 0; d < dimension; ++d)
        p[d] = *l++;
      addPointToSparseGridMapAndFeatures<dimension>(mp, p, nActive, nPlanes,
                                                    features, v, overwrite);
      v += nPlanes;
    }
  }
  if (locations.size(1) == dimension + 1) {
    // add new samples to batch as necessary
    auto &SGs = *inputSGs;
    for (Int idx = 0; idx < locations.size(0); ++idx) {
      for (Int d = 0; d < dimension; ++d)
        p[d] = *l++;
      Int batch = *l++;
      if (batch >= (Int)SGs.size()) {
        SGs.resize(batch + 1);
      }
      SparseGridMap<dimension> &mp = SGs[batch].mp;
      addPointToSparseGridMapAndFeatures<dimension>(mp, p, nActive, nPlanes,
                                                    features, v, overwrite);
      v += nPlanes;
    }
  }
}

template <Int dimension>
at::Tensor
Metadata<dimension>::getSpatialLocations(/*long*/ at::Tensor &spatialSize) {
  Int nActive = getNActive(spatialSize);
  auto &SGs = getSparseGrid(spatialSize);
  Int batchSize = SGs.size();

  auto locations = torch::zeros({(int)nActive, dimension + 1}, at::kLong);
  auto lD = locations.data_ptr<long>();

  for (Int i = 0; i < batchSize; i++) {
    auto mp = SGs[i].mp;
    auto offset = SGs[i].ctr;
    for (auto it = mp.begin(); it != mp.end(); ++it) {
      for (Int d = 0; d < dimension; ++d) {
        lD[(it->second + offset) * (dimension + 1) + d] = it->first[d];
      }
      lD[(it->second + offset) * (dimension + 1) + dimension] = i;
    }
  }
  return locations;
}
template <Int dimension>
Int
Metadata<dimension>::getBatchSize(/*long*/ at::Tensor &spatialSize) {
  auto &SGs = getSparseGrid(spatialSize);
  return SGs.size();
}
template <Int dimension>
void Metadata<dimension>::createMetadataForDenseToSparse(
    /*long*/ at::Tensor &spatialSize,
    /*long*/ at::Tensor &nz_, long batchSize) {
  clear();
  setInputSpatialSize(spatialSize);
  inputSGs->resize(batchSize);
  auto &nActive = *inputNActive;
  nActive = nz_.size(0);

  long *nz = nz_.data_ptr<long>();

  std::vector<Int> br(batchSize + 1);
  if (batchSize == 1) {
    br[1] = nActive;
  } else {
    long b = 0;
    for (Int i = 0; i < nActive; i++) {
      long B = nz[i * (dimension + 1)];
      for (; b < B;)
        br[++b] = i;
    }
    for (; b < batchSize;)
      br[++b] = nActive;
  }
  Int b;
#pragma omp parallel for private(b)
  for (b = 0; b < batchSize; b++) {
    auto &sg = inputSGs->at(b);
    for (Int i = br[b]; i < br[b + 1]; i++) {
      Point<dimension> x;
      for (Int j = 0; j < dimension; j++) {
        x[j] = nz[i * (dimension + 1) + j + 1]; // 0-indexed
      }
      sg.mp[x] = i;
    }
  }
}

template <Int dimension>
void Metadata<dimension>::sparsifyMetadata(Metadata<dimension> &mOut,
                                           /*long*/ at::Tensor &spatialSize,
                                           /*byte*/ at::Tensor &filter,
                                           /*long*/ at::Tensor &cuSum) {
  // Create a new SparseGrids with fewer entries.
  mOut.clear();
  auto p = LongTensorToPoint<dimension>(spatialSize);
  auto &sgsIn = grids[p];
  auto &sgsOut = mOut.grids[p];
  sgsOut.resize(sgsIn.size());
  if (filter.ndimension() == 1) {
    auto f = filter.data_ptr<unsigned char>();
    auto cs = cuSum.data_ptr<long>();
    auto nActive = cs[cuSum.numel() - 1];
    mOut.nActive[p] = nActive;
    Int sample;
#pragma omp parallel for private(sample)
    for (sample = 0; sample < (Int)sgsIn.size(); ++sample) {
      auto &sgIn = sgsIn[sample];
      auto &sgOut = sgsOut[sample];
      for (auto const &iter : sgIn.mp) {
        auto n = iter.second + sgIn.ctr;
        if (f[n])
          sgOut.mp[iter.first] = cs[n] - 1;
      }
    }
  } else {
    mOut.nActive[p] = 0;
  }
}

template <Int dimension>
void Metadata<dimension>::appendMetadata(Metadata<dimension> &mAdd,
                                         /*long*/ at::Tensor &spatialSize) {
  auto p = LongTensorToPoint<dimension>(spatialSize);
  auto &sgs1 = grids[p];
  auto &sgs2 = mAdd.grids[p];
  auto &nActive1 = nActive[p];
  auto &nActive2 = mAdd.nActive[p];
  Int bs1 = sgs1.size();
  Int bs2 = sgs2.size();
  sgs1.insert(sgs1.end(), sgs2.begin(), sgs2.end());
  for (Int i = bs1; i < bs1 + bs2; ++i)
    sgs1[i].ctr += nActive1;
  nActive1 += nActive2;
}

template <Int dimension>
std::vector<at::Tensor>
Metadata<dimension>::sparsifyCompare(Metadata<dimension> &mGT,
                                     /*long*/ at::Tensor &spatialSize) {
  auto p = LongTensorToPoint<dimension>(spatialSize);
  at::Tensor gt = torch::zeros({nActive[p]}, at::kByte);
  at::Tensor ref_map = torch::empty({mGT.nActive[p]}, at::kLong);
  long *ref_map_ptr = ref_map.data_ptr<long>();
  unsigned char *gt_ptr = gt.data_ptr<unsigned char>();
  auto &sgsGT = mGT.grids[p];
  auto &sgsFull = grids[p];
  Int batchSize = sgsFull.size();
  Int sample;

#pragma omp parallel for private(sample)
  for (sample = 0; sample < (Int)batchSize; ++sample) {
    auto &sgGT = sgsGT[sample];
    auto &sgFull = sgsFull[sample];
    for (auto const &iter : sgGT.mp) {
      auto f = sgFull.mp.find(iter.first);
      if (f != sgFull.mp.end()) {
        ref_map_ptr[iter.second + sgGT.ctr] = f->second + sgFull.ctr;
        gt_ptr[f->second + sgFull.ctr] = +1;
      }
    }
  }
  return {gt, ref_map};
}

// tensor is size[0] x .. x size[dimension-1] x size[dimension]
// size[0] x .. x size[dimension-1] == spatial volume
// size[dimension] == #feature planes
template <Int dimension>
void Metadata<dimension>::addSampleFromThresholdedTensor(
    /*float*/ at::Tensor &features_,
    /*float*/ at::Tensor &tensor_,
    /*long*/ at::Tensor &offset_,
    /*long*/ at::Tensor &spatialSize_, float threshold) {

  auto &nActive = *inputNActive;
  auto &SGs = *inputSGs;
  SGs.resize(SGs.size() + 1);
  auto &sg = SGs.back();

  auto tensor = tensor_.data_ptr<float>();
  auto offset = offset_.data_ptr<long>();
  auto spatialSize = spatialSize_.data_ptr<long>();
  long size[dimension + 1]; // IntList?
  for (Int i = 0; i <= dimension; ++i)
    size[i] = tensor_.size(i); //   std::vector<long> size = tensor_.size();
  auto nPlanes = size[dimension];
  long volume = 1;
  for (Int i = 0; i < dimension; ++i)
    volume *= size[i];
  features_.resize_({(int)(nActive + volume), nPlanes});
  // Increment pointers as we work through the data
  auto features = features_.data_ptr<float>() + nActive * nPlanes;

  // Active locations
  Point<dimension> point;
  for (Int i = 0; i < dimension; i++)
    point[i] = offset[i];
  for (Int ctr = 0; ctr < volume; ctr++) {
    bool active = false;
    for (Int i = 0; i < nPlanes; i++) {
      if (fabs(tensor[i]) > threshold) {
        active = true;
        break;
      }
    }
    for (Int i = 0; i < dimension; i++) {
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
    incrementPointInCube<dimension>(point, size, offset);
  }
  features_.resize_({(int)nActive, nPlanes});
}

// 3x3 submanifold convolutions, 3x3/2x2 pooling or strided convolutions
template <Int dimension> void Metadata<dimension>::generateRuleBooks3s2() {
  long sz[dimension], str[dimension], inS[dimension], outS[dimension];
  Point<dimension> p1;
  Point<2 * dimension> p2;
  Point<3 * dimension> p3;
  for (Int i = 0; i < dimension; ++i) {
    p1[i] = p2[i] = p3[i] = inS[i] = inputSpatialSize[i];
    p2[i + dimension] = p3[i + dimension] = sz[i] = 3;
    p3[i + 2 * dimension] = str[i] = 2;
  }
  while (true) {
    auto &SGs = grids[p1];
    auto &rb = submanifoldRuleBooks[p2];
    if (rb.empty())
      SubmanifoldConvolution_SgsToRules(SGs, rb, sz);
    for (Int i = 0; i < dimension; ++i)
      if (p1[i] < 3 or p1[i] % 2 != 1)
        return;
      else
        p1[i] = outS[i] = (inS[i] - 1) / 2;
    auto &SGs2 = grids[p1];
    auto &rb2 = ruleBooks[p3];
    if (rb2.empty())
      nActive[p1] = Convolution_InputSgsToRulesAndOutputSgs(SGs, SGs2, rb2, sz,
                                                            str, inS, outS);
    for (Int i = 0; i < dimension; ++i)
      p2[i] = p3[i] = inS[i] = outS[i];
  }
}

// 3x3 submanifold convolutions, 2x2 pooling or strided convolutions
template <Int dimension> void Metadata<dimension>::generateRuleBooks2s2() {
  long s2[dimension], s3[dimension], inS[dimension], outS[dimension];
  Point<dimension> p1;
  Point<2 * dimension> p2;
  Point<3 * dimension> p3;
  for (Int i = 0; i < dimension; ++i) {
    p1[i] = p2[i] = p3[i] = inS[i] = inputSpatialSize[i];
    p2[i + dimension] = s3[i] = 3;
    p3[i + dimension] = p3[i + 2 * dimension] = s2[i] = 2;
  }
  while (true) {
    auto &SGs = grids[p1];
    auto &rb = submanifoldRuleBooks[p2];
    if (rb.empty())
      SubmanifoldConvolution_SgsToRules(SGs, rb, s3);
    for (Int i = 0; i < dimension; ++i)
      if (p1[i] < 2 or p1[i] % 2 != 0)
        return;
      else
        p1[i] = outS[i] = inS[i] / 2;
    auto &SGs2 = grids[p1];
    auto &rb2 = ruleBooks[p3];
    if (rb2.empty())
      nActive[p1] = Convolution_InputSgsToRulesAndOutputSgs(SGs, SGs2, rb2, s2,
                                                            s2, inS, outS);
    for (Int i = 0; i < dimension; ++i)
      p2[i] = p3[i] = inS[i] = outS[i];
  }
}

template <Int dimension>
void Metadata<dimension>::inputLayer(/*long*/ at::Tensor &spatialSize,
                                     /*long*/ at::Tensor &coords, Int batchSize,
                                     Int mode) {
  assert(spatialSize.ndimension() == 1);
  assert(spatialSize.size(0) == dimension);
  assert(coords.ndimension() == 2);
  assert(coords.size(1) >= dimension and coords.size(1) <= dimension + 1);
  setInputSpatialSize(spatialSize);
  inputLayerRules<dimension>(*inputSGs, inputLayerRuleBook, coords.data_ptr<long>(),
                             coords.size(0), coords.size(1), batchSize, mode,
                             *inputNActive);
}
template <Int dimension>
void Metadata<dimension>::blLayer(/*long*/ at::Tensor &spatialSize,
                                  /*long*/ at::Tensor &coords, Int mode) {
  assert(spatialSize.ndimension() == 1);
  assert(spatialSize.size(0) == dimension);
  assert(coords.ndimension() == 3);
  assert(coords.size(2) == dimension);
  setInputSpatialSize(spatialSize);
  blRules<dimension>(*inputSGs, blLayerRuleBook, coords.data_ptr<long>(),
                     coords.size(0), coords.size(1), mode, *inputNActive);
}
template <Int dimension>
RuleBook &Metadata<dimension>::getSubmanifoldRuleBook(
    /*long*/ at::Tensor &spatialSize,
    /*long*/ at::Tensor &size, bool openMP) {
  auto p = TwoLongTensorsToPoint<dimension>(spatialSize, size);
  auto &rb = submanifoldRuleBooks[p];
  if (rb.empty()) {
    auto &SGs = grids[LongTensorToPoint<dimension>(spatialSize)];
#if defined(ENABLE_OPENMP)
    openMP ? SubmanifoldConvolution_SgsToRules_OMP(SGs, rb, size.data_ptr<long>()) :
#endif
           SubmanifoldConvolution_SgsToRules(SGs, rb, size.data_ptr<long>());
  }
  return rb;
}
template <Int dimension>
RuleBook &Metadata<dimension>::getPermutohedralSubmanifoldRuleBook(
    /*long*/ at::Tensor &spatialSize, bool openMP) {
  auto p = LongTensorToPoint<dimension>(spatialSize);
  auto &rb = permutohedralRuleBooks[p];
  if (rb.empty()) {
    auto &SGs = grids[LongTensorToPoint<dimension>(spatialSize)];
#if defined(ENABLE_OPENMP)
    openMP ? PermutohedralSubmanifoldConvolution_SgsToRules_OMP(SGs, rb) :
#endif
           PermutohedralSubmanifoldConvolution_SgsToRules(SGs, rb);
  }
  return rb;
}
template <Int dimension>
RuleBook &Metadata<dimension>::getActivePoolingRuleBook(
    /*long*/ at::Tensor &spatialSize) {
  auto spatialSz = LongTensorToPoint<dimension>(spatialSize);
  auto &SGs = grids[spatialSz];
  auto &rb = activePoolingRuleBooks[spatialSz];
  if (rb.empty())
    activePoolingRules(SGs, rb);
  return rb;
}
template <Int dimension>
RuleBook &Metadata<dimension>::getSparseToDenseRuleBook(
    /*long*/ at::Tensor &spatialSize, bool openMP) {
  auto ss = LongTensorToPoint<dimension>(spatialSize);
  auto &SGs = grids[ss];
  auto &rb = sparseToDenseRuleBooks[ss];
  if (rb.empty())
#if defined(ENABLE_OPENMP)
    openMP ? SparseToDense_InputSgsToRulesAndOutputSgs_OMP(
                 SGs, rb, spatialSize.data_ptr<long>())
           :
#endif
           SparseToDense_InputSgsToRulesAndOutputSgs(SGs, rb,
                                                     spatialSize.data_ptr<long>());
  return rb;
}
template <Int dimension>
RuleBook &Metadata<dimension>::getRuleBook(
    /*long*/ at::Tensor &inputSpatialSize,
    /*long*/ at::Tensor &outputSpatialSize,
    /*long*/ at::Tensor &size,
    /*long*/ at::Tensor &stride, bool openMP) {
  auto p = ThreeLongTensorsToPoint<dimension>(inputSpatialSize, size, stride);
  auto &rb = ruleBooks[p];
  if (rb.empty()) {
    auto iS = LongTensorToPoint<dimension>(inputSpatialSize);
    auto oS = LongTensorToPoint<dimension>(outputSpatialSize);
    auto &iSGs = grids[iS];
    auto &oSGs = grids[oS];
    nActive[oS] =
#if defined(ENABLE_OPENMP)
        openMP
            ? Convolution_InputSgsToRulesAndOutputSgs_OMP(
                  iSGs, oSGs, rb, size.data_ptr<long>(), stride.data_ptr<long>(),
                  inputSpatialSize.data_ptr<long>(), outputSpatialSize.data_ptr<long>())
            :
#endif
            Convolution_InputSgsToRulesAndOutputSgs(
                iSGs, oSGs, rb, size.data_ptr<long>(), stride.data_ptr<long>(),
                inputSpatialSize.data_ptr<long>(), outputSpatialSize.data_ptr<long>());
  }
  return rb;
}
template <Int dimension>
RuleBook &Metadata<dimension>::getFullConvolutionRuleBook(
    /*long*/ at::Tensor &inputSpatialSize,
    /*long*/ at::Tensor &outputSpatialSize,
    /*long*/ at::Tensor &size,
    /*long*/ at::Tensor &stride, Metadata<dimension> &newM) {
  auto &rb = newM.fullConvolutionRuleBook;
  if (rb.empty()) {
    newM.clear();
    auto iS = LongTensorToPoint<dimension>(inputSpatialSize);
    auto oS = LongTensorToPoint<dimension>(outputSpatialSize);
    newM.grids[iS] = grids[iS]; // copy
    newM.nActive[iS] = nActive[iS];
    auto &iSGs = newM.grids[iS];
    auto &oSGs = newM.grids[oS];
    newM.nActive[oS] = FullConvolution_InputSgsToRulesAndOutputSgs_OMP(
        iSGs, oSGs, rb, size.data_ptr<long>(), stride.data_ptr<long>(),
        inputSpatialSize.data_ptr<long>(), outputSpatialSize.data_ptr<long>());
  }
  return rb;
}

template <Int dimension>
RuleBook &Metadata<dimension>::getRandomizedStrideRuleBook(
    /*long*/ at::Tensor &inputSpatialSize,
    /*long*/ at::Tensor &outputSpatialSize,
    /*long*/ at::Tensor &size,
    /*long*/ at::Tensor &stride, bool openMP) {
  auto p = ThreeLongTensorsToPoint<dimension>(inputSpatialSize, size, stride);
  auto &rb = ruleBooks[p];
  if (rb.empty()) {
    auto iS = LongTensorToPoint<dimension>(inputSpatialSize);
    auto oS = LongTensorToPoint<dimension>(outputSpatialSize);
    auto &iSGs = grids[iS];
    auto &oSGs = grids[oS];
    nActive[oS] =
#if defined(ENABLE_OPENMP)
        openMP
            ? RSR_InputSgsToRulesAndOutputSgs_OMP(
                  iSGs, oSGs, rb, size.data_ptr<long>(), stride.data_ptr<long>(),
                  inputSpatialSize.data_ptr<long>(), outputSpatialSize.data_ptr<long>(),
                  re)
            :
#endif
            RSR_InputSgsToRulesAndOutputSgs(iSGs, oSGs, rb, size.data_ptr<long>(),
                                            stride.data_ptr<long>(),
                                            inputSpatialSize.data_ptr<long>(),
                                            outputSpatialSize.data_ptr<long>(), re);
  }
  return rb;
}

at::Tensor vvl2t(std::vector<std::vector<long>> v) {
  long s = 0;
  for (auto &x : v)
    s += x.size();
  at::Tensor t = torch::empty({s}, at::CPU(at::kLong));
  long *p = t.data_ptr<long>();
  for (auto &x : v) {
    std::memcpy(p, &x[0], x.size() * sizeof(long));
    p += x.size();
  }
  return t;
}

template <Int dimension>
std::vector<at::Tensor>
Metadata<dimension>::compareSparseHelper(Metadata<dimension> &mR,
                                         /* long */ at::Tensor &spatialSize) {
  auto p = LongTensorToPoint<dimension>(spatialSize);
  auto &sgsL = grids[p];
  auto &sgsR = mR.grids[p];
  Int bs = sgsL.size(), sample;
  std::vector<std::vector<long>> cL(bs), cR(bs), L(bs), R(bs);
#pragma omp parallel for private(sample)
  for (sample = 0; sample < bs; ++sample) {
    auto &sgL = sgsL[sample];
    auto &sgR = sgsR[sample];
    auto &cLs = cL[sample];
    auto &cRs = cR[sample];
    auto &Ls = L[sample];
    auto &Rs = R[sample];
    for (auto const &iter : sgL.mp) {
      if (sgR.mp.find(iter.first) == sgR.mp.end()) {
        Ls.push_back(sgL.mp[iter.first] + sgL.ctr);
      } else {
        cLs.push_back(sgL.mp[iter.first] + sgL.ctr);
        cRs.push_back(sgR.mp[iter.first] + sgR.ctr);
      }
    }
    for (auto const &iter : sgR.mp) {
      if (sgL.mp.find(iter.first) == sgL.mp.end()) {
        Rs.push_back(sgR.mp[iter.first] + sgR.ctr);
      }
    }
  }
  return {vvl2t(cL), vvl2t(cR), vvl2t(L), vvl2t(R)};
}

at::Tensor vvl2t_(std::vector<std::vector<Int>> v) {
  long s = 0;
  for (auto &x : v)
    s += x.size();
  at::Tensor t = torch::empty({s}, at::CPU(at_kINT));
  Int *p = t.data_ptr<Int>();
  for (auto &x : v) {
    std::memcpy(p, &x[0], x.size() * sizeof(Int));
    p += x.size();
  }
  return t;
}

template <Int dimension>
at::Tensor
Metadata<dimension>::copyFeaturesHelper(Metadata<dimension> &mR,
                                        /* long */ at::Tensor &spatialSize) {
  auto p = LongTensorToPoint<dimension>(spatialSize);
  auto &sgsL = grids[p];
  auto &sgsR = mR.grids[p];
  Int bs = sgsL.size(), sample;
  std::vector<std::vector<Int>> r(bs);
#pragma omp parallel for private(sample)
  for (sample = 0; sample < bs; ++sample) {
    auto &sgL = sgsL[sample];
    auto &sgR = sgsR[sample];
    auto &rs = r[sample];
    for (auto const &iter : sgL.mp) {
      if (sgR.mp.find(iter.first) != sgR.mp.end()) {
        rs.push_back(sgL.mp[iter.first] + sgL.ctr);
        rs.push_back(sgR.mp[iter.first] + sgR.ctr);
      }
    }
  }
  return vvl2t_(r);
}
template <Int dimension> Int volume(long *point) {
  Int v = 1;
  for (Int i = 0; i < dimension; i++)
    v *= point[i];
  return v;
}
