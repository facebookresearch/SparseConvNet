// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef Metadata_H
#define Metadata_H
#include "32bits.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include "sparsehash/dense_hash_map"
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

template <Int dimension>
using SparseGridMap =
    google::dense_hash_map<Point<dimension>, Int, IntArrayHash<dimension>,
                           std::equal_to<Point<dimension>>>;
template <Int dimension> class SparseGrid {
public:
  Int ctr;
  SparseGridMap<dimension> mp;
  SparseGrid();
};
template <Int dimension> using SparseGrids = std::vector<SparseGrid<dimension>>;
using RuleBook = std::vector<std::vector<Int>>;

template <Int dimension>
void addPointToSparseGridMapAndFeatures(SparseGridMap<dimension> &mp,
                                        Point<dimension> p, Int &nActive,
                                        long nPlanes,
                                        /*float*/ at::Tensor &features,
                                        float *vec, bool overwrite);

template <Int dimension> class Metadata {
public:
  // Count of active sites for each scale
  std::unordered_map<Point<dimension>, Int, IntArrayHash<dimension>> nActive;

  // Hash tables for each scale locating the active points
  std::unordered_map<Point<dimension>, SparseGrids<dimension>,
                     IntArrayHash<dimension>>
      grids;

  std::unordered_map<Point<dimension>, RuleBook, IntArrayHash<dimension>>
      activePoolingRuleBooks;

  RuleBook inputLayerRuleBook;
  RuleBook blLayerRuleBook;

  std::unordered_map<Point<2 * dimension>, RuleBook,
                     IntArrayHash<2 * dimension>>
      submanifoldRuleBooks;

  std::unordered_map<Point<dimension>, RuleBook, IntArrayHash<dimension>>
      permutohedralRuleBooks;

  std::unordered_map<Point<3 * dimension>, RuleBook,
                     IntArrayHash<3 * dimension>>
      ruleBooks;

  RuleBook fullConvolutionRuleBook;

  std::unordered_map<Point<dimension>, RuleBook, IntArrayHash<dimension>>
      sparseToDenseRuleBooks;

  Point<dimension> inputSpatialSize;
  SparseGrids<dimension> *inputSGs;
  SparseGrid<dimension> *inputSG;
  Int *inputNActive;
  std::default_random_engine re;

  Metadata();
  void clear();
  Int getNActive(/*long*/ at::Tensor &spatialSize);
  SparseGrids<dimension> &getSparseGrid(/*long*/ at::Tensor &spatialSize);
  void setInputSpatialSize(/*long*/ at::Tensor &spatialSize);
  void batchAddSample();
  void setInputSpatialLocation(/*float*/ at::Tensor &features,
                               /*long*/ at::Tensor &location,
                               /*float*/ at::Tensor &vec, bool overwrite);
  void setInputSpatialLocations(/*float*/ at::Tensor &features,
                                /*long*/ at::Tensor &locations,
                                /*float*/ at::Tensor &vecs, bool overwrite);

  at::Tensor getSpatialLocations(/*long*/ at::Tensor &spatialSize);
  Int getBatchSize(/*long*/ at::Tensor &spatialSize);
  void createMetadataForDenseToSparse(/*long*/ at::Tensor &spatialSize,
                                      /*long*/ at::Tensor &nz_, long batchSize);

  void sparsifyMetadata(Metadata<dimension> &mOut,
                        /*long*/ at::Tensor &spatialSize,
                        /*byte*/ at::Tensor &filter,
                        /*long*/ at::Tensor &cuSum);

  void appendMetadata(Metadata<dimension> &mAdd,
                      /*long*/ at::Tensor &spatialSize);

  /* std::vector<at::Tensor &> sparsifyCompare(Metadata<dimension> &mReference,
   */
  /*                                         Metadata<dimension> &mSparsified,
   */
  /*                                         /\*long*\/ at::Tensor &
   * spatialSize);
   */

  std::vector<at::Tensor> sparsifyCompare(Metadata<dimension> &mReference,
                                            /*long*/ at::Tensor &spatialSize);

  // tensor is size[0] x .. x size[dimension-1] x size[dimension]
  // size[0] x .. x size[dimension-1] == spatial volume
  // size[dimension] == #feature planes
  void addSampleFromThresholdedTensor(/*float*/ at::Tensor &features_,
                                      /*float*/ at::Tensor &tensor_,
                                      /*long*/ at::Tensor &offset_,
                                      /*long*/ at::Tensor &spatialSize_,
                                      float threshold);

  // 3x3 submanifold convolutions, 3x3/2x2 pooling or strided convolutions
  void generateRuleBooks3s2();

  // 3x3 submanifold convolutions, 2x2 pooling or strided convolutions
  void generateRuleBooks2s2();

  void inputLayer(/*long*/ at::Tensor &spatialSize,
                  /*long*/ at::Tensor &coords, Int batchSize, Int mode);
  void blLayer(/*long*/ at::Tensor &spatialSize, /*long*/ at::Tensor &coords,
               Int mode);
  RuleBook &getSubmanifoldRuleBook(/*long*/ at::Tensor &spatialSize,
                                   /*long*/ at::Tensor &size, bool openMP);
  RuleBook &
  getPermutohedralSubmanifoldRuleBook(/*long*/ at::Tensor &spatialSize,
                                      bool openMP);
  RuleBook &getActivePoolingRuleBook(/*long*/ at::Tensor &spatialSize);
  RuleBook &getSparseToDenseRuleBook(/*long*/ at::Tensor &spatialSize,
                                     bool openMP);
  RuleBook &getRuleBook(/*long*/ at::Tensor &inputSpatialSize,
                        /*long*/ at::Tensor &outputSpatialSize,
                        /*long*/ at::Tensor &size,
                        /*long*/ at::Tensor &stride, bool openMP);
  RuleBook &getFullConvolutionRuleBook(/*long*/ at::Tensor &inputSpatialSize,
                                       /*long*/ at::Tensor &outputSpatialSize,
                                       /*long*/ at::Tensor &size,
                                       /*long*/ at::Tensor &stride,
                                       Metadata<dimension> &newM);

  RuleBook &getRandomizedStrideRuleBook(/*long*/ at::Tensor &inputSpatialSize,
                                        /*long*/ at::Tensor &outputSpatialSize,
                                        /*long*/ at::Tensor &size,
                                        /*long*/ at::Tensor &stride,
                                        bool openMP);

  std::vector<at::Tensor >
  compareSparseHelper(Metadata<dimension> &mR,
                      /* long */ at::Tensor &spatialSize);
  at::Tensor copyFeaturesHelper(Metadata<dimension> &mR,
                                 /* long */ at::Tensor &spatialSize);
};

template <typename T> T *OptionalTensorData(at::Tensor &tensor);

template <Int dimension> Int volume(long *point);
#endif
