// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef SPARSECONVNET_H
#define SPARSECONVNET_H

// To use 64 bits instead of 32, replace 32bits.h with 64bits.h
#include "32bits.h"
#include <array>
#include <cstdint>
#include <google/dense_hash_map>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>
#if defined(ENABLE_OPENMP)
#include <omp.h>
#endif

// Submanifold Sparse Convolutional Networks

// A batch of samples, for each layer of a sparse convolutional network, is
// encoded as a matrix of nActive x nFeatures and a vector of
// hash tables identifying points in space with the rows of
// the matrix.

// SparseGridMap<dimension> - a hash table assigning integer labels to a sparse
// collection of 'Point<dimension>' points

template <uInt dimension>
using SparseGridMap =
    google::dense_hash_map<Point<dimension>, int, IntArrayHash<dimension>,
                           std::equal_to<Point<dimension>>>;

template <uInt dimension> class SparseGrid {
public:
  uInt ctr; // Count #active sites during output hash construction. Then store
            // offset within a batch.
  SparseGridMap<dimension> mp;
  SparseGrid() : ctr(0) {
    // Sparsehash needs a key to be set aside and never used - we use
    // (Int_MAX,...,Int_MAX)
    Point<dimension> empty_key;
    for (uInt i = 0; i < dimension; ++i)
      empty_key[i] = Int_MAX;
    mp.set_empty_key(empty_key);
  }
};

template <uInt dimension>
using SparseGrids = std::vector<SparseGrid<dimension>>;

// Each convolution/pooling operation requires the calculation of a 'rulebook'
// setting out how the output points depend on the points in the layer below
using RuleBook = std::vector<std::vector<uInt>>;

// Code relating to squares/cubes/rectangles/cuboids etc
// integer powers - ok for filter sizes, could overflow if we calculate
// inputSpatialSize^d
template <uInt m> uInt ipow(uInt n) { return n * ipow<m - 1>(n); }
template <> uInt ipow<1>(uInt n) { return n; }
template <> uInt ipow<0>(uInt n) { return 1; }

template <uInt dimension> uInt volume(long *point) {
  uInt v = 1;
  for (uInt i = 0; i < dimension; i++)
    v *= point[i];
  return v;
}

// Macro to initialize arguments passed as void*[1] from Lua.
// This allows Lua to take ownership of arbitrary C++ objects.
// The macro:
// - takes a pointer to a pointer [allocated as ffi.new('void *[1]') in Lua]
// - if the pointer has not yet been initialized, create an object for it
// - create a reference "_VAR" to the object

#define SCN_INITIALIZE_AND_REFERENCE(TYPE, VAR)                                \
  if (VAR[0] == NULL)                                                          \
    VAR[0] = (void *)new TYPE;                                                 \
  TYPE &_##VAR = *(TYPE *)VAR[0];

// Macro to free the memory allocated by SCN_INITIALIZE_AND_REFERENCE

#define SCN_DELETE(TYPE, VAR)                                                  \
  if (VAR[0] != NULL) {                                                        \
    delete (TYPE *) VAR[0];                                                    \
    VAR[0] = NULL;                                                             \
  }

uInt ruleBookMaxSize(RuleBook &rb) {
  uInt m = 0;
  for (auto &r : rb)
    m = std::max(m, (uInt)r.size());
  return m;
}
uInt ruleBookTotalSize(RuleBook &rb) {
  uInt m = 0;
  for (auto &r : rb)
    m += (uInt)r.size();
  return m;
}

#endif /* SPARSECONVNET_H */
