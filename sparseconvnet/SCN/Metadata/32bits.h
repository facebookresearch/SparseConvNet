// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <array>

// Using 32 bit integers for coordinates and memory calculations.

using Int = int32_t;

// Folly's twang_mix64 hashing function
inline uint64_t twang_mix64(uint64_t key) noexcept {
  key = (~key) + (key << 21); // key *= (1 << 21) - 1; key -= 1;
  key = key ^ (key >> 24);
  key = key + (key << 3) + (key << 8); // key *= 1 + (1 << 3) + (1 << 8)
  key = key ^ (key >> 14);
  key = key + (key << 2) + (key << 4); // key *= 1 + (1 << 2) + (1 << 4)
  key = key ^ (key >> 28);
  key = key + (key << 31); // key *= 1 + (1 << 31)
  return key;
}

// Point<dimension> is a point in the d-dimensional integer lattice
// (i.e. square-grid/cubic-grid, ...)

template <Int dimension> using Point = std::array<Int, dimension>;

template<Int dimension>
Point<dimension> generateEmptyKey() {
  Point<dimension> empty_key;
  for (Int i = 0; i < dimension; ++i)
        empty_key[i] = std::numeric_limits<Int>::min();
  return empty_key;
}

template <Int dimension>
Point<dimension> LongTensorToPoint(/*long*/ at::Tensor &t) {
  Point<dimension> p;
  long *td = t.data<long>();
  for (Int i = 0; i < dimension; i++)
    p[i] = td[i];
  return p;
}
template <Int dimension>
Point<2 * dimension> TwoLongTensorsToPoint(/*long*/ at::Tensor &t0,
                                           /*long*/ at::Tensor &t1) {
  Point<2 * dimension> p;
  long *td;
  td = t0.data<long>();
  for (Int i = 0; i < dimension; i++)
    p[i] = td[i];
  td = t1.data<long>();
  for (Int i = 0; i < dimension; i++)
    p[i + dimension] = td[i];
  return p;
}
template <Int dimension>
Point<3 * dimension> ThreeLongTensorsToPoint(/*long*/ at::Tensor &t0,
                                             /*long*/ at::Tensor &t1,
                                             /*long*/ at::Tensor &t2) {
  Point<3 * dimension> p;
  long *td;
  td = t0.data<long>();
  for (Int i = 0; i < dimension; i++)
    p[i] = td[i];
  td = t1.data<long>();
  for (Int i = 0; i < dimension; i++)
    p[i + dimension] = td[i];
  td = t2.data<long>();
  for (Int i = 0; i < dimension; i++)
    p[i + 2 * dimension] = td[i];
  return p;
}

// FNV Hash function for Point<dimension>
template <Int dimension> struct IntArrayHash {
  std::size_t operator()(Point<dimension> const &p) const {
    Int hash = 16777619;
    for (auto x : p) {
      hash *= 2166136261;
      hash ^= x;
    }
    return hash;
  }
};

// FNV Hash function for Point<dimension>
template <Int dimension> struct FastHash {
  std::size_t operator()(Point<dimension> const &p) const {
    std::size_t seed = 16777619;
    
    for (auto x : p) {

      // from boost
      seed ^= twang_mix64(x) + 0x9e3779b9 + (seed<<6) + (seed>>2);
    }

    return seed;
  }
};


#define at_kINT at::kInt
