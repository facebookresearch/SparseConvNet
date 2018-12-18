// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef RECTANGULARREGIONS_H
#define RECTANGULARREGIONS_H


// For iterating over the rectangular region with corners lb and ub.
// The .end() method and operator!= are designed to allow range based for
// loops of the region, but nothing else.

template <Int dimension> class RectangularRegionIterator;
template <Int dimension> class RectangularRegion {
public:
  Point<dimension> lb;
  Point<dimension> ub;
  RectangularRegion(Point<dimension> &lb, Point<dimension> &ub)
      : lb(lb), ub(ub) {}
  RectangularRegionIterator<dimension> begin() {
    return RectangularRegionIterator<dimension>(*this, lb);
  }
  RectangularRegionIterator<dimension> end() {
    // Not really used by the custom operator!= function
    // Otherwise it would need to represent a point just outside the region
    return RectangularRegionIterator<dimension>(*this, ub);
  }
  Int
  offset(const Point<dimension> &p) { // Enumerate the points inside the region
    Int of = 0, m = 1;
    for (Int i = dimension - 1; i >= 0; i--) {
      of += m * (p[i] - lb[i]);
      m *= ub[i] - lb[i] + 1;
    }
    return of;
  }
};

template <Int dimension> class RectangularRegionIterator {
private:
  RectangularRegion<dimension> &region;

public:
  Point<dimension> point;
  bool stillLooping;
  RectangularRegionIterator(RectangularRegion<dimension> &region,
                            Point<dimension> &point)
      : region(region), point(point), stillLooping(true) {
    // If stride > size, we can have lb[i]>ub[i] meaning region_size == 0
    for (Int i = 0; i < dimension; i++)
      if (point[i] > region.ub[i])
        stillLooping = false;
  }
  RectangularRegionIterator<dimension> &operator++() {

    for (Int i = dimension - 1;;) {
      point[i]++;
      if (point[i] <= region.ub[i])
        break;
      point[i] = region.lb[i];
      i--;
      if (i == -1) {
        stillLooping = false; // Signal to operator!= to end iteration
        break;
      }
    }

    return *this;
  }
  Point<dimension> &operator*() { return point; }
};

// Only to be used for checking the end point of range based for loops.
template <Int dimension>
inline bool operator!=(const RectangularRegionIterator<dimension> &lhs,
                       const RectangularRegionIterator<dimension> &rhs) {
  return lhs.stillLooping;
}

// Similar to above but for [ offset[0] ... offset[0]+size[0]-1 ] x ... x [..]
template <Int dimension>
void incrementPointInCube(Point<dimension> &point, long *size, long *offset) {
  for (Int i = dimension - 1; i >= 0; i--) {
    point[i]++;
    if (point[i] < offset[i] + size[i])
      break;
    point[i] = offset[i];
  }
}

// For a convolutional layer with given filter *size* and *stride*, find the
// subset of the input field corresponding to a point in the output.
template <Int dimension>
RectangularRegion<dimension>
InputRegionCalculator(const Point<dimension> &output, long *size,
                      long *stride) {
  Point<dimension> lb, ub;
  for (Int i = 0; i < dimension; i++) {
    lb[i] = output[i] * stride[i];
    ub[i] = output[i] * stride[i] + size[i] - 1;
  }
  return RectangularRegion<dimension>(lb, ub);
}

// For a convolutional layer with given filter *size* and *stride*, find the
// subset of the output field corresponding to a point in the input.
template <Int dimension>
RectangularRegion<dimension>
OutputRegionCalculator(const Point<dimension> &input, long *size, long *stride,
                       long *outputSpatialSize) {
  Point<dimension> lb, ub;
  for (Int i = 0; i < dimension; i++) {
    lb[i] = std::max(0L, (input[i] - size[i] + stride[i]) / stride[i]);
    ub[i] = std::min(outputSpatialSize[i] - 1, input[i] / stride[i]);
  }
  return RectangularRegion<dimension>(lb, ub);
}

#endif /* RECTANGULARREGIONS_H */
