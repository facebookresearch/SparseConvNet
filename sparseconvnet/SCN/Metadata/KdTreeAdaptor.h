#ifdef DICT_KD_TREE

#ifndef Kd_Tree_Adaptor_H
#define Kd_Tree_Adaptor_H

#include "32bits.h"
#include "nanoflann.hpp"
#include <google/dense_hash_map>
#include <map>
#include <torch/extension.h>

template <Int dimension> struct PointContainer {
  using VectorIndex = int;

  google::dense_hash_map<Point<dimension>, VectorIndex, IntArrayHash<dimension>,
                         std::equal_to<Point<dimension>>>
      data;
  std::vector<std::pair<Point<dimension>, Int>> vec;

  PointContainer() { data.set_empty_key(generateEmptyKey<dimension>()); }

  /* Following methods are required in order to work with nanoflann.ghpp */
  inline size_t kdtree_get_point_count() const { return data.size(); }

  inline Int kdtree_get_pt(const size_t idx, const size_t dim) const {
    return vec[idx].first[dim];
  }

  template <class BBOX> bool kdtree_get_bbox(BBOX & /* bb */) const {
    return false;
  }
};

template <Int dimension> class KdTreeContainer {

  using it_t = typename std::vector<std::pair<Point<dimension>, Int>>::iterator;

  using index_t = nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<Int, PointContainer<dimension>>,
      PointContainer<dimension>, dimension>;

public:
  KdTreeContainer() { points = PointContainer<dimension>(); }

  KdTreeContainer(const KdTreeContainer &k) : points(k.points) {
    if (k.kdTreeIndex) {
      init();
    }
  }

  void operator=(const KdTreeContainer &k) {
    points = k.points;
    if (k.kdTreeIndex) {
      init();
    } else {
      kdTreeIndex.reset();
    }
  }

  // Build the kd-tree given the points prestent in the point container
  // strcuture.
  void init() {
    kdTreeIndex = std::make_unique<index_t>(
        dimension, points, nanoflann::KDTreeSingleIndexAdaptorParams(leafSize));
    kdTreeIndex->buildIndex();
  }

  size_t size() const { return points.data.size(); }

  void setLeafSize(long *sz) {
    if (!sz)
      throw "Invalid filter size!";

    int acc = 1;
    long *ptr = sz;
    for (int i = 0; i < dimension; ++i, ++ptr)
      acc *= *ptr;

    leafSize = acc;
  }

  // Given a point and a radius, finds all the points within the radius of the
  // point. Since we are
  // using the L2 adaptor, the radius should be squared.
  const std::vector<std::pair<size_t, Int>>
  search(const Point<dimension> &point, Int radius) {
    std::vector<std::pair<size_t, Int>> ret_index;
    nanoflann::SearchParams params;

    kdTreeIndex->radiusSearch(&point[0], radius, ret_index, params);

    return ret_index;
  }

  /* Following are the functions to remain compliant with the map like interface
     for
      sparse grid maps. */

  std::pair<Point<dimension>, Int> getIndexPointData(int index) {
    return points.vec[index];
  }

  std::pair<it_t, bool>
  insert(const std::pair<Point<dimension>, Int> &mapElem) {
    auto mapRes =
        points.data.insert(make_pair(mapElem.first, points.vec.size() - 1));

    if (mapRes.second) {
      points.vec.push_back(mapElem);
      return make_pair(std::prev(points.vec.end()), true);
    }

    // Return the iterator to the element in the vector since it is present in
    // the map.
    return make_pair(points.vec.begin() + mapRes.first->second, false);
  }

  int count(const Point<dimension> &point) const {
    return points.data.count(point);
  }

  it_t begin() { return points.vec.begin(); }

  it_t end() { return points.vec.end(); }

  it_t find(const Point<dimension> &p) {
    auto it = points.data.find(p);

    if (it == points.data.end()) {
      return points.vec.end();
    }

    return points.vec.begin() + it->second;
  }

  Int &operator[](Point<dimension> point) {
    return insert(make_pair(point, 0)).first->second;
  }

private:
  PointContainer<dimension> points;
  std::unique_ptr<index_t> kdTreeIndex;
  int leafSize = 100;
};

#endif
#endif
