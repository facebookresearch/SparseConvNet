#ifndef Container_Map_H
#define Container_Map_H

#include "32bits.h"
#include "RectangularRegions.h"
#include <algorithm>
#include <cmath>
#include <google/dense_hash_map>
#include <memory>

template <Int dimension> class ContainerMapIterator;

template <Int dimension>
using inner_map_t = std::vector<std::pair<Point<dimension>, int>>;

template <Int dimension>
using outer_map_t =
    google::dense_hash_map<Point<dimension>, inner_map_t<dimension>,
                           IntArrayHash<dimension>,
                           std::equal_to<Point<dimension>>>;

template <Int dimension> class ContainerMap {
  using it_t = ContainerMapIterator<dimension>;
  using RuleBook = std::vector<std::vector<Int>>;
  using RegionIndex = Point<dimension>;

public:
  ContainerMap() {
    const static int default_size = 3;
    map = getNewGoogleMap<outer_map_t<dimension>>();
    filterDimensions.fill(default_size);
    volume = std::pow(default_size, dimension);
  }

  void setFilterDimensions(const long *szs) {
    std::copy_n(szs, dimension, filterDimensions.data());
    volume = 1;
    for (long size : filterDimensions) {
      volume *= size;
    }
  }

  /* Uses applyOnOverlappingRegions to only operate on active points of the map.
    Due to disregarding the rest of the points within the region, the number of
    map accesses is expected to decrease. */
  template <typename RulesFunc>
  void populateActiveSites(const RectangularRegion<dimension> &region,
                           RulesFunc rulesFunc) {
    applyOnOverlappingRegions(region, [&](const RegionIndex &regionIndex) {
      auto *container = getContainerByRegionIndex(regionIndex);
      if (container) {
        auto bounds = getRegionBounds(regionIndex);
        RectangularRegion<dimension> gridRegion =
            RectangularRegion<dimension>(bounds.first, bounds.second);
        RectangularRegion<dimension> overlap =
            getOverlappingRegion(gridRegion, region);

        for (auto &point : overlap) {
          int vectorIndex = gridRegion.offset(point);
          if ((*container)[vectorIndex].second != -1) {
            rulesFunc((*container)[vectorIndex]);
          }
        }
      }
    });
  }

  /* Uses applyOnOverlappingRegions to populate overlapping points.
     Used mainly for convolutions and full convolutions. If the
     output region is greater than 1x1x1, the number of map accesses is
     expected to decrease. */
  template <typename RulesFunc>
  void populateBlock(RectangularRegion<dimension> &region,
                     RulesFunc rulesFunc) {

    applyOnOverlappingRegions(region, [&](const RegionIndex &regionIndex) {
      auto &container = insertContainer(regionIndex);
      auto regionBounds = getRegionBounds(regionIndex);
      RectangularRegion<dimension> gridRegion =
          RectangularRegion<dimension>(regionBounds.first, regionBounds.second);
      RectangularRegion<dimension> overlap =
          getOverlappingRegion(gridRegion, region);
 
      for (auto &point : overlap) {
        int vectorIndex = gridRegion.offset(point);
        auto &elem = container[vectorIndex];

        if (elem.second == -1) {
          elem.first = point;
          ++ctr;
        }

        rulesFunc(elem);
      }
    });
  }

  /* The following functions are to comply with the standard hash
     map interface (begin, end, insert, find, operator[]). */
  it_t begin() { return ContainerMapIterator<dimension>(map); }

  it_t end() {
    auto container = begin();
    container.forwardToEnd();
    return container;
  }

  std::pair<it_t, bool>
  insert(const std::pair<Point<dimension>, Int> &mapElem) {
    const Point<dimension> &p = mapElem.first;
    auto index = getRegionIndex(p);
    auto outerMapIt =
        map.insert(std::make_pair(index, getEmptyInnerMap())).first;

    for (auto it = outerMapIt->second.begin(); it != outerMapIt->second.end();
         ++it) {
      if (it->second != -1 && it->first == p) {
        return std::make_pair(
            ContainerMapIterator<dimension>(outerMapIt, it, map), false);
      }
    }

    ++ctr;
    auto regionBounds = getRegionBounds(index);
    int vectorIndex = offset(p, regionBounds.first, regionBounds.second);
    outerMapIt->second[vectorIndex] = mapElem;

    return std::make_pair(
        ContainerMapIterator<dimension>(
            outerMapIt, outerMapIt->second.begin() + vectorIndex, map),
        true);
  }

  it_t find(const Point<dimension> &p) {
    auto index = getRegionIndex(p);
    auto it = map.find(index);

    if (it == map.end()) {
      return this->end();
    }

    auto regionBounds = getRegionBounds(index);
    int vectorIndex = offset(p, regionBounds.first, regionBounds.second);
    if (it->second[vectorIndex].second != -1) {
      return ContainerMapIterator<dimension>(
          it, it->second.begin() + vectorIndex, map);
    }

    return end();
  }

  int count(const Point<dimension> &p) const {
    auto index = getRegionIndex(p);
    auto it = map.find(index);

    if (it == map.end()) {
      return 0;
    }

    auto regionBounds = getRegionBounds(index);
    int vectorIndex = offset(p, regionBounds.first, regionBounds.second);

    if (it->second[vectorIndex].second != -1) {
      return 1;
    }

    return 0;
  }

  Int &operator[](const Point<dimension> point) {
    return insert(make_pair(point, 0)).first->second;
  }

  size_t size() const { return ctr; }

private:
  /* Helper to easily initialise google style dense hash maps. */
  template <typename T> static T getNewGoogleMap() {
    T map;
    map.set_empty_key(generateEmptyKey<dimension>());
    return map;
  }

  const inner_map_t<dimension> *
  getContainerByRegionIndex(const RegionIndex &index) const {
    auto it = map.find(index);
    if (it == map.end()) {
      return nullptr;
    }

    return &it->second;
  }

  RegionIndex getRegionIndex(const Point<dimension> &p) const {
    Point<dimension> res;

    for (int i = 0; i < dimension; ++i) {
      res[i] = p[i] / filterDimensions[i];
    }

    return res;
  }

  Int offset(const Point<dimension> &p, const Point<dimension> &lb,
             const Point<dimension> &ub) const {
    Int of = 0, m = 1;
    for (Int i = dimension - 1; i >= 0; --i) {
      of += m * (p[i] - lb[i]);
      m *= ub[i] - lb[i] + 1;
    }
    return of;
  }

  inner_map_t<dimension> getEmptyInnerMap() const {
    auto constant_value = make_pair(generateEmptyKey<dimension>(), -1);
    auto vec = inner_map_t<dimension>(volume, constant_value);

    return vec;
  }

  /* Given an arbitrary region, this function iterates over all the grid
  regions in ContainerMap which overlap with the region. Once it finds one,
  it applies the regionFunc on it. */
  template <typename RegionFunc>
  void applyOnOverlappingRegions(const RectangularRegion<dimension> &region,
                                 RegionFunc regionFunc) {
    RegionIndex lowerRegionIndex = getRegionIndex(region.lb);
    RegionIndex upperRegionIndex = getRegionIndex(region.ub);
    std::vector<Point<dimension>> queries;

    int size = 1;
    queries.push_back(lowerRegionIndex);
    regionFunc(lowerRegionIndex);

    for (int i = 0; i < dimension; ++i) {
      if (lowerRegionIndex[i] != upperRegionIndex[i]) {
        for (int query = 0; query < size; ++query) {
          Point<dimension> next_query = queries[query]; // copy the array
          ++next_query[i];
          regionFunc(next_query);
          queries.push_back(next_query);
        }
        size *= 2;
      }
    }
  }

  inner_map_t<dimension> &insertContainer(const RegionIndex &index) {
    return map.insert(std::make_pair(index, getEmptyInnerMap())).first->second;
  }

  std::pair<Point<dimension>, Point<dimension>>
  getRegionBounds(const RegionIndex &index) const {
    std::pair<Point<dimension>, Point<dimension>> bounds;
    Point<dimension> &lower = bounds.first;
    Point<dimension> &upper = bounds.second;

    for (int i = 0; i < dimension; ++i) {
      lower[i] = index[i] * filterDimensions[i];
      upper[i] = lower[i] + filterDimensions[i] - 1;
    }

    return bounds;
  }

  RectangularRegion<dimension>
  getOverlappingRegion(const RectangularRegion<dimension> &region1,
                       const RectangularRegion<dimension> &region2) const {
    Point<dimension> lower;
    Point<dimension> upper;

    for (int i = 0; i < dimension; ++i) {
      lower[i] = std::max(region1.lb[i], region2.lb[i]);
      upper[i] = std::min(region1.ub[i], region2.ub[i]);
    }

    return RectangularRegion<dimension>(lower, upper);
  }

  outer_map_t<dimension> map;
  std::array<long, dimension> filterDimensions;
  size_t ctr = 0;
  long volume;
};

/* Custom iterator for the ContainerMap class. Captures the state of the
  iterations via two iterators. One respresenting the outer iterator
  of container maps, and one representing the inner iterator of the
  structures within the outer container map. Stores a pointer to
  the map for the purposes of end() comparisons. */
template <Int dimension> class ContainerMapIterator {
public:
  ContainerMapIterator(outer_map_t<dimension> &p) : map(&p) {
    outer_it = p.begin();

    if (outer_it != p.end()) {
      inner_it = outer_it->second.begin();
      while (inner_it->second == -1) {
        ++inner_it;
      } // there must be at least one active point in the vector
    }
  }

  ContainerMapIterator(typename outer_map_t<dimension>::iterator o_it,
                       typename inner_map_t<dimension>::iterator i_it,
                       outer_map_t<dimension> &p)
      : outer_it(o_it), inner_it(i_it), map(&p) {}

  void forwardToEnd() { outer_it = map->end(); }

  const ContainerMapIterator<dimension> &operator++() {

    while ((++inner_it != outer_it->second.end()) && (inner_it->second == -1));

    if (inner_it == outer_it->second.end()) {
      ++outer_it;

      if (outer_it != map->end()) {
        inner_it = outer_it->second.begin();
        while (inner_it->second == -1) {
          ++inner_it;
        }
      }
    }

    return *this;
  }

  std::pair<Point<dimension>, Int> &operator*() { return *inner_it; }

  const std::pair<Point<dimension>, Int> &operator*() const {
    return *inner_it;
  }

  const typename inner_map_t<dimension>::iterator operator->() const {
    return inner_it;
  }

  typename inner_map_t<dimension>::iterator operator->() { return inner_it; }

  bool operator==(const ContainerMapIterator<dimension> &it) const {
    bool end1 = outer_it == map->end();
    bool end2 = it.outer_it == map->end();

    if (end1 && end2) {
      return true;
    }

    if (end1 || end2) {
      return false;
    }

    return outer_it == it.outer_it && inner_it == it.inner_it;
  }

  bool operator!=(const ContainerMapIterator<dimension> &it) const {
    return !(*this == it);
  }

private:
  typename outer_map_t<dimension>::iterator outer_it;
  typename inner_map_t<dimension>::iterator inner_it;
  outer_map_t<dimension> *map;
};

#endif
