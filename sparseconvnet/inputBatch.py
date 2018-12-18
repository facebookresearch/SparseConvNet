# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from .metadata import Metadata
from .utils import toLongTensor
from .sparseConvNetTensor import SparseConvNetTensor


class InputBatch(SparseConvNetTensor):
    def __init__(self, dimension, spatial_size):
        SparseConvNetTensor.__init__(self, None, None, spatial_size)
        self.dimension = dimension
        self.spatial_size = toLongTensor(dimension, spatial_size)
        self.features = torch.FloatTensor()
        self.metadata = Metadata(dimension)
        self.metadata.setInputSpatialSize(self.spatial_size)

    def add_sample(self):
        self.metadata.batchAddSample()

    def set_location(self, location, vector, overwrite=False):
        assert location.min() >= 0 and (self.spatial_size - location).min() > 0
        self.metadata.setInputSpatialLocation(
            self.features, location.contiguous(), vector.contiguous(), overwrite)

    def set_location_(self, location, vector, overwrite=False):
        self.metadata.setInputSpatialLocation(
            self.features, location, vector, overwrite)

    def set_locations(self, locations, vectors, overwrite=False):
        """
        To set n locations in d dimensions, locations can be
        - A size (n,d) LongTensor, giving d-dimensional coordinates -- points
          are added to the current sample, or
        - A size (n,d+1) LongTensor; the extra column specifies the sample
          number (within the minibatch of samples).

          Example with d==3 and n==2:
          Set
          locations = LongTensor([[1,2,3],
                                  [4,5,6]])
          to add points to the current sample at (1,2,3) and (4,5,6).
          Set
          locations = LongTensor([[1,2,3,7],
                                  [4,5,6,9]])
          to add point (1,2,3) to sample 7, and (4,5,6) to sample 9 (0-indexed).

        """
        l = locations[:, :self.dimension]
        assert l.min() >= 0 and (self.spatial_size.expand_as(l) - l).min() > 0
        self.metadata.setInputSpatialLocations(
            self.features, locations.contiguous(), vectors.contiguous(), overwrite)

    def set_locations_(self, locations, vectors, overwrite=False):
        self.metadata.setInputSpatialLocations(
            self.features, locations, vectors, overwrite)

    def add_sample_from_tensor(self, tensor, offset, threshold=0):
        self.metadata.addSampleFromThresholdedTensor(
            self.features,
            tensor,
            offset,
            self.spatial_size,
            threshold)

    def precompute_metadata(self, size):
        """
        Optional.
        Allows precomputation of 'rulebooks' in data loading threads.
        Use size == 2 if downsizing with size-2 stride-2 operations
        Use size == 3 if downsizing with size-3 stride-2 operations
        """
        if size == 2:
            self.metadata.generateRuleBooks2s2()
        if size == 3 :
            self.metadata.generateRuleBooks3s2()

    "Deprecated method names."
    def addSample(self):
        self.metadata.batchAddSample()

    def setLocation(self, location, vector, overwrite=False):
        assert location.min() >= 0 and (self.spatial_size - location).min() > 0
        self.metadata.setInputSpatialLocation(
            self.features, location, vector, overwrite)

    def setLocation_(self, location, vector, overwrite=False):
        self.metadata.setInputSpatialLocation(
            self.features, location, vector, overwrite)

    def setLocations(self, locations, vectors, overwrite=False):
        l = locations[:, :self.dimension]
        assert l.min() >= 0 and (self.spatial_size.expand_as(l) - l).min() > 0
        self.metadata.setInputSpatialLocations(
            self.features, locations, vectors, overwrite)

    def setLocations_(self, locations, vector, overwrite=False):
        self.metadata.setInputSpatialLocations(
            self.features, locations, vectors, overwrite)

    def addSampleFromTensor(self, tensor, offset, threshold=0):
        self.metadata.addSampleFromThresholdedTensor(
            self.features,
            tensor,
            offset,
            self.spatial_size,
            threshold)

    def precomputeMetadata(self, size):
        """
        Optional.
        Allows precomputation of 'rulebooks' in data loading threads.
        Use size == 2 if downsizing with size-2 stride-2 operations
        Use size == 3 if downsizing with size-3 stride-2 operations
        """
        if size == 2:
            self.metadata.generateRuleBooks2s2()
        if size == 3 :
            self.metadata.generateRuleBooks3s2()
