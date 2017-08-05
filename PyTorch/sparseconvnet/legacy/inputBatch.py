# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from .metadata import Metadata
from ..utils import toLongTensor, dim_fn
from .sparseConvNetTensor import SparseConvNetTensor

class InputBatch(SparseConvNetTensor):
    def __init__(self, dimension, spatial_size):
        self.dimension = dimension
        self.spatial_size = toLongTensor(dimension, spatial_size)
        SparseConvNetTensor.__init__(self, None, None, spatial_size)
        self.features = torch.FloatTensor()
        self.metadata = Metadata(dimension)
        dim_fn(dimension, 'setInputSpatialSize')(
            self.metadata.ffi, self.spatial_size)

    def addSample(self):
        dim_fn(self.dimension, 'batchAddSample')(
            self.metadata.ffi)

    def setLocation(self, location, vector, overwrite=False):
        assert location.min() >= 0 and (self.spatial_size - location).min() > 0
        dim_fn(self.dimension, 'setInputSpatialLocation')(
            self.metadata.ffi, self.features, location, vector, overwrite)

    def setLocation_(self, location, vector, overwrite=False):
        dim_fn(self.dimension, 'setInputSpatialLocation')(
            self.metadata.ffi, self.features, location, vector, overwrite)

    def setLocations(self, locations, vectors, overwrite=False):
        assert locations.min() >= 0 and (self.spatial_size.expand_as(locations) - locations).min() > 0

        dim_fn(self.dimension, 'setInputSpatialLocations')(
            self.metadata.ffi, self.features, locations, vectors, overwrite)

    def setLocations_(self, locations, vector, overwrite=False):
        dim_fn(self.dimension, 'setInputSpatialLocations')(
            self.metadata.ffi, self.features, locations, vectors, overwrite)

    def addSampleFromTensor(self, tensor, offset, threshold=0):
        self.nActive = dim_fn(
            self.dimension,
            'addSampleFromThresholdedTensor')(
            self.metadata.ffi,
            self.features,
            tensor,
            offset,
            self.spatial_size,
            threshold)

    def precomputeMetadata(self, stride):
        if stride == 2:
            dim_fn(self.dimension, 'generateRuleBooks2s2')(self.metadata.ffi)
        else:
            dim_fn(self.dimension, 'generateRuleBooks3s2')(self.metadata.ffi)

    def __repr__(self):
        return 'InputBatch<<' + repr(self.features) + repr(self.metadata) + \
            repr(self.spatial_size) + '>>'
