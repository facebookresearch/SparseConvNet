# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from ..utils import dim_fn

class SparseConvNetTensor(object):
    def __init__(self, features=None, metadata=None, spatial_size=None):
        self.features = features
        self.metadata = metadata
        self.spatial_size = spatial_size

    def getSpatialLocations(self, spatial_size=None):
        if spatial_size is None:
            spatial_size = self.spatial_size

        t = torch.LongTensor()
        dim_fn(self.metadata.dimension, 'getSpatialLocations')(self.metadata.ffi, spatial_size, t)
        return t

    def type(self, t=None):
        if t:
            self.features = self.features.type(t)
        return self

    def set_(self):
        self.features.set_(self.features.storage_type()())
        self.metadata.set_()
        self.spatialSize = None

    def __repr__(self):
        return 'SparseConvNetTensor<<' + \
            repr(self.features) + repr(self.metadata) + repr(self.spatial_size) + '>>'
