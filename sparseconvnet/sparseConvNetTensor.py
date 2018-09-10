# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torch.autograd import Variable


class SparseConvNetTensor(object):
    def __init__(self, features=None, metadata=None, spatial_size=None):
        self.features = features
        self.metadata = metadata
        self.spatial_size = spatial_size

    def get_spatial_locations(self, spatial_size=None):
        "Coordinates and batch index for the active spatial locations"
        if spatial_size is None:
            spatial_size = self.spatial_size
        t = torch.LongTensor()
        self.metadata.getSpatialLocations(spatial_size, t)
        return t

    def type(self, t=None):
        if t:
            self.features = self.features.type(t)
            return self
        return self.features.type()

    def cuda(self, async=False):
        self.features = self.features.cuda(async=async)
        return self

    def cpu(self):
        self.features = self.features.cpu()
        return self

    def set_(self):
        self.features.set_(self.features.storage_type()())
        self.metadata.set_()
        self.spatialSize = None

    def __repr__(self):
        sl = self.get_spatial_locations() if self.metadata else None
        return 'SparseConvNetTensor<<' + \
            'features=' + repr(self.features) + \
            ',features.shape=' + repr(self.features.shape) + \
            ',batch_locations=' + repr(sl) + \
            ',batch_locations.shape=' + repr(sl.shape if self.metadata else None) + \
            ',spatial size=' + repr(self.spatial_size) + \
            '>>'

    def to_variable(self, requires_grad=False, volatile=False):
        "Convert self.features to a variable for use with modern PyTorch interface."
        self.features = Variable(
            self.features,
            requires_grad=requires_grad,
            volatile=volatile)
        return self
