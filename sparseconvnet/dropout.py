# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.autograd import Function
from torch.nn import Module
from .utils import *
from .sparseConvNetTensor import SparseConvNetTensor


class Dropout(Module):
    def __init__(self, p=0.5):
        Module.__init__(self)
        self.p = p

    def forward(self, input):
        output = SparseConvNetTensor()
        i = input.features
        if self.training:
            m = i.new().resize_(1).expand_as(i).fill_(1 - self.p)
            output.features = i * torch.bernoulli(m)
        else:
            output.features = i * (1 - self.p)
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        return output

    def input_spatial_size(self, out_size):
        return out_size


class BatchwiseDropout(Module):
    def __init__(self, p=0.5):
        Module.__init__(self)
        self.p = p

    def forward(self, input):
        output = SparseConvNetTensor()
        i = input.features
        if self.training:
            m = i.new().resize_(1).expand(1, i.shape[1]).fill_(1 - self.p)
            output.features = i * torch.bernoulli(m)
        else:
            output.features = i * (1 - self.p)
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        return output

    def input_spatial_size(self, out_size):
        return out_size
