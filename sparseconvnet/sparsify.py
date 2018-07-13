# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sparseconvnet
from torch.autograd import Function, Variable
from torch.nn import Module, Parameter
from .utils import *
from .sparseConvNetTensor import SparseConvNetTensor
from .metadata import Metadata

class Sparsify(Module):
    def __init__(self, dimension):
        Module.__init__(self)
        self.dimension = dimension
    def forward(self, input):
        if input.features.numel():
            output = SparseConvNetTensor()
            output.metadata = Metadata(self.dimension)
            output.spatial_size = input.spatial_size
            active = input.features[:,0]>0
            output.features=input.features[active]
            active=active.type('torch.LongTensor')
            input.metadata.sparsifyMetadata(
                output.metadata,
                input.spatial_size,
                active.byte(),
                active.cumsum(0))
            return output
        else:
            return input
