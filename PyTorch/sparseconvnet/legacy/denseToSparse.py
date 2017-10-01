# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Function to convert a Dense Input into a sparse input.
If possible, avoid using this module; build the hidden layer using InputBatch.

Parameters:
dimension : of the input field
"""

import torch
from . import SparseModule
from ..utils import dim_fn, nullptr
from ..sparseConvNetTensor import SparseConvNetTensor
from ..metadata import Metadata

class DenseToSparse(SparseModule):
    def __init__(self, dimension):
        SparseModule.__init__(self)
        self.dimension = dimension
        self.output = SparseConvNetTensor(torch.Tensor(),Metadata(dimension))
        self.gradInput = torch.Tensor()

    def updateOutput(self, input):
        a=input
        aa=a.permute(*([0,]+list(range(2,2+self.dimension))+[1,])).clone()
        self.aas=aa.size()
        nz=aa.abs().sum(self.dimension+1).view(aa.size()[0:-1])
        s=torch.LongTensor(nz.stride()).view(1,self.dimension+1)
        nz=nz.nonzero()
        s=s.type_as(nz)
        aa=aa.view(-1,a.size(1))
        self.aas2=aa.size()
        self.r=(nz*s.expand_as(nz)).sum(1).view(-1)
        self.output.features=aa.index_select(0,self.r)
        self.output.spatial_size=torch.LongTensor(list(input.size()[2:]))
        dim_fn(self.dimension, 'createMetadataForDenseToSparse')(
            self.output.metadata.ffi,
            self.output.spatial_size,
            nz.cpu(),
            input.size(0))
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput.resize_(self.aas2).zero_()
        self.gradInput.index_copy_(0,self.r,gradOutput)
        self.gradInput=self.gradInput.view(self.aas).permute(*([0,self.dimension+1]+list(range(1,self.dimension+1))))
        return self.gradInput

    def clearState(self):
        SparseModule.clearState(self)
        self.aas=None
        self.r=None
    def __repr__(self):
        return 'DenseToSparse(' + str(self.dimension) + ')'
