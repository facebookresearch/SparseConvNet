# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import sparseconvnet
from . import SparseModule
from ..utils import toLongTensor, dim_typed_fn, optionalTensor, nullptr, set
from ..sparseConvNetTensor import SparseConvNetTensor


class JoinTable(SparseModule):
    def __init__(self, nPlanes):
        SparseModule.__init__(self)
        self.nPlanes = nPlanes
        self.gradInput = [torch.Tensor() for p in nPlanes]
        self.nOutputPlanes = sum(nPlanes)
        self.output = SparseConvNetTensor(torch.Tensor())

    def updateOutput(self, input):
        self.output.features.resize_(
            input[0].features.size(0),
            self.nOutputPlanes)
        self.output.metadata = input[0].metadata
        self.output.spatial_size = input[0].spatial_size
        offset = 0
        for i, n in zip(input, self.nPlanes):
            self.output.features.narrow(1, offset, n).copy_(i.features)
            offset += n
        return self.output

    def updateGradInput(self, input, gradOutput):
        offset = 0
        a = input[0].features.size(0)
        for g, n in zip(self.gradInput, self.nPlanes):
            g.resize_(a, n).copy_(gradOutput.narrow(1, offset, n))
            offset += n
        return self.gradInput

    def type(self, t, tensorCache=None):
        if t:
            self.output.type(t)
            self.gradInput = [g.type(t) for g in self.gradInput]

    def clearState(self):
        set(self.output)
        for g in self.gradInput:
            set(g)

    def __repr__(self):
        s = 'JoinTable: ' + str(self.nPlanes[0])
        for n in self.nPlanes[1:]:
            s = s + ' + ' + str(n)
        s = s + ' -> ' + str(self.nOutputPlanes)
        return s
