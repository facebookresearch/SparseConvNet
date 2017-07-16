# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Assume all the inputs have identical SparseGrids and input[i].nActive
Assume input[0].nPlanes >= input[i].nPlanes for all i=1,#input
output.validRules is taken from input[0].validRules (could do set union?)
(for resnets, make sure the residual link is input[1])
"""

import torch
import sparseconvnet
from . import SparseModule
from ..utils import toLongTensor, dim_typed_fn, optionalTensor, nullptr, set
from .sparseConvNetTensor import SparseConvNetTensor


class CAddTable(SparseModule):
    def __init__(self, ip=False):
        SparseModule.__init__(self)
        self.inplace = ip
        if ip:
            self.output = None
        else:
            self.output = SparseConvNetTensor(torch.Tensor())

    def updateOutput(self, input):
        if self.inplace:
            self.output = input[0]
        else:
            self.output.features.resize_as_(
                input[0].features).copy_(
                input[0].features)
            self.output.metadata = input[0].metadata
            self.output.spatial_size = input[0].spatial_size
        for i in input[1:]:
            self.output.features.narrow(
                1, 0, i.features.size(1)).add_(
                i.features)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = []
        n = input[0].features.size(1)
        for i in input:
            n_ = i.features.size(1)
            if self.inplace and n_ == n:
                self.gradInput.append(gradOutput)
            else:
                self.gradInput.append(gradOutput.narrow(1, 0, n_).clone())
        return self.gradInput

    def type(self, t, tensorCache=None):
        if t and not self.inplace:
            self.output.type(t)

    def clearState(self):
        if self.inplace:
            self.output = None
        else:
            set(self.output)
        self.gradInput = None
