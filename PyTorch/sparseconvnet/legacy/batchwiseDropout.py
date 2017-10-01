# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Implementation of batchwise dropout, optionally followed by LeakyReLU

Parameters:
nPlanes: number of input planes
p : dropout probability in the range [0,1]
ip : perform dropout inplace (default true)
leaky : in the range [0,1]. Set to zero to do ReLU after the dropout. Set to one
just to do dropout. Set to 1/3 for LeakyReLU after the dropout, etc. (default 1)
"""

import torch
import sparseconvnet
from . import SparseModule
from ..utils import toLongTensor, typed_fn
from ..sparseConvNetTensor import SparseConvNetTensor

class BatchwiseDropout(SparseModule):
    def __init__(
            self,
            nPlanes,
            p,
            ip=True,
            leaky=1):

        self.inplace = ip
        self.p = p
        self.leakiness = leaky
        self.noise = torch.Tensor(nPlanes)
        self.output = None if ip else SparseConvNetTensor(torch.Tensor())
        self.gradInput = None if ip else torch.Tensor()

    def updateOutput(self, input):
        if self.train:
            self.noise.bernoulli_(1-self.p)
        else:
            self.noise.fill_(1-self.p)

        if self.inplace:
            self.output = input
        else:
            self.output.metadata = input.metadata
            self.output.spatialSize = input.spatialSize

        typed_fn(input.features, 'BatchwiseMultiplicativeDropout_updateOutput')(
            input.features,
            self.output.features,
            self.noise,
            self.leakiness
            )
        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.inplace:
            self.gradInput = gradOutput

        typed_fn(input.features, 'BatchwiseMultiplicativeDropout_updateGradInput')(
            input.features,
            self.gradInput,
            gradOutput,
            self.noise,
            self.leakiness
            )
        return self.gradInput

    def type(self, t, tensorCache=None):
        self.noise.type(t)

        if not self.inplace:
            self.output.features.type(t)
            self.gradInput.features.type(t)

        SparseModule.type(self, t, tensorCache)

    def clearState(self):
        if self.inPlace:
            self.output = None
            self.gradOutput = None
        else:
            SparseModule.clearState(self)

    def __repr__(self):
        s = 'BatchwiseDropout(' + str(self.nPlanes) + ',p=' + str(self.p) + \
            ',ip=' + str(self.inplace)
        if self.leakiness > 0:
            s = s + ',leakiness=' + str(self.leakiness)
        s = s + ')'
        return s
