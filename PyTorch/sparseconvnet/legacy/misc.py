# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.legacy.nn as nn
from .sequential import Sequential
from .sparseModule import SparseModule
from .sparseConvNetTensor import SparseConvNetTensor

class Tanh(SparseModule):
    def __init__(self):
        SparseModule.__init__(self)
        self.module=nn.Tanh()
        self.output = SparseConvNetTensor()
        self.output.features=self.module.output
        self.gradInput = self.module.gradInput

    def updateOutput(self, input):
        self.output.metadata = input.metadata
        self.output.spatial_size = input.spatial_size
        self.module.forward(input.features)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.module.updateGradInput(input.features,gradOutput)
        return self.gradInput

    def type(self, t, tensorCache=None):
        if t:
            self.module.type(t,tensorCache)
            self.output.features=self.module.output
            self.gradInput = self.module.gradInput

class ELU(SparseModule):
    def __init__(self):
        SparseModule.__init__(self)
        self.module=nn.ELU()
        self.output = SparseConvNetTensor()
        self.gradInput = self.module.gradInput

    def updateOutput(self, input):
        self.output.metadata = input.metadata
        self.output.spatial_size = input.spatial_size
        self.module.forward(input.features)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.module.updateGradInput(input.features,gradOutput)
        return self.gradInput

    def type(self, t, tensorCache=None):
        if t:
            self.module.type(t,tensorCache)
            self.output.features=self.module.output
            self.gradInput = self.module.gradInput

def BatchNormELU(nPlanes, eps=1e-4, momentum=0.9):
    return Sequential().add(BatchNormalization(nPlanes,eps,momentum)).add(ELU())
