# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import sparseconvnet
from torch.legacy.nn import ConcatTable as C
from ..utils import toLongTensor, dim_typed_fn, optionalTensor, nullptr, set
from .sparseConvNetTensor import SparseConvNetTensor


class ConcatTable(C):
    def __init__(self):
        C.__init__(self)
        self.gradInput = torch.Tensor()

    def updateOutput(self, input):
        self.output = []
        for m in self.modules:
            self.output.append(m.forward(input))
        return self.output

    def backward(self, input, gradOutput, scale=1):
        self.gradInput.resize_as_(input.features).zero_()
        for m, g in zip(self.modules, gradOutput):
            self.gradInput.add_(m.backward(input, g, scale))
        return self.gradInput

    def clearState(self):
        self.output = None
        set(self.gradInput)
        for m in self.modules:
            m.clearState()

    def suggestInputSize(self, nOut):
        return self.modules[0].suggestInputSize(nOut)
