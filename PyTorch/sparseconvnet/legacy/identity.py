# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.legacy.nn import Identity as I
from .sparseModule import SparseModule


class Identity(SparseModule):
    def forward(self, input):
        self.output = input
        return self.output

    def backward(self, input, gradOutput, scale=1):
        self.gradInput = gradOutput
        return self.gradInput

    def clearState(self):
        self.output = None
        self.gradInput = None

    def suggestInputSize(self, out_size):
        return out_size
