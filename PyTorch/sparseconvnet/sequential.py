# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.nn import Sequential as S
from .utils import set

class Sequential(S):
    def input_spatial_size(self, out_size):
        for m in reversed(self._modules):
            out_size = self._modules[m].input_spatial_size(out_size)
        return out_size
    def add(self, module):
        self._modules[str(len(self._modules))]=module
        return self
