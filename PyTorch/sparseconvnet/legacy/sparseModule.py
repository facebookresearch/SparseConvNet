# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.legacy.nn import Module
from ..utils import set


class SparseModule(Module):
    def __init__(self):
        Module.__init__(self)

    def clearState(self):
        set(self.output)
        set(self.gradInput)

    def suggestInputSize(self, out_size):
        return out_size
