# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.legacy.nn import Sequential as S
from ..utils import set


class Sequential(S):
    def __init__(self):
        S.__init__(self)

    def suggestInputSize(self, out_size):
        for m in self.modules[::-1]:
            out_size = m.suggestInputSize(out_size)
        return out_size

    def clearState(self):
        set(self.output)
        set(self.gradInput)
        for m in self.modules:
            m.clearState()
