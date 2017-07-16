# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
If a LeakyReLU has leakiness zero, what is it?

Parameters
ip : operate in place (default true)
"""

import torch
import sparseconvnet
from torch.legacy.nn import Module
from .leakyReLU import LeakyReLU
from ..utils import toLongTensor, dim_typed_fn, optionalTensor, nullptr
from .sparseConvNetTensor import SparseConvNetTensor


class ReLU(LeakyReLU):
    def __init__(self, ip):
        LeakyReLU.__init__(self, 0, ip)
