# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sparseconvnet
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import Module, Parameter
from .utils import *
from .sparseConvNetTensor import SparseConvNetTensor
from .batchNormalization import BatchNormalization


class Sigmoid(Module):
    def forward(self, input):
        output = SparseConvNetTensor()
        output.features = torch.sigmoid(input.features)
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        return output


class LeakyReLU(Module):
    def __init__(self,leak=1/3):
        Module.__init__(self)
        self.leak=leak
    def forward(self, input):
        output = SparseConvNetTensor()
        output.features = F.leaky_relu(input.features,self.leak)
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        return output


class Tanh(Module):
    def forward(self, input):
        output = SparseConvNetTensor()
        output.features = torch.tanh(input.features)
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        return output


class ReLU(Module):
    def forward(self, input):
        output = SparseConvNetTensor()
        output.features = F.relu(input.features)
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        return output


class ELU(Module):
    def forward(self, input):
        output = SparseConvNetTensor()
        output.features = F.elu(input.features)
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        return output

class SELU(Module):
    def forward(self, input):
        output = SparseConvNetTensor()
        output.features = F.selu(input.features)
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        return output

def BatchNormELU(nPlanes, eps=1e-4, momentum=0.9):
    return sparseconvnet.Sequential().add(BatchNormalization(nPlanes,eps,momentum)).add(ELU())
