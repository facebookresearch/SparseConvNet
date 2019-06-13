# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.autograd import Function, Variable
from torch.nn import Module
import sparseconvnet
from .utils import *
from .sparseConvNetTensor import SparseConvNetTensor
from .metadata import Metadata
from .sequential import Sequential
from .activations import Sigmoid
from .networkInNetwork import NetworkInNetwork

class SparsifyFCS(Module):
    """
    Sparsify by looking at the first feature channel's sign.
    """
    def __init__(self, dimension):
        Module.__init__(self)
        self.dimension = dimension
    def forward(self, input):
        if input.features.numel():
            output = SparseConvNetTensor()
            output.metadata = Metadata(self.dimension)
            output.spatial_size = input.spatial_size
            active = input.features[:,0]>0
            output.features=input.features[active]
            active=active.type('torch.LongTensor')
            input.metadata.sparsifyMetadata(
                output.metadata,
                input.spatial_size,
                active.byte(),
                active.cumsum(0))
            return output
        else:
            return input

class FakeGradHardSigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            x):
        ctx.save_for_backward(x)
        with torch.no_grad():
            y=(x>0).float()
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
        x, = ctx.saved_tensors
        with torch.no_grad():
            #Either:
            #y=torch.sigmoid(x) #torch.sigmoid(x/5)?
            #df = y*(1-y)
            #Or:
            df = ((-2<x)*(x<+2)).float()*0.25
            #
            grad_input = grad_output*df
        return grad_input
class FakeGradHardSigmoid(Module):
    def forward(self, input):
        output = SparseConvNetTensor()
        output.features = FakeGradHardSigmoidFunction.apply(input.features)
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        return output

class Sparsify(Module):
    def __init__(self,dimension, nIn, activation=None):
        Module.__init__(self)
        self.dimension=dimension
        self.activation = activation
        if activation == 'fakeGradHardSigmoid':
            self.net = Sequential(NetworkInNetwork(nIn,1,True),FakeGradHardSigmoid())
        elif activation == 'sigmoid':
            self.net = Sequential(NetworkInNetwork(nIn,1,True),Sigmoid())
        else:
            self.net = NetworkInNetwork(nIn,1,True)
        self.threshold=0.5 if activation else 0
    def forward(self,input):
        if input.features.numel():
            output = SparseConvNetTensor()
            output.spatial_size = input.spatial_size
            output.metadata = Metadata(self.dimension)
            output.mask = self.net(input).features.view(-1)
            if self.threshold<0:
                print(output.mask.mean(),output.mask.std())
            active = output.mask>self.threshold
            output.features=input.features[active]
            active=active.cpu()
            input.metadata.sparsifyMetadata(
                output.metadata,
                input.spatial_size,
                active.byte(),
                active.long().cumsum(0))
            #print('Sparsify2 output', output.features.shape, output.mask.features.shape)
            return output
        else:
            input.mask=None
            return input
