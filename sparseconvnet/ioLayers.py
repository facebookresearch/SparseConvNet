# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sparseconvnet.SCN
from torch.autograd import Function
from torch.nn import Module, Parameter
from .utils import *
from .sparseConvNetTensor import SparseConvNetTensor
from .metadata import Metadata


class InputLayer(Module):
    """
    Takes a tuple (coords, features, batch_size [optional])
    * coords is 2d torch.LongTensor with size
       N x dimension   (batch size == 1)
    or
       N x (dimension+1)  (first d columns are coordinates, last column is batch index)

    * features is a CPU or CUDA float tensor with size

      N x n_feature_planes

    * batch_size if given, set a lower bound on the the number of samples in the output tensor.
    Batch size can normally be inferred from the last column of coords, but this may fail if
    some of the batch items are totally empty.

    In case of repetition in coords:
    mode == 0 if the input is guaranteed to have no duplicates
    mode == 1 to use the last item at each spatial location
    mode == 2 to keep the first item at each spatial location
    mode == 3 to sum feature vectors sharing one spatial location
    mode == 4 to average feature vectors at each spatial location

    Output is a SparseConvNetTensor
    """
    def __init__(self, dimension, spatial_size, mode=3):
        Module.__init__(self)
        self.dimension = dimension
        self.spatial_size = toLongTensor(dimension, spatial_size)
        self.mode = mode
        self.device = None

    def to(self, device):
        self.device=device
        return self

    def forward(self, input):
        output = SparseConvNetTensor(
            metadata=Metadata(
                self.dimension),
            spatial_size=self.spatial_size)
        output.features = InputLayerFunction.apply(
            self.dimension,
            output.metadata,
            self.spatial_size,
            input[0].cpu().long(),
            input[1].to(self.device) if self.device else input[1],
            0 if len(input) == 2 else input[2],
            self.mode
        )
        return output


class OutputLayer(Module):
    """
    Used in conjunction with an InputLayer for 'autoencoder' style networks
    Takes a SparseConvNetTensor and results a float Tensor of size

    N x n_feature_planes

    N is defined by the InputLayer

    Behavior during forward-/back-propagation depends on the InputLayer's mode
    """
    def __init__(self, dimension):
        Module.__init__(self)
        self.dimension = dimension

    def forward(self, input):
        output = OutputLayerFunction.apply(
            self.dimension,
            input.metadata,
            input.features
        )
        return output


class BLInputLayer(Module):
    """
    Takes a tuple (coords, features)
    * coords is 3d torch.LongTensor with size
       batch_size x length x dimension

      Coordinates should be >=0, or -1 to indicate 'empty'

    * features is a 3d CPU or CUDA float Tensor with size

      batch_size x length x n_feature_planes

    mode == 0 Assumes that for each coords[i, :], the locations are unique and not 'empty'.
    mode == 1 Use the last item at each spatial location
    mode == 2 Keep the first item at each spatial location
    mode == 3 Sum feature vectors sharing one spatial location
    mode == 4 Average feature vectors at each spatial location

    Output is a SparseConvNetTensor
    """
    def __init__(self, dimension, spatial_size, mode=3):
        Module.__init__(self)
        self.dimension = dimension
        self.spatial_size = toLongTensor(dimension, spatial_size)
        self.mode = mode
        self.device = None

    def to(self, device):
        self.device=device
        return self

    def forward(self, input):
        output = SparseConvNetTensor(
            metadata=Metadata(
                self.dimension),
            spatial_size=self.spatial_size)
        output.features = BLInputLayerFunction.apply(
            self.dimension,
            output.metadata,
            self.spatial_size,
            input[0].cpu().long(),
            input[1].to(self.device) if self.device else input[1],
            self.mode
        )
        return output


class BLOutputLayer(Module):
    """
    Used in conjunction with a BLInputLayer for 'autoencoder' style networks
    Takes a SparseConvNetTensor and results a float Tensor of batch_size

    batch_size x length x n_feature_planes

    batch_size and length are defined by the BLInputLayer

    Behavior during forward-/back-propagation depends on the BLInputLayer's mode
    """
    def __init__(self, dimension):
        Module.__init__(self)
        self.dimension = dimension

    def forward(self, input):
        output = BLOutputLayerFunction.apply(
            self.dimension,
            input.metadata,
            input.features
        )
        return output


class InputLayerFunction(Function):
    @staticmethod
    def forward(
            ctx,
            dimension,
            metadata,
            spatial_size,
            coords,
            input_features,
            batch_size,
            mode):
        output_features = input_features.new()
        ctx.dimension = dimension
        ctx.metadata_ = metadata
        sparseconvnet.SCN.InputLayer_updateOutput(
            metadata,
            spatial_size,
            coords,
            input_features.contiguous(),
            output_features,
            batch_size,
            mode
        )
        return output_features

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.new()
        sparseconvnet.SCN.InputLayer_updateGradInput(
            ctx.metadata_,
            grad_input,
            grad_output.contiguous())
        return None, None, None, None, grad_input, None, None


class OutputLayerFunction(Function):
    @staticmethod
    def forward(
            ctx,
            dimension,
            metadata,
            input_features):
        output_features = input_features.new()
        ctx.metadata_ = metadata
        ctx.dimension = dimension
        sparseconvnet.SCN.OutputLayer_updateOutput(
            metadata,
            input_features.contiguous(),
            output_features
        )
        return output_features

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.new()
        grad_output=grad_output.contiguous()
        sparseconvnet.SCN.OutputLayer_updateGradInput(
            ctx.metadata_,
            grad_input,
            grad_output.contiguous())
        return None, None, grad_input


class BLInputLayerFunction(Function):
    @staticmethod
    def forward(
            ctx,
            dimension,
            metadata,
            spatial_size,
            coords,
            input_features,
            mode):
        output_features = input_features.new()
        ctx.metadata_ = metadata
        ctx.dimension = dimension
        sparseconvnet.SCN.BLInputLayer_updateOutput(
            metadata,
            spatial_size,
            coords,
            input_features.contiguous(),
            output_features,
            mode
        )
        return output_features

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.new()
        sparseconvnet.SCN.BLInputLayer_updateGradInput(
            ctx.metadata_,
            grad_input,
            grad_output.contiguous())
        return None, None, None, None, grad_input, None


class BLOutputLayerFunction(Function):
    @staticmethod
    def forward(
            ctx,
            dimension,
            metadata,
            input_features):
        output_features = input_features.new()
        ctx.metadata_ = metadata
        ctx.dimension = dimension
        sparseconvnet.SCN.BLOutputLayer_updateOutput(
            metadata,
            input_features.contiguous(),
            output_features
        )
        return output_features

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.new()
        sparseconvnet.SCN.BLOutputLayer_updateGradInput(
            ctx.metadata_,
            grad_input,
            grad_output.contiguous())
        return None, None, grad_input

class InputLayerInput(object):
    def __init__(self,coords,features):
        self.x=[coords,features]
    def __getitem__(self,n):
        return self.x[n]
    def __len__(self):
        return 2
    def cuda(self):
        self.x[1]=self.x[1].cuda()
        return self
