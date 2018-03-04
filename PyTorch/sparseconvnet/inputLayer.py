# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.autograd import Function
from torch.nn import Module, Parameter
from .utils import *
from .sparseConvNetTensor import SparseConvNetTensor
from .metadata import Metadata


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
        ctx.metadata = metadata
        ctx.dimension = dimension
        dim_typed_fn(dimension, input_features, 'InputLayer_updateOutput')(
            metadata.ffi,
            spatial_size,
            coords,
            input_features,
            output_features,
            batch_size,
            mode,
            torch.cuda.IntTensor() if input_features.is_cuda else nullptr
        )
        return output_features

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.data.new()
        dim_typed_fn(
            ctx.dimension,
            grad_output.data,
            'InputLayer_updateGradInput')(
            ctx.metadata.ffi,
            grad_input.data,
            grad_output.contiguous().data,
            torch.cuda.IntTensor() if grad_output.data.is_cuda else nullptr)
        return None, None, None, None, grad_input, None, None


class InputLayer(Module):
    def __init__(self, dimension, spatial_size, mode=3):
        Module.__init__(self)
        self.dimension = dimension
        self.spatial_size = toLongTensor(dimension, spatial_size)
        self.mode = mode
    # (coords,input_features,batch_size or None) = input

    def forward(self, input):
        output = SparseConvNetTensor(
            metadata=Metadata(
                self.dimension),
            spatial_size=self.spatial_size)
        output.features = InputLayerFunction.apply(
            self.dimension,
            output.metadata,
            self.spatial_size,
            input[0],
            input[1],
            0 if len(input == 2) else input[2],
            self.mode
        )
        return output


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
        ctx.metadata = metadata
        ctx.dimension = dimension
        dim_typed_fn(dimension, input_features, 'BLInputLayer_updateOutput')(
            metadata.ffi,
            spatial_size,
            coords,
            input_features,
            output_features,
            mode,
            torch.cuda.IntTensor() if input_features.is_cuda else nullptr
        )
        return output_features

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.data.new()
        dim_typed_fn(
            ctx.dimension,
            grad_output.data,
            'BLInputLayer_updateGradInput')(
            ctx.metadata.ffi,
            grad_input.data,
            grad_output.contiguous().data,
            torch.cuda.IntTensor() if grad_output.data.is_cuda else nullptr)
        return None, None, None, None, grad_input, None


class BLInputLayer(Module):
    def __init__(self, dimension, spatial_size, mode=3):
        Module.__init__(self)
        self.dimension = dimension
        self.spatial_size = toLongTensor(dimension, spatial_size)
        self.mode = mode
    # (coords,input_features) = input

    def forward(self, input):
        output = SparseConvNetTensor(
            metadata=Metadata(
                self.dimension),
            spatial_size=self.spatial_size)
        output.features = BLInputLayerFunction.apply(
            self.dimension,
            output.metadata,
            self.spatial_size,
            input[0],
            input[1],
            self.mode
        )
        return output


class BLOutputLayerFunction(Function):
    @staticmethod
    def forward(
            ctx,
            dimension,
            metadata,
            input_features):
        output_features = input_features.new()
        ctx.metadata = metadata
        ctx.dimension = dimension
        dim_typed_fn(dimension, input_features, 'BLOutputLayer_updateOutput')(
            metadata.ffi,
            input_features,
            output_features,
            torch.cuda.IntTensor() if input_features.is_cuda else nullptr
        )
        return output_features

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.data.new()
        dim_typed_fn(
            ctx.dimension,
            grad_output.data,
            'BLOutputLayer_updateGradInput')(
            ctx.metadata.ffi,
            grad_input.data,
            grad_output.contiguous().data,
            torch.cuda.IntTensor() if grad_output.data.is_cuda else nullptr)
        return None, None, grad_input


class BLOutputLayer(Module):
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
