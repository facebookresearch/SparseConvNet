# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sparseconvnet.SCN
from torch.autograd import Function
from torch.nn import Module
from .utils import *
from .sparseConvNetTensor import SparseConvNetTensor


class GlobalFusionFunction(Function):
    @staticmethod
    def forward(
            ctx,
            local_input_features,
            local_input_metadata,
            local_input_spatial_size,
            local_input_base,
            global_input_features,
            global_input_metadata,
            global_input_spatial_size,
            global_input_base,
            dimension,
            scale_ratio):

        ctx.local_input_metadata = local_input_metadata
        ctx.global_input_metadata = global_input_metadata
        ctx.dimension = dimension

        output_features = local_input_features.new()
        sparseconvnet.SCN.GlobalFusion_updateOutput(
            local_input_spatial_size,
            global_input_spatial_size,
            local_input_base,
            global_input_base,
            local_input_metadata,
            global_input_metadata,
            local_input_features,
            global_input_features,
            output_features,
            scale_ratio)

        ctx.save_for_backward(
            local_input_spatial_size,
            global_input_spatial_size,
            local_input_base,
            global_input_base,
            scale_ratio)
        return output_features

    @staticmethod
    def backward(ctx, grad_output):
        local_spatial_size,\
            global_spatial_size,\
            local_base,\
            global_base,\
            scale_ratio = ctx.saved_tensors

        local_grad_input = grad_output.new()
        global_grad_input = grad_output.new()
        sparseconvnet.SCN.GlobalFusion_backward(
            local_spatial_size,
            global_spatial_size,
            local_base,
            global_base,
            ctx.local_metadata,
            ctx.global_metadata,
            local_grad_input,
            global_grad_input,
            grad_output,
            ctx.scale_ratio)
        return local_grad_input, None, None, None, global_grad_input, None, None, None, None, None


class GlobalFusion(Module):
    def __init__(self, dimension, scale_ratio):
        # super(Module, self).__init__()
        Module.__init__(self)
        self.dimension = dimension
        # self.scale_ratio = toLongTensor(dimension, scale_ratio)
        self.scale_ratio = torch.Tensor([scale_ratio] * dimension)
        print(self)

    def forward(self, local_input, global_input, local_base, global_base):
        output = SparseConvNetTensor()
        output.metadata = local_input.metadata
        output.spatial_size = local_input.spatial_size
        output.features = GlobalFusionFunction.apply(
            local_input.features,
            local_input.metadata,
            local_input.spatial_size,
            local_base,
            global_input.features,
            global_input.metadata,
            global_input.spatial_size,
            global_base,
            self.dimension,
            self.scale_ratio)
        return output

    def input_spatial_size(self, out_size):
        return out_size
        # return (out_size - 1) * self.pool_stride + self.pool_size

    def __repr__(self):
        s = 'GlobalFusion scale=('
        for i in self.scale_ratio[:-1]:
            s = s + str(i.item()) + ','
        s = s + str(self.scale_ratio[-1].item()) + ')'

        return s
