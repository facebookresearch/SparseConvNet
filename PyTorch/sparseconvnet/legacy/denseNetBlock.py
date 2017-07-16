# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import sparseconvnet as s
from torch.legacy.nn import Container
from ..utils import toLongTensor, typed_fn, optionalTensor, nullptr, set
from .sparseConvNetTensor import SparseConvNetTensor
from .batchNormalization import *
from .affineReLUTrivialConvolution import AffineReLUTrivialConvolution
from .validConvolution import ValidConvolution
import math


class DenseNetBlock(Container):
    def __init__(self, dimension, nInputPlanes, nExtraLayers=2, growthRate=16):
        Container.__init__(self)
        self.dimension = dimension
        self.nInputPlanes = nInputPlanes
        self.nExtraLayers = nExtraLayers
        self.growthRate = growthRate
        assert(self.nExtraLayers >= 1)
        self.nOutputPlanes = nInputPlanes + nExtraLayers * growthRate
        self.output = SparseConvNetTensor(torch.Tensor())

        # Module 1: Batchnorm the input into the start of self.output
        self.add(
            BatchNormalizationInTensor(
                nInputPlanes,
                output_column_offset=0))
        self.modules[0].output = self.output
        self.gradInput = self.modules[0].gradInput

        for i in range(nExtraLayers):
            nFeatures = self.nInputPlanes + i * growthRate
            nFeaturesB = 4 * growthRate
            # Modules 4*i+1
            self.add(AffineReLUTrivialConvolution(nFeatures, nFeaturesB, True))
            # Module 4*i+2
            self.add(BatchNormalization(nFeaturesB))
            # Module 4*i+3
            self.add(
                ValidConvolution(
                    dimension,
                    nFeaturesB,
                    growthRate,
                    3,
                    False))
            # Module 4*i+4
            self.add(
                BatchNormalizationInTensor(
                    growthRate,
                    output_column_offset=self.nInputPlanes +
                    i *
                    growthRate))
            self.modules[4 * i + 4].output = self.output

    def updateOutput(self, input):
        assert input.features.size(1) == self.nInputPlanes
        self.output.spatial_size = input.spatial_size
        self.output.metadata = input.metadata
        self.output.features.resize_(
            input.features.size(0), self.nOutputPlanes)
        i = input
        for m in self.modules:
            i = m.updateOutput(i)
        return self.output

    def backward(self, input, gradOutput, scale=1):
        assert scale == 1
        g = gradOutput
        for i in range(self.nExtraLayers):
            self.modules[4 * i + 1].gradInput = gradOutput
        for m, m_ in zip(self.modules[:0:-1],
                         self.modules[len(self.modules) - 2::-1]):
            g = m.backward(m_.output, g)
        self.modules[0].backward(input, g)
        return self.gradInput

    def type(self, type, tensorCache=None):
        self._type = type
        self.output.features = self.output.features.type(type)
        for x in self.modules:
            x.type(type)
        self.gradInput = self.modules[0].gradInput

    def __repr__(self):
        s = 'DenseNetBlock(' + str(self.nInputPlanes) + '->' + str(self.nInputPlanes) + '+' + str(
            self.nExtraLayers) + '*' + str(self.growthRate) + '=' + str(self.nOutputPlanes) + ')'
        return s

    def clearState(self):
        for _, m in ipairs(self.modules):
            m.clearState()
        set(self.output)
        set(self.gradInput)

    def suggestInputSize(self, out_size):
        return out_size
