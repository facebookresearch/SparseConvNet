# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Functions to build some networks

DeepCNet
VggNet - deep
ResNet - deeper
"""

import torch.legacy.nn
import math
from .affineReLUTrivialConvolution import AffineReLUTrivialConvolution
from .averagePooling import AveragePooling
from .cAddTable import CAddTable
from .convolution import Convolution
from .deconvolution import Deconvolution
from .denseNetBlock import DenseNetBlock
from .validConvolution import ValidConvolution
from .networkInNetwork import NetworkInNetwork
from .batchNormalization import BatchNormReLU, BatchNormalizationInTensor
from .maxPooling import MaxPooling
from .concatTable import ConcatTable
from .joinTable import JoinTable
from .sequential import Sequential
from .reLU import ReLU
from .identity import Identity


def DeepCNet(dimension, nInputPlanes, nPlanes, bn=True):
    """
    i.e. sparseconvnet(2,nInputPlanes,{16,32,48,64,80},4,32) maps
    (batchSize,nInputPlanes,16n+32,16n+32)->(batchSize,80,n,n)
    Regular (i.e. not 'valid') convolutions
    https://arxiv.org/abs/1409.6070
    Based on "Multi-column Deep Neural Networks for Image Classification",
    Dan Ciresan, Ueli Meier, Jonathan Masci and Jurgen Schmidhuber
    """
    m = Sequential()

    def c(nIn, nOut, size):
        m.add(Convolution(dimension, nIn, nOut, size, 1, false))
        if bn:
            m.add(BatchNormReLU(nOut))
        else:
            m.add(ReLU(True))
    c(nInputPlanes, nPlanes[0], 3)
    for i in range(1, len(nPlanes)):
        m.add(MaxPooling(dimension, 2, 2))
        c(nPlanes[i - 1], nPlanes[i], 2)
    end
    m.nOutputPlanes = nPlanes[-1]
    return m


def SparseVggNet(dimension, nInputPlanes, layers):
    """
    VGG style nets
    Use valid convolutions
    Also implements 'Plus'-augmented nets
    """
    nPlanes = nInputPlanes
    m = Sequential()
    for x in layers:
        if x == 'MP':
            m.add(MaxPooling(dimension, 3, 2))
        elif x[0] == 'MP':
            m.add(MaxPooling(dimension, x[1], x[2]))
        elif x[0] == 'C' and len(x) == 2:
            m.add(ValidConvolution(dimension, nPlanes, x[1], 3, False))
            nPlanes = x[1]
            m.add(BatchNormReLU(nPlanes))
        elif x[0] == 'C' and len(x) == 3:
            m.add(ConcatTable()
                  .add(
                ValidConvolution(dimension, nPlanes, x[1], 3, False)
            )
                .add(
                Sequential()
                .add(Convolution(dimension, nPlanes, x[2], 3, 2, False))
                .add(BatchNormReLU(x[2]))
                .add(ValidConvolution(dimension, x[2], x[2], 3, False))
                .add(BatchNormReLU(x[2]))
                .add(Deconvolution(dimension, x[2], x[2], 3, 2, False))
            )).add(JoinTable([x[1], x[2]]))
            nPlanes = x[1] + x[2]
            m.add(BatchNormReLU(nPlanes))
        elif x[0] == 'C' and len(x) == 4:
            m.add(ConcatTable()
                  .add(
                ValidConvolution(dimension, nPlanes, x[1], 3, False)
            )
                .add(
                Sequential()
                .add(Convolution(dimension, nPlanes, x[2], 3, 2, False))
                .add(BatchNormReLU(x[2]))
                .add(ValidConvolution(dimension, x[2], x[2], 3, False))
                .add(BatchNormReLU(x[2]))
                .add(Deconvolution(dimension, x[2], x[2], 3, 2, False))
            )
                .add(Sequential()
                     .add(Convolution(dimension, nPlanes, x[3], 3, 2, False))
                     .add(BatchNormReLU(x[3]))
                     .add(ValidConvolution(dimension, x[3], x[3], 3, False))
                     .add(BatchNormReLU(x[3]))
                     .add(Convolution(dimension, x[3], x[3], 3, 2, False))
                     .add(BatchNormReLU(x[3]))
                     .add(ValidConvolution(dimension, x[3], x[3], 3, False))
                     .add(BatchNormReLU(x[3]))
                     .add(Deconvolution(dimension, x[3], x[3], 3, 2, False))
                     .add(BatchNormReLU(x[3]))
                     .add(ValidConvolution(dimension, x[3], x[3], 3, False))
                     .add(BatchNormReLU(x[3]))
                     .add(Deconvolution(dimension, x[3], x[3], 3, 2, False))
                     )).add(JoinTable({x[1], x[2], x[3]}))
            nPlanes = x[1] + x[2] + x[3]
            m.add(BatchNormReLU(nPlanes))
    return m


def SparseResNet(dimension, nInputPlanes, layers):
    """
    pre-activated ResNet
    e.g. layers = {{'basic',16,2,1},{'basic',32,2}}
    """
    nPlanes = nInputPlanes
    m = Sequential()

    def residual(nIn, nOut, stride):
        if stride > 1:
            return Convolution(dimension, nIn, nOut, 3, stride, False)
        elif nIn != nOut:
            return NetworkInNetwork(nIn, nOut, False)
        else:
            return Identity()
    for blockType, n, reps, stride in layers:
        for rep in range(reps):
            if blockType[0] == 'b':  # basic block
                if rep == 0:
                    m.add(BatchNormReLU(nPlanes))
                    m.add(
                        ConcatTable().add(
                            Sequential().add(
                                ValidConvolution(
                                    dimension,
                                    nPlanes,
                                    n,
                                    3,
                                    False) if stride == 1 else Convolution(
                                    dimension,
                                    nPlanes,
                                    n,
                                    3,
                                    stride,
                                    False)) .add(
                                BatchNormReLU(n)) .add(
                                ValidConvolution(
                                    dimension,
                                    n,
                                    n,
                                    3,
                                    False))) .add(
                            residual(
                                nPlanes,
                                n,
                                stride)))
                else:
                    m.add(
                        ConcatTable().add(
                            Sequential().add(
                                BatchNormReLU(nPlanes)) .add(
                                ValidConvolution(
                                    dimension,
                                    nPlanes,
                                    n,
                                    3,
                                    False)) .add(
                                BatchNormReLU(n)) .add(
                                ValidConvolution(
                                    dimension,
                                    n,
                                    n,
                                    3,
                                    False))) .add(
                            Identity()))
            nPlanes = n
            m.add(CAddTable(True))
    m.add(BatchNormReLU(nPlanes))
    return m


def SparseDenseNet(dimension, nInputPlanes, layers):
    """
    SparseConvNet meets DenseNets using valid convolutions
    Could do with a less confusing name
    """
    nPlanes = nInputPlanes
    m = Sequential()
    for x in layers:
        if 'pool' in x:
            if 'size' not in x:
                x['size'] = 2
            if 'stride' not in x:
                x['stride'] = 2
            if 'base' not in x:
                x['base'] = 16
            if 'compression' not in x:
                x['compression'] = 0
            nDrop = x['base'] * \
                math.floor(nPlanes * x['compression'] / x['base'])
            if x['pool'] == 'MP':
                m.add(MaxPooling(dimension, x['size'], x['stride'], nDrop))
                nPlanes -= nDrop
            if x['pool'] == 'AP':
                m.add(AveragePooling(dimension, x['size'], x['stride'], nDrop))
                nPlanes -= nDrop
            elif x['pool'] == 'BN-R-C-AP':
                m.add(BatchNormReLU(nPlanes))
                m.add(NetworkInNetwork(nPlanes, nPlanes - nDrop))
                nPlanes = nPlanes - nDrop
                m.add(AveragePooling(dimension, x['size'], x['stride']))
            elif x['pool'] == 'C-AP':
                m.add(NetworkInNetwork(nPlanes, nPlanes - nDrop))
                nPlanes = nPlanes - nDrop
                m.add(AveragePooling(dimension, x['size'], x['stride']))
        else:
            if 'nExtraLayers' not in x:
                x['nExtraLayers'] = 2
            if 'growthRate' not in x:
                x['growthRate'] = 16
            m.add(
                DenseNetBlock(
                    dimension,
                    nPlanes,
                    x['nExtraLayers'],
                    x['growthRate']))
            nPlanes = nPlanes + x['nExtraLayers'] * x['growthRate']
    m.nOutputPlanes = nPlanes
    return m
