# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Store Metadata relating to which spatial locations are active at each scale.
Convolutions, valid convolutions and 'convolution reversing' deconvolutions
all coexist within the same MetaData object as long as each spatial size
only occurs once.

Serialization is emulated by storing the pointer as an integer.
This is sufficient for mutithreaded batch preparation: each 'serialized'
object must be de-serialized exactly once.
"""

import cffi
from ..utils import dim_fn
from ..SCN import scn_readPtr, scn_writePtr, scn_3_setInputSpatialSize

ffi = cffi.FFI()


class Metadata(object):
    def __init__(self, dimension, ptr=0):
        #print('make meta',dimension, ptr)
        self.dimension = dimension
        self.ffi = ffi.new('void *[1]')
        scn_writePtr(ptr, self.ffi)
        self.ffigc = ffi.gc(self.ffi, dim_fn(self.dimension, 'freeMetadata'))

    def set_(self):
        if hasattr(self, 'ffi'):
            del self.ffigc
            del self.ffi

    def __reduce__(self):
        if hasattr(self, 'ffi'):
            del self.ffigc
            del self.ffi
        return (self.__class__, (self.dimension,))

    def __repr__(self):
        if hasattr(self, 'ffi'):
            return '<<Metadata:dim=' + \
                str(self.dimension) + ', p=' + str(scn_readPtr(self.ffi)) + '>>'
        else:
            return '<<Metadata:dim=' + str(self.dimension) + '>>'
