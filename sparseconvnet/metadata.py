# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Store Metadata relating to which spatial locations are active at each scale.
Convolutions, submanifold convolutions and 'convolution reversing' deconvolutions
all coexist within the same MetaData object as long as each spatial size
only occurs once.
"""

import sparseconvnet.SCN

def Metadata(dim):
    return getattr(sparseconvnet.SCN, 'Metadata_%d'%dim)()
