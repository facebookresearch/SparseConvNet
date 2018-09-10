# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

class Sequential(torch.nn.Sequential):
    def input_spatial_size(self, out_size):
        for m in reversed(self._modules):
            out_size = self._modules[m].input_spatial_size(out_size)
        return out_size

    def add(self, module):
        self._modules[str(len(self._modules))] = module
        return self

    def reweight(self, input):
        for module in self._modules.values():
            if isinstance(module, Sequential):
                input = module.reweight(input)
            elif hasattr(input, 'features') and hasattr(module, 'weight') and hasattr(module, 'bias'):
                f = module(input).features
                f = f - module.bias
                s = f.std(0)
                f = f / s
                module.weight = torch.nn.Parameter(module.weight/s)
                module.bias = torch.nn.Parameter(-f.mean(0))
                input = module(input)
            else:
                input = module(input)
        return input
