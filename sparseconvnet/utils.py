# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from .sparseConvNetTensor import SparseConvNetTensor
from .metadata import Metadata

def toLongTensor(dimension, x):
    if hasattr(x, 'type') and x.type() == 'torch.LongTensor':
        return x
    elif isinstance(x, (list, tuple)):
        assert len(x) == dimension
        return torch.LongTensor(x)
    else:
        return torch.LongTensor(dimension).fill_(x)


def optionalTensor(a, b):
    return getattr(a, b) if hasattr(a, b) else torch.Tensor()


def optionalTensorReturn(a):
    return a if a.numel() else None


def threadDatasetIterator(d):
    try:
        import queue
    except BaseException:
        import Queue as queue
    import threading

    def iterator():
        def worker(i):
            for k in range(i, len(d), 8):
                q.put(d[k])
        q = queue.Queue(16)
        for i in range(8):
            t = threading.Thread(target=worker, args=(i,))
            t.start()
        for _ in range(len(d)):
            item = q.get()
            yield item
            q.task_done()
        q.join()
    return iterator


def appendSparseConvTensors(tensors):
    spatial_size=tensors[0].spatial_size
    dimension=len(spatial_size)
    x=SparseConvNetTensor(
        features=torch.cat([t.features for t in features],0),
        metadata=Metadata(dimension),
        spatial_size=spatial_size)
    for t in tensors:
        x.metadata.appendMetadata(t.metadata,spatial_size)
    return x

class AddCoords(torch.nn.Module):
    def forward(self, input):
        output = SparseConvNetTensor()
        if input.features.numel():
            with torch.no_grad():
                coords = input.get_spatial_locations()
                d = (input.spatial_size.type_as(input.features)-1)/2
                coords=coords[:,:-1].type_as(input.features)/ d[None,:] - 1
            output.features = torch.cat([input.features,coords],1)
        else:
            output.features = input.features
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        return output
