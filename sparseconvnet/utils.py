# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import sparseconvnet_SCN as scn

def toLongTensor(dimension, x):
    if hasattr(x, 'type') and x.type() == 'torch.LongTensor':
        return x
    elif isinstance(x, (list, tuple)):
        assert len(x) == dimension
        return torch.LongTensor(x)
    else:
        return torch.LongTensor(dimension).fill_(x)


typeTable = {
    'torch.FloatTensor': 'cpu_float_',
    'torch.DoubleTensor': 'cpu_double_',
    'torch.cuda.FloatTensor': 'cuda_float_'}


def dim_fn(dimension, name):
    f=getattr(scn, name + '_' + str(dimension))
    return f


def typed_fn(t, name):
    f=getattr(scn, typeTable[t.type()] + name)
    return f


def dim_typed_fn(dimension, t, name):
    f=getattr(scn, typeTable[t.type()] + name + '_' + str(dimension))
    return f


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
        for i in range(len(d)):
            item = q.get()
            yield item
            q.task_done()
        q.join()
    return iterator
