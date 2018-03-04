# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import sparseconvnet.SCN as scn
from cffi import FFI


def toLongTensor(dimension, x):
    if hasattr(x, 'type') and x.type() == 'torch.LongTensor':
        return x
    elif isinstance(x, (list, tuple)):
        assert len(x) == dimension
        return torch.LongTensor(x)
    else:
        return torch.LongTensor(dimension).fill_(x)


typeTable = {
    'torch.FloatTensor': 'cpu_float',
    'torch.DoubleTensor': 'cpu_double',
    'torch.cuda.FloatTensor': 'gpu_float'}


def dim_fn(dimension, name):
    # print('dim_fn',dimension,name)
    return getattr(scn, 'scn_' + str(dimension) + '_' + name)


def typed_fn(t, name):
    # print('typed_fn',t.type(),name)
    return getattr(scn, 'scn_' + typeTable[t.type()] + '_' + name)


def dim_typed_fn(dimension, t, name):
    # print('dim_typed_fn',dimension,t.type(),name)
    return getattr(scn, 'scn_' +
                   typeTable[t.type()] +
                   str(dimension) +
                   name)


ffi = FFI()
nullptr = ffi.NULL


def optionalTensor(a, b):
    return getattr(a, b) if hasattr(a, b) else nullptr


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


# def threadDatasetIterator(d):
#     print('not threads!!!')
#     def iterator():
#         for x in d:
#             yield x
#     return iterator


def set(obj):
    if hasattr(obj, 'storage_type'):
        obj.set_(obj.storage_type()())
    else:
        obj.set_()
