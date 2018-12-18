# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os
import pickle
train = []
test = []
p = [
    45,
    9,
    3,
    34,
    31,
    25,
    23,
    17,
    22,
    37,
    27,
    2,
    10,
    13,
    7,
    42,
    4,
    8,
    33,
    11,
    12,
    1,
    39,
    38,
    36,
    20,
    14,
    21,
    40,
    24,
    32,
    5,
    35,
    18,
    44,
    41,
    30,
    28,
    29,
    19,
    15,
    26,
    6,
    16,
    43]


def rescaleCharacter(c):
    cc = torch.cat(c, 0)
    m = cc.min(0)[0]
    s = (cc.max(0)[0] - m).float()
    for i in range(len(c)):
        c[i] = (
            torch.div(
                (c[i] -
                 m.expand_as(
                    c[i])).float(),
                s.expand_as(
                    c[i])) *
            255.99).byte()
    return c


for char in range(1, 183 + 1):
    for writer in range(0, 36):
        exec('c=' + open('tmp/' + str(char) + '.' +
                         str(p[writer]) + '.py', 'r').read())
        train.append({'input': rescaleCharacter(c), 'target': char})
for char in range(1, 183 + 1):
    for writer in range(36, 45):
        exec('c=' + open('tmp/' + str(char) + '.' +
                         str(p[writer]) + '.py', 'r').read())
        test.append({'input': rescaleCharacter(c), 'target': char})
print(len(train), len(test))
os.mkdir('pickle')
pickle.dump(train, open('pickle/train.pickle', 'wb'))
pickle.dump(test, open('pickle/test.pickle', 'wb'))
