# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import glob, plyfile, numpy as np, multiprocessing as mp, torch

#label_ids are the IDs defined in ScanNet dataset
#class_ids are the IDs mapped from label_ids, for the network (starting from 0 and incrementing) where UNKNOWN_ID is the ignored class_id

SELECTED_LABEL_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]

LABEL_ID_TO_CLASS_ID = {}
for i, label_id in enumerate(SELECTED_LABEL_IDS):
    LABEL_ID_TO_CLASS_ID[label_id] = i
UNKNOWN_ID = -100

files=sorted(glob.glob('*/*_vh_clean_2.ply'))
files2=sorted(glob.glob('*/*_vh_clean_2.labels.ply'))
assert len(files) == len(files2)

def f(fn):
    fn2 = fn[:-3]+'labels.ply'

    #train file
    a=plyfile.PlyData().read(fn)
    v=np.array([list(x) for x in a.elements[0]])
    coords=np.ascontiguousarray(v[:,:3]-v[:,:3].mean(0))
    colors=np.ascontiguousarray(v[:,3:6])/127.5-1

    #val file
    a=plyfile.PlyData().read(fn2)
    w = np.array(list(map(lambda label_id: LABEL_ID_TO_CLASS_ID[label_id] if label_id in LABEL_ID_TO_CLASS_ID else UNKNOWN_ID, a.elements[0]['label'])))
    v=np.array([list(x) for x in a.elements[0]])
    label_colors=np.ascontiguousarray(v[:,3:6])

    torch.save((coords,colors,w,label_colors),fn[:-4]+'.pth')
    print(fn, fn2)

p = mp.Pool(processes=mp.cpu_count())
p.map(f,files)
p.close()
p.join()
