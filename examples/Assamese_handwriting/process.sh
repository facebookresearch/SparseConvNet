# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#!/bin/bash
set -e
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00208/Online%20Handwritten%20Assamese%20Characters%20Dataset.rar
unrar e -cl -y "Online Handwritten Assamese Characters Dataset.rar"
mkdir tmp
for char in `seq 1 183`; do
  for writer in `seq 1 45`; do
    echo 'return {f{' > z
    tail -n+5 $char.$writer.txt|head -n-1 |grep -v 'PEN_UP'|uniq|sed 's/PEN_DOWN/},f{/'|sed 's/      /,/g'|cut -c1-12 >> z
    echo '}}' >>z
    echo 'f=function(x) return torch.LongTensor(x):view(-1,2) end' > tmp/$char.$writer.lua
    cat z |tr -d '\n' |sed 's/f{},//'>> tmp/$char.$writer.lua

    echo '[f([' > z
    tail -n+5 $char.$writer.txt|head -n-1 |grep -v 'PEN_UP'|uniq|sed 's/PEN_DOWN/]).v,f([/'|sed 's/      /,/g'|cut -c1-12 >> z
    echo ']).v]' >>z
    cat z |tr -d '\n' |sed 's/f(\[\]).v,//g'|sed 's/v/view(-1,2)/g'|sed 's/f/torch.LongTensor/g'> tmp/$char.$writer.py

    rm $char.$writer.txt z
  done
done
