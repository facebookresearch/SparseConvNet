-- Copyright 2016-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.

package = "sparseconvnet"
version = "0.1-0"

source = {
  url = "",
  tag = "master"
}

description = {
  summary = "Submanifold Sparse ConvNets for Torch",
  detailed = [[
  ]],
  homepage = "",
  license = "CC-BY-NC"
}

dependencies = {
  "torch >= 7.0",
  "nn",
}

build = {
  type = "command",
  build_command = [[
  cmake -E make_directory build;
  cd build;
  cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"
  ]],
  install_command = "cd build && $(MAKE) install"
}
