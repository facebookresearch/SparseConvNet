# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set(CMAKE_VERBOSE_MAKEFILE on)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)

SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_MODULE_PATH}")
FIND_PACKAGE(Torch REQUIRED)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

FILE(GLOB luasrc *.lua)

SET(src SCN/init.cpp)

ADD_TORCH_PACKAGE(sparseconvnet "${src}" "${luasrc}")

LINK_DIRECTORIES("${Torch_INSTALL_LIB}")
TARGET_LINK_LIBRARIES(sparseconvnet TH)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../PyTorch/sparseconvnet/SCN)

FIND_PACKAGE(CUDA 7.5)

IF(CUDA_FOUND)
  # Detect CUDA architecture and get best NVCC flags
  INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/SCN/cmake/FindCudaArch.cmake)
  SELECT_NVCC_ARCH_FLAGS(NVCC_FLAGS_EXTRA)
  LIST(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})

  INCLUDE_DIRECTORIES("${Torch_INSTALL_INCLUDE}/THC")
  LINK_DIRECTORIES("${Torch_INSTALL_LIB}")

  FILE(GLOB src-cuda SCN/init.cu)

  CUDA_ADD_LIBRARY(cusparseconvnet MODULE ${src-cuda})
  TARGET_LINK_LIBRARIES(cusparseconvnet THC TH)
  IF(APPLE)
    SET_TARGET_PROPERTIES(inn PROPERTIES
      LINK_FLAGS "-undefined dynamic_lookup")
  ENDIF()

  ### Torch packages supposes libraries prefix is "lib"
  SET_TARGET_PROPERTIES(cusparseconvnet PROPERTIES
    PREFIX "lib"
    IMPORT_PREFIX "lib")

  INSTALL(TARGETS cusparseconvnet
    RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
    LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")
ENDIF(CUDA_FOUND)
