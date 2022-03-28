///////////////////////////////////////////////////////////////////////////////
// 
// "Hemi" CUDA Portable C/C++ Utilities
// 
// Copyright 2012-2015 NVIDIA Corporation
//
// License: BSD License, see LICENSE file in Hemi home directory
//
// The home for Hemi is https://github.com/harrism/hemi
//
///////////////////////////////////////////////////////////////////////////////
// Please see the file README.md (https://github.com/harrism/hemi/README.md) 
// for full documentation and discussion.
///////////////////////////////////////////////////////////////////////////////
#pragma once

// Functions available inside "device" code (whether compiled for CUDA device
// or CPU.)

#include "hemi.h"

#ifdef HEMI_DEV_CODE
  #define HEMI_SHARED __shared__
#else
  #define HEMI_SHARED
#endif

namespace hemi
{
    HEMI_DEV_CALLABLE_INLINE
    unsigned int globalThreadIndex(int dim=0) {
    #ifdef HEMI_DEV_CODE
        if(dim == 0)
            return threadIdx.x + blockIdx.x * blockDim.x;
        if(dim == 1)
            return threadIdx.y + blockIdx.y * blockDim.y;
        if(dim == 2)
            return threadIdx.z + blockIdx.z * blockDim.z;
    #endif
    return 0;
    }


    HEMI_DEV_CALLABLE_INLINE
    unsigned int globalThreadCount(int dim=0) {
    #ifdef HEMI_DEV_CODE
        if(dim == 0)
            return blockDim.x * gridDim.x;
        if(dim == 1)
            return blockDim.y * gridDim.y;
        if(dim == 2)
            return blockDim.z * gridDim.z;
    #endif
    return 1;
    }


    HEMI_DEV_CALLABLE_INLINE
    unsigned int globalBlockCount(int dim=0) {
    #ifdef HEMI_DEV_CODE
        if(dim == 0)
            return gridDim.x;
        if(dim == 1)
            return gridDim.y;
        if(dim == 2)
            return gridDim.z;
    #endif
    return 1;
    }


    HEMI_DEV_CALLABLE_INLINE
    unsigned int localThreadIndex(int dim=0) {
    #ifdef HEMI_DEV_CODE
        if(dim == 0)
            return threadIdx.x;
        if(dim == 1)
            return threadIdx.y;
        if(dim == 2)
            return threadIdx.z;
    #endif
    return 0;
    }


    HEMI_DEV_CALLABLE_INLINE
    unsigned int localThreadCount(int dim=0) {
    #ifdef HEMI_DEV_CODE
        if(dim == 0)
            return blockDim.x;
        if(dim == 1)
            return blockDim.y;
        if(dim == 2)
            return blockDim.z;
    #endif
    return 1;
    }


    HEMI_DEV_CALLABLE_INLINE
    unsigned int globalBlockIndex(int dim=0) {
    #ifdef HEMI_DEV_CODE
        if(dim == 0)
            return blockIdx.x;
        if(dim == 1)
            return blockIdx.y;
        if(dim == 2)
            return blockIdx.z;
    #endif
    return 0;
    }


    HEMI_DEV_CALLABLE_INLINE
    void synchronize() {
    #ifdef HEMI_DEV_CODE
        __syncthreads();
    #endif
    }
}
