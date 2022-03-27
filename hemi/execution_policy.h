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

#include "hemi/hemi.h"

namespace hemi {

class ExecutionPolicy {
public:
    enum ConfigurationState {
        Automatic = 0,
        SharedMem = 1,
        BlockSize = 2,
        GridSize = 4,
        FullManual = GridSize | BlockSize | SharedMem
    };

    ExecutionPolicy() 
    : mState(Automatic), 
      mGridSize(0,1,1), 
      mBlockSize(0,0,0), 
      mSharedMemBytes(0),
      mStream((hemiStream_t)0) {}
    
    ExecutionPolicy(int gridSize, int blockSize, size_t sharedMemBytes)
    : mState(0), mStream(0) {
      setGridSize(gridSize);
      setBlockSize(blockSize);
      setSharedMemBytes(sharedMemBytes);  
    }

    ExecutionPolicy(int gridSize, int blockSize, size_t sharedMemBytes, hemiStream_t stream)
    : mState(0) {
      setGridSize(gridSize);
      setBlockSize(blockSize);
      setSharedMemBytes(sharedMemBytes);
      setStream(stream);
    }
          
    ~ExecutionPolicy() {}

    int    getConfigState()    const { return mState;          }
    
    dim3   getGridSize()       const { return mGridSize;       }
    dim3   getBlockSize()      const { return mBlockSize;      }
    dim3   getMaxBlockSize()   const { return mMaxBlockSize;   }
    size_t getSharedMemBytes() const { return mSharedMemBytes; }
    hemiStream_t getStream()   const { return mStream; }
 
    void setGridSize(int arg) { 
        mGridSize.x = arg;
        mGridSize.y = 1;
        mGridSize.z = 1;
        if (arg > 0) mState |= GridSize; 
        else mState &= (FullManual - GridSize);
    }   
    void setBlockSize(int arg) {
        mBlockSize.x = arg; 
        mBlockSize.y = 1; 
        mBlockSize.z = 1; 
        if (arg > 0) mState |= BlockSize; 
        else mState &= (FullManual - BlockSize);
    }
    void setMaxBlockSize(int arg) {
    	mMaxBlockSize.x = arg;
    	mMaxBlockSize.y = 1;
    	mMaxBlockSize.z = 1;
    }
    void setSharedMemBytes(size_t arg) { 
        mSharedMemBytes = arg; 
        mState |= SharedMem; 
    }
    void setStream(hemiStream_t stream) {
        mStream = stream;
    }

private:
    int    mState;
    dim3   mGridSize;
    dim3   mBlockSize;
    dim3   mMaxBlockSize;
    size_t mSharedMemBytes;
    hemiStream_t mStream;
};

}
