/*
 * kernels.h
 *
 *  Created on: Jun 6, 2011
 *      Author: consciousness
 */

#ifndef KERNELS_H_
#define KERNELS_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>

#include <iostream>
#include "../parameters.h"

void callGRActKernel(cudaStream_t &st, unsigned int nBlocks, unsigned int nThreadPerB, float *vGPU, float *gKCaGPU, float *threshGPU,
		unsigned int *apBufGPU, unsigned char *apOutGRGPU, float *gESumGPU, float *gISumGPU,
		float gLeak, float eLeak, float eGOIn,
		float threshBase, float threshMax, float threshDecay);

template<typename Type, unsigned int blockSize, bool inMultiP, bool outMultiP>
void callSumPFKernel(cudaStream_t &st, Type *inPFGPU, unsigned int inPFGPUP, Type *inPFSumGPU, unsigned int inPFSumGPUP,
		unsigned int nOutCells, unsigned int nOutCols, unsigned int rowLength);

template<unsigned int nCols, unsigned int nBlocks, unsigned int nThreadPerB>
void callSumGRGOOutKernel(cudaStream_t &st, unsigned int *grInGOGPU, unsigned int grInGOGPUPitch,
		unsigned int *grInGOSGPU);

template<unsigned int nCellIn, unsigned int nBlocks, unsigned int nThreadsPerB>
void callUpdateInGRKernel(cudaStream_t &st, unsigned int *apInGPU, float *gGPU, unsigned int gGPUP,
		unsigned int *conInGRGPU, unsigned int conInGRGPUP,
		int *numInPerGRGPU, float *gSumGPU, float gDecay, float gInc);

template<unsigned int numPFInPerBC, unsigned int numPFInPerSC, unsigned int numBlocks, unsigned int numThreadsPerB>
void callUpdatePFBCSCOutKernel(cudaStream_t &st, unsigned int *apBufGPU, unsigned int *delayMaskGPU,
		unsigned int *inPFBCGPU, unsigned int inPFBCGPUPitch,
		unsigned int *inPFSCGPU, unsigned int inPFSCGPUPitch);

template<unsigned int numPFInPerPC, unsigned int numBlocks, unsigned int numThreadsPerB>
void callUpdatePFPCOutKernel(cudaStream_t &st, unsigned int *apBufGPU, unsigned int *delayMaskGPU,
		float *pfPCSynWGPU, float *inPFPCGPU, unsigned int inPFPCGPUPitch);

template<unsigned int nGO, unsigned int nRows, unsigned int threadsPerRow>
void callUpdateGROutGOKernel(cudaStream_t &st, unsigned int *apBufGPU,
		unsigned int *grInGOGPU, unsigned int grInGOGPUPitch,
		unsigned int *delayMasksGPU, unsigned int delayMasksGPUPitch,
		unsigned int *conGRtoGOGPU, unsigned int conGRtoGOGPUPitch,
		int *numGOPerGRGPU);

template<unsigned int numBlocks, unsigned int numThreadsPerB>
void callUpdateGRHistKernel(cudaStream_t &st, unsigned int *apBufGPU, unsigned long *historyGPU);

template<unsigned int numBlocks, unsigned int numThreadsPerB>
void callUpdatePFPCPlasticityIOKernel(cudaStream_t &st, int plastTimerIO, float *synWeightGPU,
		unsigned long *historyGPU, int offSet, float pfPCLTDStep, float pfPCLTPStep);

void callUpdatePSHGPU(unsigned int *apBufGPU, unsigned int *pshGPU, unsigned int pshGPUP,
		int nBins, int tsPerBin, unsigned int extrashift, unsigned int nBlocks, unsigned int nThreadPerB);

#endif /* KERNELS_H_ */
