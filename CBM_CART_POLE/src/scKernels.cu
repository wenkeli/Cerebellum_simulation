/*
 * scKernels.cu
 *
 *  Created on: Feb 25, 2010
 *      Author: wen
 */

#include "../includes/scKernels.h"
#include "commonCUDAKernels.cu"

__global__ void calcPFSC(unsigned char *inputGR, unsigned short *inputPFSC, unsigned int iPFSCPitch)
{
	unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
	unsigned short *inputPFSCRow=(unsigned short *)((char *)inputPFSC+(i/PFSCSYNPERSC)*iPFSCPitch);
	inputPFSCRow[i%PFSCSYNPERSC]=(inputGR[i]&0x08)>>3;
}

void runSCKernels()
{
	calcPFSC<<<CUDAGRNUMTBLOCK, CUDAGRNUMTHREAD>>>(apOutGRGPU, inputPFSCGPU, iPFSCGPUPitch);
	cudaThreadSynchronize();

	dim3 dimGrid1(2, NUMSC);
	sumInputs<512, unsigned short><<<dimGrid1, 512>>>(inputPFSCGPU, iPFSCGPUPitch, tempSumPFSCGPU, tempSumPFSCGPUPitch);
	cudaThreadSynchronize();
	dim3 dimGrid2(1, NUMSC);
	sumInputs<1, unsigned short><<<dimGrid2, 1>>>(tempSumPFSCGPU, tempSumPFSCGPUPitch, inputSumPFSCGPU, 1);
	cudaThreadSynchronize();
}

