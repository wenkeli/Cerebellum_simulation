/*
 * bcKernels.cu
 *
 *  Created on: Feb 25, 2010
 *      Author: wen
 */

#include "../includes/bcKernels.h"
#include "commonCUDAKernels.cu"

__global__ void calcPFBC(unsigned char *inputGR, unsigned short *inputPFBC, unsigned int iPFBCPitch)
{
	unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
	unsigned short *inputPFBCRow=(unsigned short *)((char *)inputPFBC+(i/PFBCSYNPERBC)*iPFBCPitch);
	inputPFBCRow[i%PFBCSYNPERBC]=(inputGR[i]&0x08)>>3;
}

void runBCKernels()
{
	calcPFBC<<<CUDAGRNUMTBLOCK, CUDAGRNUMTHREAD>>>(apOutGRGPU, inputPFBCGPU, iPFBCGPUPitch);
	cudaThreadSynchronize();

	dim3 dimGrid1(8, NUMBC); //TODO: generalize the literal numbers
	sumInputs<512, unsigned short><<<dimGrid1, 512>>>(inputPFBCGPU, iPFBCGPUPitch, tempSumPFBCGPU, tempSumPFBCGPUPitch);
	cudaThreadSynchronize();
	dim3 dimGrid2(1, NUMBC);
	sumInputs<4, unsigned short><<<dimGrid2, 4>>>(tempSumPFBCGPU, tempSumPFBCGPUPitch, inputSumPFBCGPU, 1);
	cudaThreadSynchronize();
}
