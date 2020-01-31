/*
 * pcKernels.cu
 *
 *  Created on: Feb 1, 2010
 *      Author: wen
 */

#include "../includes/pcKernels.h"
#include "commonCUDAKernels.cu"

__global__ void calcPFPC(float *synWPFPC, unsigned char *apGRInput, float *inputPFPC, unsigned int iPFPCPitch)
{
	unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
	float *inputPFPCRow=(float *)((char*)inputPFPC+(i/PFPCSYNPERPC)*iPFPCPitch);
	inputPFPCRow[i%PFPCSYNPERPC]=synWPFPC[i]*((apGRInput[i]&0x08)>>3);
}

void runPCKernels()
{
	calcPFPC<<<CUDAGRNUMTBLOCK, CUDAGRNUMTHREAD>>>(pfSynWeightPCGPU, apOutGRGPU, inputPFPCGPU, iPFPCGPUPitch);
	cudaThreadSynchronize();

	dim3 dimGrid1(32, NUMPC); //TODO: generalize the literal numbers
	sumInputs<512, float><<<dimGrid1, 512>>>(inputPFPCGPU, iPFPCGPUPitch, tempSumPFPCGPU, tempSumPFPCGPUPitch);
	cudaThreadSynchronize();
	dim3 dimGrid2(1, NUMPC);
	sumInputs<16, float><<<dimGrid2, 16>>>(tempSumPFPCGPU, tempSumPFPCGPUPitch, inputSumPFPCGPU, 1);
	cudaThreadSynchronize();
}
