/*
 * commonCUDAKernels.cu
 *
 *  Created on: Feb 8, 2010
 *      Author: wen
 */
#ifndef COMMONCUDAKERNELS_CU_
#define COMMONCUDAKERNELS_CU_

#include "../includes/commonCUDAKernels.h"

template <unsigned int blockSize, typename Type> __global__ void sumInputs(Type *input, unsigned int inputPitch, Type *output, unsigned int outputPitch)
{
	__shared__ Type sData[blockSize];

	int tid=threadIdx.x;
	int index=blockIdx.x*(blockDim.x*2)+tid;
	Type *inputRow;

	if(inputPitch>1)
	{
		inputRow=(Type *)((char *)input+blockIdx.y*inputPitch);
	}
	else
	{
		inputRow=input+blockIdx.y;
	}

	sData[tid]=inputRow[index]+inputRow[index+blockDim.x];
	__syncthreads();

	if(blockSize>=512)
	{
		if(tid<256)
			sData[tid]+=sData[tid+256];
	}
	__syncthreads();

	if(blockSize>=256)
	{
		if(tid<128)
			sData[tid]+=sData[tid+128];
	}
	__syncthreads();

	if(blockSize>=128)
	{
		if(tid<64)
			sData[tid]+=sData[tid+64];
	}
	__syncthreads();

	if(tid<32)
	{
		if(blockSize>=64)
			sData[tid]+=sData[tid+32];
		if(blockSize>=32)
			sData[tid]+=sData[tid+16];
		if(blockSize>=16)
			sData[tid]+=sData[tid+8];
		if(blockSize>=8)
			sData[tid]+=sData[tid+4];
		if(blockSize>=4)
			sData[tid]+=sData[tid+2];
		if(blockSize>=2)
			sData[tid]+=sData[tid+1];
	}
	if(tid==0)
	{
		Type *outputRow;
		if(outputPitch>1)
		{
			outputRow=(Type *)((char *)output+blockIdx.y*outputPitch);
		}
		else
		{
			outputRow=output+blockIdx.y;
		}
		outputRow[blockIdx.x]=sData[0];
	}
}

#endif
