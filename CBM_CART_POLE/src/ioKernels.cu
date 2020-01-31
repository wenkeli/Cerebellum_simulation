/*
 * ioKernels.cu
 *
 *  Created on: Mar 31, 2010
 *      Author: wen
 */

#include "../includes/ioKernels.h"
#include "commonCUDAKernels.cu"


__global__ void updatePFPCSynIO(float *synWPFPC, unsigned char *historyGR, int binPitch, unsigned short binN, int offset, unsigned char doLTD)
{
	int i=blockIdx.x*blockDim.x+threadIdx.x+offset;
	unsigned char *checkBinGR=(unsigned char *)((char *)historyGR+binN*binPitch);
	synWPFPC[i]=synWPFPC[i]+checkBinGR[i]*(doLTD*PFPCLTDDECPF+(!doLTD)*PFPCLTPINCPF);
	synWPFPC[i]=(synWPFPC[i]>0)*synWPFPC[i];
	synWPFPC[i]=(synWPFPC[i]>1)+(synWPFPC[i]<=1)*synWPFPC[i];
}


void runIOKernels()
{
	for(int i=0; i<NUMIO; i++)
	{
		updatePFPCSynIO<<<CUDAGRIONUMTBLOCK, CUDAGRIONUMTHREAD>>>(pfSynWeightPCGPU, historyGRGPU, histGRGPUPitch, (histBinNGR+1)%NUMHISTBINSGR, i*(NUMGR/NUMIO), plasticityPFPCTimerIO[i]<0);
		//<<<CUDAGRIONUMTBLOCK, CUDAGRIONUMTHREAD>>>
		plasticityPFPCTimerIO[i]=plasticityPFPCTimerIO[i]+HISTBINWIDTHGR;
	}
}
