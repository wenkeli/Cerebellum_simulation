/*
 * pshgpu.cpp
 *
 *  Created on: Jul 1, 2011
 *      Author: consciousness
 */

#include "../../includes/analysismodules/pshgpu.h"

PSHAnalysisGPU::PSHAnalysisGPU(unsigned int nCells, const unsigned int *buf,
		unsigned int preSNBins, unsigned int sNBins, unsigned int postSNBins,
		unsigned int binSize, unsigned int bufSize, unsigned int nBinsInBuf,
		unsigned int cudaNB, unsigned int cudaNTPB)
		:PSHAnalysis(nCells, buf, preSNBins, sNBins, postSNBins, binSize, bufSize, nBinsInBuf)
{
	cudaNBlocks=cudaNB;
	cudaNThreadPerB=cudaNTPB;

	initCUDA();
}

PSHAnalysisGPU::PSHAnalysisGPU(ifstream &infile, const unsigned int *buf)
		:PSHAnalysis(infile, buf)
{
	infile.read((char *)&cudaNBlocks, sizeof(unsigned int));
	infile.read((char *)&cudaNThreadPerB, sizeof(unsigned int));
	initCUDA();
}

PSHAnalysisGPU::~PSHAnalysisGPU()
{
	cudaFree(pshBufGPU);
}

void PSHAnalysisGPU::exportPSH(ofstream &outfile)
{
	PSHAnalysis::exportPSH(outfile);

	outfile.write((char *)&cudaNBlocks, sizeof(unsigned int));
	outfile.write((char *)&cudaNThreadPerB, sizeof(unsigned int));
}

void PSHAnalysisGPU::initCUDA()
{
	cudaMallocPitch((void **)&pshBufGPU, (size_t *)&pshBufGPUP, numCells*sizeof(unsigned int), numBinsInBuf);
}

void PSHAnalysisGPU::updatePSH()
{
	unsigned int extrashift;
	extrashift=apBufTimeSize-(numBinsInBuf*binTimeSize);

	cudaMemcpy2D(pshBufGPU, (size_t)pshBufGPUP,
			pshData[currBinN], numCells*sizeof(unsigned int),
			numCells*sizeof(unsigned int), numBinsInBuf, cudaMemcpyHostToDevice);

	callUpdatePSHGPU((unsigned int *)apBuf, pshBufGPU, pshBufGPUP, (int)numBinsInBuf, (int)binTimeSize, extrashift, cudaNBlocks, cudaNThreadPerB);

	cudaMemcpy2D(pshData[currBinN], numCells*sizeof(unsigned int),
			pshBufGPU, (size_t)pshBufGPUP,
			numCells*sizeof(unsigned int), numBinsInBuf, cudaMemcpyDeviceToHost);

	currBinN=currBinN+numBinsInBuf;

	if(currBinN>=totalNumBins)
	{
		numTrials++;
		currBinN=0;
	}

}
