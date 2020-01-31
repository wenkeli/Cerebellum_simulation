/*
 * pshgpu.h
 *
 *  Created on: Jul 1, 2011
 *      Author: consciousness
 */

#ifndef PSHGPU_H_
#define PSHGPU_H_

#include "psh.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include "../cuda/kernels.h"


class PSHAnalysisGPU : public PSHAnalysis
{
public:
	PSHAnalysisGPU(unsigned int nCells, const unsigned int *buf,
			unsigned int preSNBins, unsigned int sNBins, unsigned int postSNBins,
			unsigned int binSize, unsigned int bufSize, unsigned int nBinsInBuf,
			unsigned int cudaNB, unsigned int cudaNTPB);
	PSHAnalysisGPU(ifstream &infile, const unsigned int *buf);

	~PSHAnalysisGPU();

	void exportPSH(ofstream &outfile);

	void updatePSH();

private:
	PSHAnalysisGPU();

	void initCUDA();

	unsigned int cudaNBlocks;
	unsigned int cudaNThreadPerB;

	unsigned int *pshBufGPU;
	unsigned int pshBufGPUP;
};

#endif /* PSHGPU_H_ */
