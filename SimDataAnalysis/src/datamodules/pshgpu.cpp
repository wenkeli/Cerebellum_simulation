/*
 * pshgpu.cpp
 *
 *  Created on: Jul 7, 2011
 *      Author: consciousness
 */

#include "../../includes/datamodules/pshgpu.h"

PSHDataGPU::PSHDataGPU(ifstream &infile):PSHData(infile)
{
	infile.read((char *)&cudaNBlocks, sizeof(unsigned int));
	infile.read((char *)&cudaNThreadPerB, sizeof(unsigned int));
}

void PSHDataGPU::exportPSH(ofstream &outfile)
{
	PSHData::exportPSH(outfile);

	outfile.write((char *)&cudaNBlocks, sizeof(unsigned int));
	outfile.write((char *)&cudaNThreadPerB, sizeof(unsigned int));
}

