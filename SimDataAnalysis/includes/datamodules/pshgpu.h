/*
 * pshgpu.h
 *
 *  Created on: Jul 7, 2011
 *      Author: consciousness
 */

#ifndef PSHGPU_H_
#define PSHGPU_H_

#include "psh.h"

class PSHDataGPU : public PSHData
{
public:
	PSHDataGPU(ifstream &infile);

	void exportPSH(ofstream &outfile);
protected:
	unsigned int cudaNBlocks;
	unsigned int cudaNThreadPerB;

private:
	PSHDataGPU();
};

#endif /* PSHGPU_H_ */
