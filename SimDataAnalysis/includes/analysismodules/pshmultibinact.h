/*
 * pshmultibinact.h
 *
 *  Created on: May 9, 2012
 *      Author: consciousness
 */

#ifndef PSHMULTIBINACT_H_
#define PSHMULTIBINACT_H_

#include <math.h>
#include <iostream>
#include "../datamodules/psh.h"

class PSHMultiBinAct
{
public:
	PSHMultiBinAct(unsigned int startBN, unsigned int endBN, PSHData *psh);
	~PSHMultiBinAct();

	int getNumTrials();
	int getNumCells();
	int getNumBins();
	int getStartBinN();
	int getEndBinN();
	int getTotalTimeLen();
	const float* getActData();
	const float* getActDataNormToTrials();

private:
	PSHMultiBinAct();

	unsigned int startBinN;
	unsigned int endBinN;

	int numTrials;
	int numCells;
	int numBins;
	int totalTimeLen;
	float *actData;
	float *actDataNormToTrials;
};


#endif /* PSHMULTIBINACT_H_ */
