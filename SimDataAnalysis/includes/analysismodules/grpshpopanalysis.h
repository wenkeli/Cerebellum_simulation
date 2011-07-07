/*
 * grpopanalysis.h
 *
 *  Created on: Jul 7, 2011
 *      Author: consciousness
 */

#ifndef GRPOPANALYSIS_H_
#define GRPOPANALYSIS_H_
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>
#include <ctime>

#include "../datamodules/psh.h"

class GRPSHPopAnalysis
{
public:
	GRPSHPopAnalysis(PSHData *grPSH);
protected:
	const unsigned int **grPSH;
	unsigned int numTrials;
	unsigned int preStimNumBins;
	unsigned int stimNumBins;
	unsigned int postStimNumBins;
	unsigned int totalNumBins;
	unsigned int binTimeSize;
	unsigned int pshMaxVal;
	unsigned int numGR;

	unsigned int *totalActivity;
	unsigned int *pfPCSynW;

private:
	GRPSHPopAnalysis();
};

#endif /* GRPOPANALYSIS_H_ */
