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
	~GRPSHPopAnalysis();

	void calcPFPCPlast(unsigned int usTime);

	void exportPFPCPlastAct(ofstream &outfile);

protected:
	const unsigned int **grPSH;
	float **grPSHNormalized;
	unsigned int numTrials;
	unsigned int preStimNumBins;
	unsigned int stimNumBins;
	unsigned int postStimNumBins;
	unsigned int totalNumBins;
	unsigned int binTimeSize;
	unsigned int pshMaxVal;
	unsigned int numGR;

	float *refPFPCPopAct;
	float *curItePFPCPopActLTD;
	float *curItePFPCPopActLTP;
	float *curItePFPCPopActBGAdj;

	float *refPFPCSynW;
	float *curPFPCSynW;

	void calcPFPCPopActivity(float *actvivity, float *pfPCSynW);
	float calcPFPCPopActivityBin(float *pfPCSynW, float *pshRow);

	void runPFPCPlastIterationNew(unsigned int usTime);

	void runPFPCPlastIteration(unsigned int usTime);
	void doPFPCPlast(float plastStep, const float *pshRow, float *pfPCSynW);

	void adjustPFPCBG();


private:
	GRPSHPopAnalysis();
};

#endif /* GRPOPANALYSIS_H_ */
