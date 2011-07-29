/*
 * spikerateanalysis.h
 *
 *  Created on: Jul 29, 2011
 *      Author: consciousness
 */

#ifndef SPIKERATEANALYSIS_H_
#define SPIKERATEANALYSIS_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "../datamodules/psh.h"

class SpikeRateAnalysis
{
public:
	SpikeRateAnalysis(PSHData *pshData);
	~SpikeRateAnalysis();

	void calcSpikeRates();

	void exportSpikeRates(ofstream &outfile);
protected:
	const unsigned int **psh;
	unsigned int numTrials;
	unsigned int preStimNumBins;
	unsigned int stimNumBins;
	unsigned int postStimNumBins;
	unsigned int binTimeSize;
	unsigned int numCells;

	float *preStimSR;
	float *stimulusSR;
	float *postStimSR;

	void calcSR(float *spikeRates, unsigned int startBinN, unsigned int nBins);

private:
	SpikeRateAnalysis();
};

#endif /* SPIKERATEANALYSIS_H_ */
