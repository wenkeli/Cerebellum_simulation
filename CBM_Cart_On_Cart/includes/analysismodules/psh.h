/*
 * psh.h
 *
 *  Created on: Jun 30, 2011
 *      Author: consciousness
 */

#ifndef PSH_H_
#define PSH_H_
#include "../common.h"

class PSHAnalysis
{
public:
	PSHAnalysis(unsigned int nCells, const unsigned int *buf,
			unsigned int preSNBins, unsigned int sNBins, unsigned int postSNBins,
			unsigned int binSize, unsigned int bufSize, unsigned int nBinsInBuf);
	PSHAnalysis(ifstream &infile, const unsigned int *buf);
	virtual ~PSHAnalysis();

	virtual void exportPSH(ofstream &outfile);

	virtual void updatePSH();

	void resetCurrentBinN();

protected:
	unsigned int numCells;
	unsigned int preStimNumBins;
	unsigned int stimNumBins;
	unsigned int postStimNumBins;
	unsigned int totalNumBins;
	unsigned int binTimeSize;
	unsigned int apBufTimeSize;
	unsigned int numBinsInBuf;
	unsigned int numTrials;
	unsigned int currBinN;

	const unsigned int *apBuf;
	unsigned int **pshData;

private:
	PSHAnalysis();
};

#endif /* PSH_H_ */
