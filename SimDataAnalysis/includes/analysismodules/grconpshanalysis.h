/*
 * grconpshanalysis.h
 *
 *  Created on: Sep 7, 2011
 *      Author: consciousness
 */

#ifndef GRCONPSHANALYSIS_H_
#define GRCONPSHANALYSIS_H_

#include <vector>

#include "../datamodules/psh.h"
#include "../datamodules/siminnet.h"


class GRConPSHAnalysis
{
public:
	GRConPSHAnalysis(PSHData *goP, PSHData *mfP, SimInNet *net);

	void getGRInMFGOPSHs(unsigned int grInd,
			vector<QPixmap *> &goPSHs, vector<QPixmap *> &mfPSHs);

	void getGROutGOPSHs(unsigned int grInd, vector<QPixmap *> &goPSHs);

protected:
	PSHData *goPSH;
	PSHData *mfPSH;

	SimInNet *inNet;

private:
	GRConPSHAnalysis();
};

#endif /* GRCONPSHANALYSIS_H_ */
