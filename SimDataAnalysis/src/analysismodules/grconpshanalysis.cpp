/*
 * grconpshanalysis.cpp
 *
 *  Created on: Sep 7, 2011
 *      Author: consciousness
 */

#include "../../includes/analysismodules/grconpshanalysis.h"

GRConPSHAnalysis::GRConPSHAnalysis(PSHData *goP, PSHData *mfP, SimInNet *net)
{
	goPSH=goP;
	mfPSH=mfP;
	inNet=net;
}

void GRConPSHAnalysis::getGRInMFGOPSHs(unsigned int grInd,
		vector<QPixmap *> &goPSHs, vector<QPixmap *> &mfPSHs)
{
	vector<unsigned int> goInds;
	vector<unsigned int> mfInds;

	inNet->getGRInMFGOInds(grInd, goInds, mfInds);

	for(int i=0; i<goInds.size(); i++)
	{
		goPSHs.push_back(goPSH->paintPSHInd(goInds[i]));
	}

	for(int i=0; i<mfInds.size(); i++)
	{
		mfPSHs.push_back(mfPSH->paintPSHInd(mfInds[i]));
	}
}

void GRConPSHAnalysis::getGROutGOPSHs(unsigned int grInd, vector<QPixmap *> &goPSHs)
{
	vector<unsigned int> goInds;

	inNet->getGROutGOInds(grInd, goInds);

	for(int i=0; i<goInds.size(); i++)
	{
		goPSHs.push_back(goPSH->paintPSHInd(goInds[i]));
	}
}

