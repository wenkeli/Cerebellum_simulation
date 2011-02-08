/*
 * bayesianeval.cpp
 *
 *  Created on: Feb 8, 2011
 *      Author: consciousness
 */

#include "../includes/bayesianeval.h"

void bayesianCalcSV(int trialN)
{
	float sv;

	double lnPRIA=0;
	double lnPRIB=0;
	double ratio;
	double pRIX;

	for(int i=0; i<NUMMF; i++)
	{
		lnPRIA=lnPRIA+lnPoisson(spikeCountsMF[i], ratesMFInputA[i]*TRIALTIMESEC);
		lnPRIB=lnPRIB+lnPoisson(spikeCountsMF[i], ratesMFInputB[i]*TRIALTIMESEC);
	}

	if(lnPRIA<=lnPRIB)
	{
		ratio=exp(lnPRIA-lnPRIB);
	}
	else
	{
		ratio=exp(lnPRIB-lnPRIA);
	}

	pRIX=1/(1+ratio);

	sv=(pRIX-0.5)/0.5;

	sVs[trialN]=sv;
}

double lnPoisson(unsigned int k, double lambda)
{
	double lnKFact=0;

	for(unsigned int i=k; i>1; i--)
	{
		lnKFact=lnKFact+log(i);
	}

	return k*log(lambda)-lambda-lnKFact;
}
