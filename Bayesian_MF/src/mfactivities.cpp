/*
 * mfactivities.cpp
 *
 *  Created on: Feb 7, 2011
 *      Author: consciousness
 */

#include "../includes/mfactivities.h"

void calcMFActsPoisson(int inputType, CRandomSFMT0 &randGen)
{
	float *inputSRs;

	if(inputType>0)
	{
		inputSRs=ratesMFInputA;
	}
	else
	{
		inputSRs=ratesMFInputB;
	}


	memset(spikeCountsMF, 0, NUMMF*sizeof(int));

	for(int t=0; t<TRIALTIMEMS; t++)
	{
		for(int i=0; i<NUMMF; i++)
		{
			bool ap;
			ap=randGen.Random()<(inputSRs[i]*TBINLENSEC);
			spikeCountsMF[i]=spikeCountsMF[i]+ap;
		}
	}
}

void calcMFActsRegenPoisson(int inputType, CRandomSFMT0 &randGen)
{
	float *inputSRs;
	float threshMF[NUMMF];

	if(inputType>0)
	{
		inputSRs=ratesMFInputA;
	}
	else
	{
		inputSRs=ratesMFInputB;
	}

	memset(spikeCountsMF, 0, NUMMF*sizeof(int));
	for(int i=0; i<NUMMF; i++)
	{
		threshMF[i]=1;
	}

	for(int t=0; t<TRIALTIMEMS; t++)
	{
		for(int i=0; i<NUMMF; i++)
		{
			bool ap;
			threshMF[i]=threshMF[i]+(1-threshMF[i])*threshDecayMF;

			ap=randGen.Random()<((inputSRs[i]*TBINLENSEC)*threshMF[i]);
			spikeCountsMF[i]=spikeCountsMF[i]+ap;
			threshMF[i]=(!ap)*threshMF[i];
		}
	}
}
