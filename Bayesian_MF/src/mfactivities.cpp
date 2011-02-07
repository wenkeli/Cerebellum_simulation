/*
 * mfactivities.cpp
 *
 *  Created on: Feb 7, 2011
 *      Author: consciousness
 */

#include "../includes/mfactivities.h"

void calcMFActsPoisson(int inputType, CRandomSFMT0 &randGen)
{
	int *inputSRs;

	if(inputType>0)
	{
		inputSRs=ratesMFInputA;
	}
	else
	{
		inputSRs=ratesMFInputB;
	}
}

void calcMFActsRegenPoisson(int inputType, CRandomSFMT0 &randGen)
{
	int *inputSRs;

	if(inputType>0)
	{
		inputSRs=ratesMFInputA;
	}
	else
	{
		inputSRs=ratesMFInputB;
	}
}
