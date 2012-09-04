/*
 * ecmanagementdelay.cpp
 *
 *  Created on: Sep 4, 2012
 *      Author: consciousness
 */

#include "../../includes/ectrial/ecmanagementdelay.h"

ECManagementDelay::ECManagementDelay(int numT, int iti, int csOn, int csOff, int csPOff,
		int csStartTN, int dataStartTN, int nDataT,
		float fracCSTMF, float fracCSPMF, float fracCtxtMF,
		float bgFreqMin, float csBGFreqMin, float CtxtFreqMin, float csTFreqMin, float csPFreqMin,
		float bgFreqMax, float csBGFreqMax, float CtxtFreqMax, float csTFreqMax, float csPFreqMax)
		:ECManagementBase(numT, iti)
{
	csOnTime=csOn;
	csOffTime=csOff;
	csPOffTime=csPOff;

	csStartTrialN=csStartTN;
	dataStartTrialN=dataStartTN;
	numDataTrials=nDataT;


}


