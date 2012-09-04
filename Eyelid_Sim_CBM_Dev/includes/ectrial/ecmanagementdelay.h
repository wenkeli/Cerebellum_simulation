/*
 * ecmanagementdelay.h
 *
 *  Created on: Sep 4, 2012
 *      Author: consciousness
 */

#ifndef ECMANAGEMENTDELAY_H_
#define ECMANAGEMENTDELAY_H_

#include "ecmanagementbase.h"

#include <CBMDataInclude/interfaces/ecrastertrial.h>

#include <CBMCoreInclude/tools/randomc.h>
#include <CBMCoreInclude/tools/sfmt.h>

class ECManagementDelay : public ECManagementBase
{
public:
	ECManagementDelay(int numT, int iti, int csOn, int csOff, int csPOff,
			int csStartTN, int dataStartTN, int nDataT,
			float fracCSTMF, float fracCSPMF, float fracCtxtMF,
			float bgFreqMin, float csBGFreqMin, float CtxtFreqMin, float csTFreqMin, float csPFreqMin,
			float bgFreqMax, float csBGFreqMax, float CtxtFreqMax, float csTFreqMax, float csPFreqMax);

	virtual ~ECManagementDelay();

protected:
	virtual void calcMFActivity();
	virtual void calcSimActivity();

	int csOnTime;
	int csOffTime;
	int csPOffTime;

	int csStartTrialN;
	int dataStartTrialN;
	int numDataTrials;

	float fracCSTonicMF;
	float fracCSPhasicMF;
	float fracContextMF;

	float backgroundFreqMin;
	float csBackgroundFreqMin;
	float contextFreqMin;
	float csTonicFreqMin;
	float csPhasicFreqMin;

	float backgroundFreqMax;
	float csBackgroundFreqMax;
	float contextFreqMax;
	float csTonicFreqMax;
	float csPhasicFreqMax;

//	float *mfBGFreq;
	float *mfCSTonicFreq;
	float *mfCSPhasicFreq;

private:
	ECManagementDelay();
};


#endif /* ECMANAGEMENTDELAY_H_ */
