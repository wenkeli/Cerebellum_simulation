/*
 * ecmanagementdelay.h
 *
 *  Created on: Sep 4, 2012
 *      Author: consciousness
 */

#ifndef ECMANAGEMENTDELAY_H_
#define ECMANAGEMENTDELAY_H_

#include "ecmanagementbase.h"

#include <time.h>
#include <fstream>

#include <CXXToolsInclude/stdDefinitions/pstdint.h>
#include <CXXToolsInclude/randGenerators/sfmt.h>

#include <CBMToolsInclude/poissonregencells.h>

#include <CBMStateInclude/interfaces/cbmstate.h>
#include <CBMStateInclude/interfaces/iconnectivityparams.h>

#include <CBMCoreInclude/interface/cbmsimcore.h>
#include <CBMCoreInclude/interface/innetinterface.h>
#include <CBMCoreInclude/interface/mzoneinterface.h>

#include <CBMDataInclude/interfaces/ecrastertrial.h>


class ECManagementDelay : public ECManagementBase
{
public:
	ECManagementDelay(std::ofstream *dfOut, int numT, int iti, int csOn, int csOff, int csPOff,
			int csStartTN, int dataStartTN, int nDataT,
			float fracCSTMF, float fracCSPMF, float fracCtxtMF,
			float bgFreqMin, float csBGFreqMin, float ctxtFreqMin, float csTFreqMin, float csPFreqMin,
			float bgFreqMax, float csBGFreqMax, float ctxtFreqMax, float csTFreqMax, float csPFreqMax);

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

	float backGFreqMin;
	float csBackGFreqMin;
	float contextFreqMin;
	float csTonicFreqMin;
	float csPhasicFreqMin;

	float backGFreqMax;
	float csBackGFreqMax;
	float contextFreqMax;
	float csTonicFreqMax;
	float csPhasicFreqMax;

	float *mfFreqInCSTonic;
	float *mfFreqInCSPhasic;

	std::ofstream *dataFileOut;

private:
	ECManagementDelay();
};


#endif /* ECMANAGEMENTDELAY_H_ */
