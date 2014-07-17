/*
 * ecmanagementdelay.h
 *
 *  Created on: Sep 4, 2012
 *      Author: consciousness
 */

#ifndef ECMANAGEMENTDELAY_H_
#define ECMANAGEMENTDELAY_H_

#include <time.h>
#include <fstream>
#include <string>

#include <CXXToolsInclude/stdDefinitions/pstdint.h>
#include <CXXToolsInclude/randGenerators/sfmt.h>

#include <CBMToolsInclude/poissonregencells.h>
#include <CBMToolsInclude/eyelidintegrator.h>
#include <CBMToolsInclude/ecmfpopulation.h>

#include <CBMCoreInclude/interface/cbmsimcore.h>
#include <CBMCoreInclude/interface/innetinterface.h>
#include <CBMCoreInclude/interface/mzoneinterface.h>

#include <CBMDataInclude/interfaces/ectrialsdata.h>

#include "ecmanagementbase.h"


class ECManagementDelay : public ECManagementBase
{
public:
	ECManagementDelay(std::string conParamFile, std::string actParamFile, int randSeed,
			int numT, int iti, int csOn, int csOff, int csPOff,
			int csStartTN, int dataStartTN, int nDataT,
			float fracCSTMF, float fracCSPMF, float fracCtxtMF,
			float bgFreqMin, float csBGFreqMin, float ctxtFreqMin, float csTFreqMin, float csPFreqMin,
			float bgFreqMax, float csBGFreqMax, float ctxtFreqMax, float csTFreqMax, float csPFreqMax,
			std::string dataFileName, int gpuIndStart=-1, int numGPUP2=-1);
	ECManagementDelay(std::string stateDataFile, int randSeed,
			int numT, int iti, int csOn, int csOff, int csPOff,
			int csStartTN, int dataStartTN, int nDataT,
			std::string dataFileName, int gpuIndStart=-1, int numGPUP2=-1);

	virtual ~ECManagementDelay();

	void writeDataToFile();

protected:
	virtual void calcMFActivity();
	virtual void calcSimActivity();

	ECMFPopulation *mfFreqs;
	PoissonRegenCells *mfs;

	float *mfFreqBG;
	float *mfFreqInCSPhasic;
	float *mfFreqInCSTonic;

	int rSeed;

	int csOnTime;
	int csOffTime;
	int csPOffTime;

	int csStartTrialN;
	int dataStartTrialN;
	int numDataTrials;

	bool grPCPlastSet;
	bool grPCPlastReset;

	EyelidIntegrator *eyelidFunc;

	ECTrialsData *data;

	std::string dataFileName;

private:
	ECManagementDelay();
	void initialize(int randSeed, int csOn, int csOff, int csPOff,
			int csStartTN, int dataStartTN, int nDataT, std::string dataFileName);
};


#endif /* ECMANAGEMENTDELAY_H_ */
