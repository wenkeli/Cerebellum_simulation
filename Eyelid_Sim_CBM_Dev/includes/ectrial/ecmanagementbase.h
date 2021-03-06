/*
 * ecmanagement.h
 *
 *  Created on: Mar 8, 2012
 *      Author: consciousness
 */

#ifndef ECMANAGEMENT_H_
#define ECMANAGEMENT_H_

#include <iostream>
#include <fstream>
#include <string>
#include <time.h>

#include <CBMStateInclude/interfaces/cbmstate.h>
#include <CBMStateInclude/interfaces/iconnectivityparams.h>

#include <CBMCoreInclude/interface/cbmsimcore.h>
#include <CBMCoreInclude/interface/innetinterface.h>
#include <CBMCoreInclude/interface/mzoneinterface.h>

#include <CBMDataInclude/interfaces/ectrialsdata.h>

class ECManagementBase
{
public:
	ECManagementBase(std::string conParamFile, std::string actParamFile,
			int numT, int iti, int randSeed,
			int gpuIndStart=-1, int numGPUP2=-1);
	ECManagementBase(std::string stateDataFile, int numT, int iti, int randSeed,
			int gpuIndStart=-1, int numGPUP2=-1);

	virtual ~ECManagementBase();

	bool runStep();

	int getCurrentTrialN();
	int getCurrentTime();
	int getNumTrials();
	int getInterTrialI();

	IConnectivityParams* getConParams();
	InNetInterface* getInputNet();
	MZoneInterface* getMZone();

	const ct_uint8_t* exportAPMF();

protected:
	virtual void calcMFActivity()=0;
	virtual void calcSimActivity();

	CBMState *simState;

	CBMSimCore *simulation;

	const ct_uint8_t *apMF;
	int numMF;

	int numTrials;
	int interTrialI;

	int currentTrial;
	int currentTime;

private:
	ECManagementBase();

	void initialize(int randSeed, int numT, int iti, int gpuIndStart, int numGPUP2);
};

#endif /* ECMANAGEMENT_H_ */
