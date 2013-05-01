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

#include <CBMStateInclude/interfaces/cbmstatex2grgodecouple.h>
#include <CBMStateInclude/interfaces/iconnectivityparams.h>

#include <CBMCoreInclude/interface/cbmsimx2grgodecouple.h>
#include <CBMCoreInclude/interface/innetinterface.h>
#include <CBMCoreInclude/interface/mzoneinterface.h>

#include <CBMDataInclude/interfaces/ectrialsdata.h>

class ECManagementBase
{
public:
	ECManagementBase(std::string conParamFile, std::string actParamFile,
			int numT, int iti, int randSeed);
	virtual ~ECManagementBase();

	bool runStep();

	int getCurrentTrialN();
	int getCurrentTime();
	int getNumTrials();
	int getInterTrialI();

	IConnectivityParams* getConParams();
	InNetInterface* getInputNet(unsigned int stateN);
	MZoneInterface* getMZone(unsigned int zoneN);

	const ct_uint8_t* exportAPMF();

protected:
	virtual void calcMFActivity()=0;
	virtual void calcSimActivity();

	CBMStateX2GRGODecouple *simState;

	CBMSimX2GRGODecouple *simulation;

	const ct_uint8_t *apMF;
	int numMF;

	int numTrials;
	int interTrialI;

	int currentTrial;
	int currentTime;

private:
	ECManagementBase();
};

#endif /* ECMANAGEMENT_H_ */
