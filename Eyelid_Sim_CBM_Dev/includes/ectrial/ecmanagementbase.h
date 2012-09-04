/*
 * ecmanagement.h
 *
 *  Created on: Mar 8, 2012
 *      Author: consciousness
 */

#ifndef ECMANAGEMENT_H_
#define ECMANAGEMENT_H_

#include <iostream>
#include <time.h>

#include <CBMCoreInclude/interface/cbmsimcore.h>
#include <CBMCoreInclude/tools/mfpoissonregen.h>
#include <CBMCoreInclude/tools/randomc.h>
#include <CBMCoreInclude/tools/sfmt.h>

#include <CBMDataInclude/interfaces/ecrastertrial.h>

class ECManagementBase
{
public:
	ECManagementBase(int numT, int iti);
	virtual ~ECManagementBase();

	bool runStep();

	int getCurrentTrialN();
	int getCurrentTime();
	int getNumTrials();
	int getInterTrialI();

	const bool* exportAPMF();
	const bool* exportAPGO();
	const bool* exportAPGR();
	const bool* exportAPGL();
	const bool* exportAPSC();
	const bool* exportAPBC();
	const bool* exportAPPC();
	const bool* exportAPIO();
	const bool* exportAPNC();

	const float* exportVmGO();
	const float* exportVmSC();
	const float* exportVmBC();
	const float* exportVmPC();
	const float* exportVmNC();
	const float* exportVmIO();

	const unsigned int* exportAPBufMF();
	const unsigned int* exportAPBufGO();
	const unsigned int* exportAPBufGR();
	const unsigned int* exportAPBufSC();
	const unsigned int* exportAPBufBC();
	const unsigned int* exportAPBufPC();
	const unsigned int* exportAPBufIO();
	const unsigned int* exportAPBufNC();

	unsigned int getGRX();
	unsigned int getGRY();
	unsigned int getGOX();
	unsigned int getGOY();
	unsigned int getGLX();
	unsigned int getGLY();

	unsigned int getNumMF();
	unsigned int getNumGO();
	unsigned int getNumGR();
	unsigned int getNumGL();
	unsigned int getNumSC();
	unsigned int getNumBC();
	unsigned int getNumPC();
	unsigned int getNumNC();
	unsigned int getNumIO();


protected:

	virtual void calcMFActivity();
	virtual void calcSimActivity();

	CBMSimCore *simulation;
	MFPoissonRegen *mf;

	float *mfFreq;
	const bool *apMF;
	int numMF;

	int numTrials;
	int interTrialI;

	int currentTrial;
	int currentTime;

private:
	ECManagementBase();
};

#endif /* ECMANAGEMENT_H_ */
