/*
 * ecmanagement.h
 *
 *  Created on: Mar 8, 2012
 *      Author: consciousness
 */

#ifndef ECMANAGEMENT_H_
#define ECMANAGEMENT_H_

#include <interface/cbmsimcore.h>
#include <tools/mfpoissonregen.h>

class ECManagement
{
public:
	ECManagement(int numT, int iti);
	virtual ~ECManagement();

	bool runStep();

	int getCurrentTrialN();
	int getCurrentTime();

	const bool* exportAPMF();
	const bool* exportAPGO();
	const bool* exportAPGR();
	const bool* exportAPSC();
	const bool* exportAPBC();
	const bool* exportAPPC();
	const bool* exportAPIO();
	const bool* exportAPNC();

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
	unsigned int getNumSC();
	unsigned int getNumBC();
	unsigned int getNumPC();
	unsigned int getNumNC();
	unsigned int getNumIO();


private:
	CBMSimCore *simulation;
	MFPoissonRegen *mf;

	float *mfFreq;
	int numMF;

	int numTrials;
	int interTrialI;

	int currentTrial;
	int currentTime;
};

#endif /* ECMANAGEMENT_H_ */