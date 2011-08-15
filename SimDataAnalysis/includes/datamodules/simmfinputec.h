/*
 * simmfinputec.h
 *
 *  Created on: Aug 15, 2011
 *      Author: consciousness
 */

#ifndef SIMMFINPUTEC_H_
#define SIMMFINPUTEC_H_

#include <fstream>
#include <iostream>

using namespace std;

class SimMFInputEC
{
public:
	SimMFInputEC(ifstream &infile);

	~SimMFInputEC();
private:
	SimMFInputEC();

	static const char mfTypeDefNonAct=0;
	static const char mfTypeDefContext=1;
	static const char mfTypeDefCSTonic=2;
	static const char mfTypedefCSPhasic=3;

	unsigned int numMF;
	float timeStepSize; //time step size
	float tsUnitInS;

	float threshDecay;

	float contextMFProportion;
	float csTonicMFProportion;
	float csPhasicMFProportion;
	float csStart;
	float csEnd;
	float csPhasicMFEnd;

	char *mfType;
	float *bgFreq;
	float *incFreq;
	float *incStart;
	float *incEnd;

	float *threshold;
};

#endif /* SIMMFINPUTEC_H_ */
