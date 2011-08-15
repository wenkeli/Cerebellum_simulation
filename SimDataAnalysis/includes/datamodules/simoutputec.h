/*
 * simoutputec.h
 *
 *  Created on: Aug 15, 2011
 *      Author: consciousness
 */

#ifndef SIMOUTPUTEC_H_
#define SIMOUTPUTEC_H_

#include <fstream>
#include <iostream>

using namespace std;

class SimOutputEC
{
public:
	SimOutputEC(ifstream &infile);

private:
	SimOutputEC();
	unsigned int numNCIn;
	float timeStepSize;
	float tsUnitInS;

	float output;

	float gForceDecay;
	float gForce;
	float gForceLeak;
	float vForce;

	float velocityDecay;
	float velocity;

	float positionDecay;
	float position;
};

#endif /* SIMOUTPUTEC_H_ */
