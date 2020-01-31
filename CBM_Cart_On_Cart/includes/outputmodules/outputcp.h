/*
 * outputcp.h
 *
 *  Created on: May 27, 2011
 *      Author: mhauskn
 */

#ifndef OUTPUTCP_H_
#define OUTPUTCP_H_

#include "outputbase.h"
#include "../common.h"

class CPOutput: public BaseOutput
{
public:
	CPOutput(unsigned int numNC, float ts, float tsus);

	CPOutput(ifstream &infile);

	void exportState(ofstream &outfile);

	void calcOutput();

private:

	static const float gForceDecayT=60;//40;//15;//0.1;
//	float gForceIncScale=0.1;//0.1;//0.03;
	static const float eForce=100;
	static const float eForceLeak=0;
	static const float gForceLeakConst=250;//500;

	float gForceInc;
	float gForceIncDecay;
	float gForceDecay;
	float gForce;
	float gForceLeak;
	float vForce;

	CPOutput();
};

#endif /* OUTPUTCP_H_ */
