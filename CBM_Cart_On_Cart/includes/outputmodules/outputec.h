/*
 * outputec.h
 *
 *  Created on: May 25, 2011
 *      Author: consciousness
 */

#ifndef OUTPUTEC_H_
#define OUTPUTEC_H_

#include "outputbase.h"
#include "../common.h"

class ECOutput : public BaseOutput
{
public:
	ECOutput(unsigned int numNC, float ts, float tsus);
	ECOutput(ifstream &infile);

	void exportState(ofstream &outfile);

	void calcOutput();
private:
	ECOutput();

	static const float gForceDecayT=8;//0.1;
	static const float gForceIncScale=0.002;
	static const float eForce=10;
	static const float eForceLeak=0;
	static const float gForceLeakConst=500;

	static const float minVelocity=0;
	static const float maxVelocity=10;
	static const float velocityDecayT=4;//2.5;//0.1;

	static const float minPosition=0;
	static const float maxPosition=100;
	static const float positionDecayT=4;//2.5;//0.1;

	float gForceDecay;
	float gForce;
	float gForceLeak;
	float vForce;

	float velocityDecay;
	float velocity;

	float positionDecay;
	float position;
};

#endif /* OUTPUTEC_H_ */
