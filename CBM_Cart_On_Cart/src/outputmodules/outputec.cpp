/*
 * outputec.cpp
 *
 *  Created on: May 25, 2011
 *      Author: consciousness
 */

#include "../../includes/outputmodules/outputec.h"
#include <iostream>
using namespace std;
ECOutput::ECOutput(unsigned int numNC, float ts, float tsus)
	:BaseOutput(numNC, ts, tsus)
{
	gForceDecay=exp(-(ts/gForceDecayT));
	gForce=0;
	gForceLeak=ts*tsus*gForceLeakConst;
	vForce=eForceLeak;

	velocityDecay=exp(-(ts/velocityDecayT));
	velocity=minVelocity;

	positionDecay=exp(-(ts/positionDecayT));
	position=minPosition;
}

ECOutput::ECOutput(ifstream &infile):BaseOutput(infile)
{
	infile.read((char *)&gForceDecay, sizeof(float));
	infile.read((char *)&gForce, sizeof(float));
	infile.read((char *)&gForceLeak, sizeof(float));
	infile.read((char *)&vForce, sizeof(float));
	infile.read((char *)&velocityDecay, sizeof(float));
	infile.read((char *)&velocity, sizeof(float));
	infile.read((char *)&positionDecay, sizeof(float));
	infile.read((char *)&position, sizeof(float));

	cout<<"output mod read:"<<numNCIn<<" "<<timeStepSize<<" "<<tsUnitInS
			<<" "<<gForceDecay<<" "<<gForce<<" "<<gForceLeak<<" "<<" "<<vForce
			<<" "<<velocityDecay<<" "<<velocity<<" "<<positionDecay<<" "<<position<<endl;
}

void ECOutput::exportState(ofstream &outfile)
{
	BaseOutput::exportState(outfile);
	outfile.write((char *)&gForceDecay, sizeof(float));
	outfile.write((char *)&gForce, sizeof(float));
	outfile.write((char *)&gForceLeak, sizeof(float));
	outfile.write((char *)&vForce, sizeof(float));
	outfile.write((char *)&velocityDecay, sizeof(float));
	outfile.write((char *)&velocity, sizeof(float));
	outfile.write((char *)&positionDecay, sizeof(float));
	outfile.write((char *)&position, sizeof(float));
}

void ECOutput::calcOutput()
{
	float inputSum=0;

	for(int i=0; i<numNCIn; i++)
	{
		inputSum+=apNCIn[i];
	}

	inputSum=inputSum/numNCIn;
	inputSum=inputSum*inputSum;

	gForce=gForce*gForceDecay;
	gForce=gForce+inputSum*gForceIncScale;


	vForce=vForce+gForce*(eForce-vForce)+gForceLeak*(eForceLeak-vForce);

	velocity=velocity*velocityDecay;
	velocity=velocity+vForce;
	velocity=(velocity>maxVelocity)*maxVelocity+(!(velocity>maxVelocity))*velocity;

	position=position*positionDecay;
	position=position+velocity;
	position=(position>maxPosition)*maxPosition+(!(position>maxPosition))*position;

	output=(position-minPosition)/(maxPosition-minPosition);
//	cout<<inputSum<<" "<<gForce<<" "<<vForce<<" "<<velocity<<" "<<position<<endl;
}
