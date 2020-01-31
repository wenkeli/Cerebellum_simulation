/*
 * outputcp.cpp
 *
 *  Created on: May 27, 2011
 *      Author: mhauskn
 */

#include "../../includes/outputmodules/outputcp.h"

CPOutput::CPOutput(unsigned int numNC, float ts, float tsus) :
	BaseOutput(numNC, ts, tsus)
{
	gForceInc=0.1;
	gForceIncDecay=exp(-(timeStepSize/5));
	gForceDecay=exp(-(timeStepSize/gForceDecayT));
	gForce=0;
	gForceLeak=timeStepSize*tsUnitInS*gForceLeakConst;
	vForce=eForceLeak;
}

CPOutput::CPOutput(ifstream &infile):BaseOutput(infile)
{
	infile.read((char *)&gForceDecay, sizeof(float));
	infile.read((char *)&gForce, sizeof(float));
	infile.read((char *)&gForceLeak, sizeof(float));
	infile.read((char *)&vForce, sizeof(float));

	gForceInc=0.1;
	gForceIncDecay=exp(-(timeStepSize/5));
	gForceDecay=exp(-(timeStepSize/gForceDecayT));
	gForce=0;
	gForceLeak=timeStepSize*tsUnitInS*gForceLeakConst;
	vForce=eForceLeak;
}

void CPOutput::exportState(ofstream &outfile)
{
	BaseOutput::exportState(outfile);
	outfile.write((char *)&gForceDecay, sizeof(float));
	outfile.write((char *)&gForce, sizeof(float));
	outfile.write((char *)&gForceLeak, sizeof(float));
	outfile.write((char *)&vForce, sizeof(float));
}

void CPOutput::calcOutput()
{
	float inputSum=0;

	for(int i=0; i<numNCIn; i++)
	{
		inputSum+=apNCIn[i];
	}

	inputSum=inputSum/numNCIn;
	//inputSum=inputSum*inputSum;

        //inputSum *= 2;

	// gForceInc=gForceInc-(gForceInc-0.1)*gForceIncDecay;
	// gForceInc=gForceInc+inputSum*0.03;
	// gForce=gForce*gForceDecay;
	// gForce=gForce+inputSum*gForceInc;
        gForce += inputSum * 0.05;
        gForce *= .99;
        output = gForce;


	// vForce=vForce+gForce*(eForce-vForce)+gForceLeak*(eForceLeak-vForce);

	// output=(vForce-eForceLeak)/(eForce-eForceLeak);
}
