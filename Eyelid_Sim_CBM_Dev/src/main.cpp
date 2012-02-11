/*
 * main.cpp
 *
 *  Created on: Dec 21, 2011
 *      Author: consciousness
 */

#include "../includes/main.h"

using namespace std;
int main(int argc, char **argv)
{
	CBMSimCore *simCore;
	MFPoissonRegen *mf;
	float *freqs;

	int t;

	simCore=new CBMSimCore(1);
	mf=new MFPoissonRegen(simCore->getNumMF());
	freqs=new float[simCore->getNumMF()];

	for(int i=0; i<simCore->getNumMF(); i++)
	{
		freqs[i]=5;
	}

	cerr<<"starting run"<<endl;

	for(int i=0; i<1000; i++)
	{
		t=time(0);
		cerr<<"iteration #"<<i<<": ";
		cerr.flush();
		for(int j=0; j<5000; j++)
		{
			const bool *mfAct;
			mfAct=mf->calcActivity(freqs);

			simCore->updateMFInput(mfAct);
			simCore->updateErrDrive(0, 0);
			simCore->calcActivity();
		}
		cerr<<time(0)-t<<" sec"<<endl;
	}

	delete simCore;
	delete mf;
	delete[] freqs;
}
