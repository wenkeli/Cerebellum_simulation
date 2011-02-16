/*
 * main.cpp
 *
 *  Created on: Feb 15, 2011
 *      Author: consciousness
 */

#include "../includes/main.h"

using namespace std;

int main(int argc, char *argv[])
{
	CRandomSFMT0 randGen(time(NULL));
	int runtime;
	if(!readInputs(argv[1]))
	{
		cout<<"failed to read input file"<<endl;
		return 0;
	}

	runtime=time(NULL);
	for(int i=0; i<NUMTRIALS; i++)
	{
		if(randGen.Random()<0.5)
		{
			calcMFActsRegenPoisson(0, randGen);
		}
		else
		{
			calcMFActsRegenPoisson(1, randGen);
		}

		bayesianCalcSV(i);

		if(i%10000==0)
		{
			cout<<i<<" "<<time(NULL)-runtime<<endl;
			runtime=time(NULL);
		}

	}

	writeOutputs(argv[2]);

	return 1;
}
