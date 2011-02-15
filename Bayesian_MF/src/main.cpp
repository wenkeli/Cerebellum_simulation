/*
 * main.cpp
 *
 *  Created on: Feb 15, 2011
 *      Author: consciousness
 */

#include "../includes/main.h"

using namespace std;

int main(int argc, char **argv)
{
	CRandomSFMT0 randGen(time(NULL));

	if(!readInputs(argv[0]))
	{
		cout<<"failed to read input file"<<endl;
		return 0;
	}

	for(int i=0; i<NUMTRIALS; i++)
	{
		if(randGen.Random()<0.5)
		{
			calcMFActsPoisson(0, randGen);
		}
		else
		{
			calcMFActsPoisson(1, randGen);
		}

		bayesianCalcSV(i);
	}

	writeOutputs(argv[1]);

	return 1;
}
