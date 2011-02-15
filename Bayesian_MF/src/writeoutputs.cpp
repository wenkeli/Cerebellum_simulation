/*
 * writeoutputs.cpp
 *
 *  Created on: Feb 15, 2011
 *      Author: consciousness
 */

#include "../includes/writeoutputs.h"

using namespace std;

bool writeOutputs(char *outputFileName)
{
	ofstream outfile;

	outfile.open(outputFileName, ios::out);

	for(int i=0; i<NUMTRIALS; i++)
	{
		outfile<<sVs[i]<<endl;
	}

	return true;
}
