/*
 * main.cpp
 *
 *  Created on: Feb 16, 2011
 *      Author: consciousness
 */

#include "../includes/main.h"

using namespace std;

int main(int argc, char *argv[])
{
	CRandomSFMT0 randGen(time(NULL));

	cout<<argv[1]<<endl;
	if(!readin(argv[1]))
	{
		cerr<<"failed to open input file, exiting"<<endl;
		return 0;
	}

	for(int i=0; i<numMF; i++)
	{
		float tempSR;
		tempSR=randGen.Random()*4;
		if(pAMFSR[i]<0)
		{
			pAMFSR[i]=tempSR;
		}
		if(pBMFSR[i]<0)
		{
			pBMFSR[i]=tempSR;
		}
	}

	if(!writeout(argv[2]))
	{
		cerr<<"failed to write results to files, exiting"<<endl;
		return 0;
	}

	return 1;
}
