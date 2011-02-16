/*
 * readin.cpp
 *
 *  Created on: Feb 16, 2011
 *      Author: consciousness
 */

#include "../includes/readin.h"

using namespace std;

bool readin(char *infileName)
{
	string dummy;
	ifstream infile;

	int numMFDefined;
	int mfIndex;
	float definedSR;

	infile.open(infileName, ios::in);

	if(!infile.good() || !infile.is_open())
	{
		cout<<"error opening file "<<infileName<<endl;
		return false;
	}

	infile>>dummy;
	cout<<dummy<<endl;
	infile>>numMF;
	cout<<numMF<<endl;

	pAMFSR.clear();
	pBMFSR.clear();
	pAMFSR.resize(numMF, -1);
	pBMFSR.resize(numMF, -1);

	infile>>dummy;
	cout<<dummy<<endl;
	infile>>numMFDefined;
	cout<<numMFDefined<<endl;
	for(int i=0; i<numMFDefined; i++)
	{
		infile>>mfIndex;
		infile>>definedSR;

		pAMFSR[mfIndex]=definedSR;
	}

	infile>>dummy;
	cout<<dummy<<endl;
	infile>>numMFDefined;
	cout<<numMFDefined<<endl;
	for(int i=0; i<numMFDefined; i++)
	{
		infile>>mfIndex;
		infile>>definedSR;
		pBMFSR[mfIndex]=definedSR;

		cout<<mfIndex<<" "<<definedSR<<endl;
	}

	return true;
}
