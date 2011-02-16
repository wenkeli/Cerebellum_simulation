/*
 * writeout.cpp
 *
 *  Created on: Feb 16, 2011
 *      Author: consciousness
 */

#include "../includes/writeout.h"

using namespace std;

bool writeout(char *outFNsRoot)
{
	stringstream binFN;
	stringstream txtFN;

	ofstream binout;
	ofstream txtout;

	binFN.str("");
	txtFN.str("");
	binFN<<outFNsRoot<<".MFCompletedef.bin";
	txtFN<<outFNsRoot<<".MFCompletedef.txt";

	binout.open(binFN.str().c_str(), ios::binary);
	txtout.open(txtFN.str().c_str(), ios::out);

	if(!binout.good() || !binout.is_open())
	{
		cerr<<"failed to create/open file for binary output: "<<binFN.str()<<endl;
		return false;
	}
	if(!txtout.good() || !txtout.is_open())
	{
		cerr<<"failed to create/open file for txt output: "<<txtFN.str()<<endl;
		return false;
	}

	for(int i=0; i<numMF; i++)
	{
		float tempout;
		tempout=pAMFSR[i];
		binout.write((char *)&tempout, sizeof(float));
		txtout<<pAMFSR[i]<<" "<<pBMFSR[i]<<endl;
	}
	for(int i=0; i<numMF; i++)
	{
		float tempout;
		tempout=pBMFSR[i];
		binout.write((char *)&tempout, sizeof(float));
	}

	binout.close();
	txtout.close();

	return true;
}
