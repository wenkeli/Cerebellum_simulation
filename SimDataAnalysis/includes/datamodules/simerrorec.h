/*
 * simerrorec.h
 *
 *  Created on: Aug 15, 2011
 *      Author: consciousness
 */

#ifndef SIMERROREC_H_
#define SIMERROREC_H_

#include <fstream>

using namespace std;

class SimErrorEC
{
public:
	SimErrorEC(ifstream &infile);

private:
	SimErrorEC();

	float maxErrSig;
	float minErrSig;

	float timeStepSize;
	float tsUnitInS;

	float errOnsetT;
	float tsWindowInS;
	float errOnsetST;
	float errOnsetET;
};

#endif /* SIMERROREC_H_ */
