/*
 * errorinputec.h
 *
 *  Created on: May 25, 2011
 *      Author: consciousness
 */

#ifndef ERRORINPUTEC_H_
#define ERRORINPUTEC_H_

#include "../common.h"
#include "errorinputbase.h"

class ECErrorInput : public BaseErrorInput
{
public:
 	ECErrorInput(float maxE, float minE, float ts, float tsus, unsigned int errOnsetTSN);
 	ECErrorInput(ifstream &infile);

 	virtual void calcActivity(unsigned int tsN, int trial);

 	virtual void exportState(ofstream &outfile);
private:
	ECErrorInput();

	float errOnsetT;
	float tsWindowInS;
	float errOnsetST;
	float errOnsetET;
};

#endif /* ERRORINPUTEC_H_ */
