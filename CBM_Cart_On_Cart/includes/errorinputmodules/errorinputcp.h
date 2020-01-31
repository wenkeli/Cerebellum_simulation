/*
 * errorinputcp.h
 *
 *  Created on: May 27, 2011
 *      Author: mhauskn
 */

#ifndef ERRORINPUTCP_H_
#define ERRORINPUTCP_H_

#include "../common.h"
#include "errorinputbase.h"
#include "../externalmodules/cartpole.h"

class CPErrorInput: public BaseErrorInput
{
public:
	CPErrorInput(float maxE, float minE, float ts, float tsus);
	CPErrorInput(ifstream &infile);

	void calcActivity(unsigned int tsN, int trial);

	void exportState(ofstream &outfile);

private:
	CPErrorInput();
};

#endif /* ERRORINPUTCP_H_ */
