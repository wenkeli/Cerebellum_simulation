/*
 * errorinputbase.h
 *
 *  Created on: Apr 27, 2011
 *      Author: consciousness
 */

#ifndef ERRORINPUTBASE_H_
#define ERRORINPUTBASE_H_

#include "../common.h"

class BaseErrorInput
{
public:
	BaseErrorInput(float maxE, float minE, float ts, float tsus);
	BaseErrorInput(ifstream &infile);
	virtual ~BaseErrorInput();
	virtual void calcActivity(unsigned int tsN, int trial)=0;
	float exportErr(){return currErrSig;};
	void setError(bool error){inError=error;};
        bool getError() {return inError;};

	virtual void exportState(ofstream &outfile);
protected:
	float maxErrSig;
	float minErrSig;

	float timeStepSize;
	float tsUnitInS;

	float currErrSig;

	bool inError;
private:
	BaseErrorInput();
};

#endif /* ERRORINPUTBASE_H_ */
