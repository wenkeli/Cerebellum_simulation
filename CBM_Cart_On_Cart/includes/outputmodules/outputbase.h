/*
 * outputbase.h
 *
 *  Created on: May 25, 2011
 *      Author: consciousness
 */

#ifndef OUTPUTBASE_H_
#define OUTPUTBASE_H_

#include "../common.h"

class BaseOutput
{
public:
	BaseOutput(unsigned int numNC, float ts, float tsus);

	BaseOutput(ifstream &infile);

	virtual ~BaseOutput();

	virtual void exportState(ofstream &outfile);

	void setApNCIn(const bool *apIn){apNCIn=apIn;};
	virtual void calcOutput()=0;
	float exportOutput(){return output;}; //output needs to be betwen 0 and 1
protected:
	unsigned int numNCIn;
	float timeStepSize;
	float tsUnitInS;


	const bool *apNCIn;
	float output;

private:
	BaseOutput();
};

#endif /* OUTPUTBASE_H_ */
