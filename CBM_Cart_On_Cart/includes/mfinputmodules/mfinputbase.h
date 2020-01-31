/*
 * mfinputbase.h
 *
 *  Created on: Apr 27, 2011
 *      Author: consciousness
 */

#ifndef MFINPUTBASE_H_
#define MFINPUTBASE_H_
#include "../common.h"

class BaseMFInput
{
public:
	BaseMFInput(unsigned int nmf, float ts, float tsus);

	BaseMFInput(ifstream &infile);

	virtual void exportState(ofstream &outfile);

	virtual ~BaseMFInput();

	void exportActDisp(vector<bool> &apRaster, int numCells);

	unsigned int getNumMF(){return numMF;};

	virtual void calcActivity(unsigned int tsN, unsigned int trial)=0;

	const bool *exportApMF(unsigned int &nmf){nmf=numMF; return (const bool *)apMF;};

//	virtual unsigned bool* exportAct(unsigned int startN, unsigned int endN, bool *actOut)=0;
protected:
	unsigned int numMF;
	bool *apMF;
	float timeStepSize; //time step size
	float tsUnitInS; //time step units in seconds, e.g., ms should be 0.001
private:
	BaseMFInput();
};



#endif /* MFINPUTBASE_H_ */
