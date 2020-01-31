/*
 * mfinputec.h
 *
 *  Created on: Apr 29, 2011
 *      Author: consciousness
 */

#ifndef MFINPUTEC_H_
#define MFINPUTEC_H_

#include <ctime>
#include <string.h>

#include "../common.h"
#include "../randomc.h"
#include "../sfmt.h"

#include "mfinputbase.h"

//eyelid conditionig mossy fiber input
class ECMFInput : public BaseMFInput
{
public:
	ECMFInput(unsigned int nmf, unsigned int csSTSN, unsigned int csETSN, unsigned int csPhaETSN,
			float csTonicP, float csPhasicP, float contextP, float ts, float tsus);
	ECMFInput(ifstream &infile);

	void exportState(ofstream &outfile);

	~ECMFInput();

	void calcActivity(unsigned int tsN, unsigned int trial);

//	void exportAct(unsigned int startN, unsigned int endN, bool *actOut);
protected:

private:
	ECMFInput();
	void assignMF(char assignType, unsigned int assignNum);

	static const float threshDecayT=4;

	static const char mfTypeDefNonAct=0;
	static const char mfTypeDefContext=1;
	static const char mfTypeDefCSTonic=2;
	static const char mfTypedefCSPhasic=3;

	static const float bgFreqMin=1;
	static const float bgFreqMax=10;

	static const float bgContextFreqMin=30;
	static const float bgContextFreqMax=60;

	static const float bgCSFreqMin=1;
	static const float bgCSFreqMax=5;

	static const float csTonicIncFreq=40;
	static const float csPhasicIncFreq=120;

	float threshDecay;

	float contextMFProportion;
	float csTonicMFProportion;
	float csPhasicMFProportion;
	float csStart;
	float csEnd;
	float csPhasicMFEnd;

	char *mfType;
	float *bgFreq;
	float *incFreq;
	float *incStart;
	float *incEnd;

	float *threshold;
};


#endif /* MFINPUTEC_H_ */
