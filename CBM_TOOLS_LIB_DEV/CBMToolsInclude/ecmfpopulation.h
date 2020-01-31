/*
 * ecmfpopulation.h
 *
 *  Created on: Jul 11, 2014
 *      Author: consciousness
 */

#ifndef ECMFPOPULATION_H_
#define ECMFPOPULATION_H_

#include <vector>
#include <fstream>

#include <CXXToolsInclude/stdDefinitions/pstdint.h>
#include <CXXToolsInclude/randGenerators/randomc.h>
#include <CXXToolsInclude/randGenerators/sfmt.h>
//#include <CXXToolsInclude/fileIO/rawbytesrw.h>

#include <CBMDataInclude/peristimhist/peristimhistfloat.h>
#include <CBMDataInclude/interfaces/ectrialsdata.h>

class ECMFPopulation
{
public:
	ECMFPopulation(ECTrialsData *data);
	ECMFPopulation(int numMF, int randSeed, float fracCSTMF, float fracCSPMF, float fracCtxtMF,
			float bgFreqMin, float csBGFreqMin, float ctxtFreqMin, float csTFreqMin, float csPFreqMin,
			float bgFreqMax, float csBGFreqMax, float ctxtFreqMax, float csTFreqMax, float csPFreqMax);
	ECMFPopulation(std::fstream &infile);

	~ECMFPopulation();

	void writeToFile(std::fstream &outfile);

	float *getMFBG();
	float *getMFInCSTonic();
	float *getMFFreqInCSPhasic();

private:
	ct_uint32_t numMF;
	float *mfFreqBG;
	float *mfFreqInCSTonic;
	float *mfFreqInCSPhasic;
};



#endif /* ECMFPOPULATION_H_ */
