/*
 * pshtravclustereucdist.h
 *
 *  Created on: Nov 29, 2011
 *      Author: consciousness
 */

#ifndef PSHTRAVCLUSTEREUCDIST_H_
#define PSHTRAVCLUSTEREUCDIST_H_

#include "pshtravclusterbase.h"
#include <stdlib.h>
#ifdef INTELCC
#include <mathimf.h>
#else
#include <math.h>
#endif

class EucDistPSHTravCluster : public BasePSHTravCluster
{
public:
	EucDistPSHTravCluster(PSHData *data, float thresh, unsigned int distNumBins);
	~EucDistPSHTravCluster();
protected:
	bool isDifferent(float *psh1, float *psh2);

private:
	EucDistPSHTravCluster();

	void generateDist();

	float calcEuclideanDist(float *psh1, float *psh2);

	float *eculideanPDF;
	float *eculideanCDF;
	unsigned int distNumBins;
	float threshP;
	float threshVal;
};

#endif /* PSHTRAVCLUSTEREUCDIST_H_ */
