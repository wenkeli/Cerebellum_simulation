/*
 * pshtravclustereucdist.h
 *
 *  Created on: Nov 29, 2011
 *      Author: consciousness
 */

#ifndef PSHTRAVCLUSTEREUCDIST_H_
#define PSHTRAVCLUSTEREUCDIST_H_

#include "pshtravclusterbase.h"
#include <ctime>
#ifdef INTELCC
#include <mathimf.h>
#else
#include <math.h>
#endif

#include <vector>
#include <algorithm>
#include <functional>

#include <iostream>
#include <iomanip>
#include "../randomc.h"
#include "../sfmt.h"

using namespace std;

class EucDistPSHTravCluster : public BasePSHTravCluster
{
public:
	EucDistPSHTravCluster(PSHData *data, float thresh);
protected:
	bool isDifferent(float *psh1, float *psh2);

private:
	EucDistPSHTravCluster();

	void generateDist();

	double calcEuclideanDist(float *psh1, float *psh2);

	float threshP;
	double threshVal;

	vector<double> distances;
};

#endif /* PSHTRAVCLUSTEREUCDIST_H_ */
