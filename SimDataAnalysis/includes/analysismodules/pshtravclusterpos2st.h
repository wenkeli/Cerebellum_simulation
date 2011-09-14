/*
 * pshtravclusterpos2st.h
 *
 *  Created on: Sep 14, 2011
 *      Author: consciousness
 */

#ifndef PSHTRAVCLUSTERPOS2ST_H_
#define PSHTRAVCLUSTERPOS2ST_H_

#ifdef INTELCC
#include <mathimf.h>
#else
#include <math.h>
#endif


#include "pshtravclusterbase.h"

class Pos2STPSHTravCluster : public BasePSHTravCluster
{
public:
	Pos2STPSHTravCluster(PSHData *data);
protected:
	bool isDifferent(float *psh1, float *psh2);

private:
	Pos2STPSHTravCluster();

	double poisson2SampleT(float count1, float count2);

	static const double sqrt2;
};

#endif /* PSHTRAVCLUSTERPOS2ST_H_ */
