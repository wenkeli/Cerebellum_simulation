/*
 * pshtravclusterbayesian.h
 *
 *  Created on: Sep 13, 2011
 *      Author: consciousness
 */

#ifndef PSHTRAVCLUSTERBAYESIAN_H_
#define PSHTRAVCLUSTERBAYESIAN_H_

#include "pshtravclusterbase.h"

class BayesianPSHTravCluster : public BasePSHTravCluster
{
public:
	BayesianPSHTravCluster(PSHData *data);

protected:
	bool isSimilar(float *psh1, float *psh2);

private:
	BayesianPSHTravCluster();
};

#endif /* PSHTRAVCLUSTERBAYESIAN_H_ */
