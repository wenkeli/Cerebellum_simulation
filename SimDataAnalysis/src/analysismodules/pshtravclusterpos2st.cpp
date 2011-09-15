/*
 * pshtravclusterpos2st.cpp
 *
 *  Created on: Sep 14, 2011
 *      Author: consciousness
 */

#include "../../includes/analysismodules/pshtravclusterpos2st.h"

const double Pos2STPSHTravCluster::sqrt2=sqrt(2);

Pos2STPSHTravCluster::Pos2STPSHTravCluster(PSHData *data)
	:BasePSHTravCluster(data)
{

}

bool Pos2STPSHTravCluster::isDifferent(float *psh1, float *psh2)
{
	for(int i=0; i<numBins; i++)
	{
		double pval;

		pval=poisson2SampleT(psh1[i], psh2[i]);

		if(pval<0.0001)
		{
			return true;
		}
	}

	return false;
}


double Pos2STPSHTravCluster::poisson2SampleT(float count1, float count2)
{
	double lambda;
	double zScore;

	double pVal;
	double pValInv;

	double countDiff;

	double count1sqr;
	double count2sqr;

	lambda=(count1+count2)/(2);

	countDiff=count1-count2;

	count1sqr=(count1-lambda)*(count1-lambda)/lambda;
	count2sqr=(count2-lambda)*(count2-lambda)/lambda;

	zScore=((countDiff>0)-(countDiff<0))*sqrt(count1sqr+count2sqr);

	pVal=(1+erf(zScore/sqrt2))/2;
	pValInv=1-pVal;

	if(pVal<pValInv)
	{
		return pVal*2;
	}
	return pValInv*2;
}
