/*
 * innetallgrgo.h
 *
 *  Created on: Aug 9, 2011
 *      Author: consciousness
 */

#ifndef INNETALLGRGO_H_
#define INNETALLGRGO_H_

#include "innet.h"

class InNetAllGRGO : public InNet
{
public:
	InNetAllGRGO(const bool *actInMF);
	InNetAllGRGO(ifstream &infile, const bool *actInMF);

	void runSumGRGOOutCUDA(cudaStream_t &st);
private:
	InNetAllGRGO();
};

#endif /* INNETALLGRGO_H_ */
