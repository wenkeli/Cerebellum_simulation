/*
 * innetsparegrgo.h
 *
 *  Created on: Aug 9, 2011
 *      Author: consciousness
 */

#ifndef INNETSPAREGRGO_H_
#define INNETSPAREGRGO_H_

#include "innet.h"

class InNetSparseGRGO : public InNet
{
public:
	InNetSparseGRGO(const bool *actInMF);
	InNetSparseGRGO(ifstream &infile, const bool *actInMF);
private:
	InNetSparseGRGO();
	void reinitializeVars();
	void reinitializeCuda();
};

#endif /* INNETSPAREGRGO_H_ */
