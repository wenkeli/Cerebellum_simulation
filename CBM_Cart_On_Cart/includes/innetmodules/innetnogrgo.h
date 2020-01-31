/*
 * innetnogrgo.h
 *
 *  Created on: Aug 3, 2011
 *      Author: consciousness
 */

#ifndef INNETNOGRGO_H_
#define INNETNOGRGO_H_

#include "innet.h"

class InNetNoGRGO : public InNet
{
public:
	InNetNoGRGO(const bool *actInMF);
	InNetNoGRGO(ifstream &infile, const bool *actInMF);

private:
	InNetNoGRGO();
};
#endif /* INNETNOGRGO_H_ */
