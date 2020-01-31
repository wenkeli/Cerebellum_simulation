/*
 * innetnomfgo.h
 *
 *  Created on: Aug 3, 2011
 *      Author: admin
 */

#ifndef INNETNOMFGO_H_
#define INNETNOMFGO_H_

#include "innet.h"

class InNetNoMFGO : public InNet
{
public:
	InNetNoMFGO(const bool *actInMF);
	InNetNoMFGO(ifstream &infile, const bool *actInMF);

private:
	InNetNoMFGO();
};

#endif /* INNETNOMFGO_H_ */
