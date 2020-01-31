/*
 * innetnogolgi.h
 *
 *  Created on: Aug 2, 2011
 *      Author: consciousness
 */

#ifndef INNETNOGOLGI_H_
#define INNETNOGOLGI_H_

#include "innet.h"

class InNetNoGO : public InNet
{
public:
	InNetNoGO(const bool *actInMF);
	InNetNoGO(ifstream &infile, const bool *actInMF);
private:
	InNetNoGO();
};

#endif /* INNETNOGOLGI_H_ */
