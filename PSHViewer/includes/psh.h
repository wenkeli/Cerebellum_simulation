/*
 * psh.h
 *
 *  Created on: Jul 1, 2011
 *      Author: consciousness
 */

#ifndef PSH_H_
#define PSH_H_

#include <iostream>
#include <fstream>

using namespace std;

template<typename Type>
class PSHData
{
public:
	PSHData(ifstream &infile);

private:
	PSHData();

	unsigned int numCells;
	unsigned int numBins;
	unsigned int binTimeSize;
	unsigned int numTrials;

	unsigned int maxBinVal;

	Type **pshData;
};

#endif /* PSH_H_ */
