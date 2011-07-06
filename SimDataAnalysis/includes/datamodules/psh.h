/*
 * psh.h
 *
 *  Created on: Jul 6, 2011
 *      Author: consciousness
 */

#ifndef PSH_H_
#define PSH_H_

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>
#include <ctime>
#include <QtGui/QPixmap>
#include <QtGui/QPainter>

using namespace std;

class PSHAnalysis
{
public:
	PSHAnalysis(ifstream &infile);
	virtual ~PSHAnalysis();

	virtual void exportPSH(ofstream &outfile);

	unsigned int getCellNum();
	unsigned int getNumTrials();
	unsigned int getNumBins();
	unsigned int getBinTimeSize();
	unsigned int getPSHBinMaxVal();

	const unsigned int **getPSHData();

	QPixmap *paintPSHPop(unsigned int startCellN, unsigned int endCellN);
	QPixmap *paintPSHInd(unsigned int cellN);
protected:

	unsigned int numCells;
	unsigned int numBins;
	unsigned int binTimeSize;
	unsigned int apBufTimeSize;
	unsigned int numBinsInBuf;
	unsigned int numTrials;
	unsigned int currBinN;

	unsigned int pshBinMaxVal;
	unsigned int **pshData;

private:
	PSHAnalysis();
};

#endif /* PSH_H_ */
