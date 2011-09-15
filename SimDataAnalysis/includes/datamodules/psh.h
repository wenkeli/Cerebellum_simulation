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
#include <QtCore/QString>

using namespace std;

class PSHData
{
public:
	PSHData(ifstream &infile);
	virtual ~PSHData();

	virtual void exportPSH(ofstream &outfile);

	unsigned int getCellNum();
	unsigned int getNumTrials();
	unsigned int getPreStimNumBins();
	unsigned int getStimNumBins();
	unsigned int getPostStimNumBins();
	unsigned int getTotalNumBins();
	unsigned int getBinTimeSize();
	unsigned int getPSHBinMaxVal();

	const unsigned int **getData();

	QPixmap *paintPSHPop(unsigned int startCellN, unsigned int endCellN);
	QPixmap *paintPSHInd(unsigned int cellN);
	QPixmap *paintPSH(float *psh);

protected:

	unsigned int numCells;
	unsigned int preStimNumBins;
	unsigned int stimNumBins;
	unsigned int postStimNumBins;
	unsigned int totalNumBins;
	unsigned int binTimeSize;
	unsigned int apBufTimeSize;
	unsigned int numBinsInBuf;
	unsigned int numTrials;
	unsigned int currBinN;

	unsigned int pshBinMaxVal;
	unsigned int **data;

//	unsigned int **dataTrans

private:
	PSHData();
};

#endif /* PSH_H_ */
