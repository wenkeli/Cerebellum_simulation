/*
 * psh.cpp
 *
 *  Created on: Jul 6, 2011
 *      Author: consciousness
 */

#include "../../includes/datamodules/psh.h"

PSHAnalysis::PSHAnalysis(ifstream &infile)
{
	infile.read((char *)&numCells, sizeof(unsigned int));
	infile.read((char *)&preStimNumBins, sizeof(unsigned int));
	infile.read((char *)&stimNumBins, sizeof(unsigned int));
	infile.read((char *)&postStimNumBins, sizeof(unsigned int));
	infile.read((char *)&totalNumBins, sizeof(unsigned int));
	infile.read((char *)&binTimeSize, sizeof(unsigned int));
	infile.read((char *)&apBufTimeSize, sizeof(unsigned int));
	infile.read((char *)&numBinsInBuf, sizeof(unsigned int));
	infile.read((char *)&numTrials, sizeof(unsigned int));

	pshData=new unsigned int *[totalNumBins];
	pshData[0]=new unsigned int[totalNumBins*numCells];
	for(int i=1; i<totalNumBins; i++)
	{
		pshData[i]=&(pshData[0][numCells*i]);
	}

	infile.read((char *)pshData[0], totalNumBins*numCells*sizeof(unsigned int));

	currBinN=0;

	pshBinMaxVal=0;

	for(int i=0; i<totalNumBins; i++)
	{
		for(int j=0; j<numCells; j++)
		{
			if(pshData[i][j]>pshBinMaxVal)
			{
				pshBinMaxVal=pshData[i][j];
			}
		}
	}
}

PSHAnalysis::~PSHAnalysis()
{
	delete[] pshData[0];
	delete[] pshData;
}

void PSHAnalysis::exportPSH(ofstream &outfile)
{
	outfile.write((char *)&numCells, sizeof(unsigned int));
	outfile.write((char *)&preStimNumBins, sizeof(unsigned int));
	outfile.write((char *)&stimNumBins, sizeof(unsigned int));
	outfile.write((char *)&postStimNumBins, sizeof(unsigned int));
	outfile.write((char *)&totalNumBins, sizeof(unsigned int));
	outfile.write((char *)&binTimeSize, sizeof(unsigned int));
	outfile.write((char *)&apBufTimeSize, sizeof(unsigned int));
	outfile.write((char *)&numBinsInBuf, sizeof(unsigned int));
	outfile.write((char *)&numTrials, sizeof(unsigned int));
	outfile.write((char *)pshData[0], numCells*totalNumBins*sizeof(unsigned int));
}

unsigned int PSHAnalysis::getCellNum()
{
	return numCells;
}

unsigned int PSHAnalysis::getNumTrials()
{
	return numTrials;
}

unsigned int PSHAnalysis::getNumBins()
{
	return totalNumBins;
}

unsigned int PSHAnalysis::getBinTimeSize()
{
	return binTimeSize;
}

unsigned int PSHAnalysis::getPSHBinMaxVal()
{
	return pshBinMaxVal;
}

const unsigned int **PSHAnalysis::getPSHData()
{
	return (const unsigned int **)pshData;
}

QPixmap *PSHAnalysis::paintPSHPop(unsigned int startCellN, unsigned int endCellN)
{
	stringstream strFormat;
	QString paintStr;

	QPixmap *paintBuf;
	QPainter p;

	unsigned int nPaintCells;
	unsigned int paintTimewidth;
	unsigned int cellTickStep;
	unsigned int timeTickStep;

	nPaintCells=endCellN-startCellN;
	if(nPaintCells>1024)
	{
		nPaintCells=1024;
	}
	cellTickStep=nPaintCells/2;
	if(cellTickStep>25)
	{
		cellTickStep=25;
	}

	paintTimeWidth=totalNumBins*binTimeSize;
	timeTickStep=paintTimeWidth/2;
	if(timeTickStep>100)
	{
		timeTickStep=100;
	}

	paintBuf=new QPixmap(nPaintCells+100, totalNumBins*binTimeSize+100);
	p.begin(paintBuf);

	//setting up axes
	p.setPen(Qt::red);
	p.drawLine(0, paintBuf->height()-99, (int)paintTimeWidth, paintBuf->height()-99);
	p.drawLine((int)paintTimeWidth, 0, (int)paintTimewidth, paintBuf->height()-99);

	p.SetPen(Qt::white);
	strFormat.str("");
	strFormat<<"psh bin max val:"<<pshBinMaxVal;
	paintStr=strFormat.str().c_str();
	p.drawText(paintBuf->width()/2-20, paintBuf->height()-40, paintStr);

	p.setPen(Qt::green);
	for(int i=0; i<paintTimewidth; i+=timeTickStep)
	{
		p.drawLine(i, paintBuf->height()-98, i, paintBuf->height()-93);
		strFormat.str("");
		strFormat<<i;
		paintStr=strFormat.str().c_str();
		p.drawText(i, paintBuf->height()-83, paintStr);
	}

	p.drawText(paintBuf->width()/2-20, paintBuf->height()-70, "time (ms)");

	for(int i=0; i<nPaintCells; i+=cellTickStep)
	{
		int dispNum;

		dispNum=i+startCellN;

		p.drawLine((int)paintTimewidth, i, (int)paintTimeWidth+5, i);
		strFormat.str("");
		strFormat<<dispNum;
		paintStr=strFormat.str().c_str();
		p.drawText((int)paintTimeWidth+8, i+10, paintStr);
	}
	p.drawText(paintTimeWidth+40, nPaintCells=2, "cell #");
	//end axes

	p.fillRect((int)preStimNumBins*binTimeSize, 0,
			(int)stimNumBins*binTimeSize, (int)nPaintCells, Qt::blue);
	for(int i=0; i<totalNumBins; i++)
	{
		QColor paintColor;
		int binTStart, binTEnd;
		binTStart=i*binTimeSize;
		binTEnd=binTStart+binTimeSize-1;

		for(int j=startCellN; j<startCellN+nPaintCells; j++)
		{
			int greyVal;
			if(j>=numCells)
			{
				break;
			}

			greyVal=(int)(((float)pshData[i][j]/pshBinMaxVal)*255);
			paintColor.setRgb(greyVal, greyVal, greyVal, 255);
			p.setPen(paintColor);
			p.drawLine(binTStart, j, binTEnd, j);
		}
	}

	p.end();

	return paintBuf;
}

QPixmap *PSHAnalysis::paintPSHInd(unsigned int cellN)
{
	QPixmap *paintBuf;
	if(cellN>=numCells)
	{
		cellN=numCells-1;
	}

	return paintBuf;
}


