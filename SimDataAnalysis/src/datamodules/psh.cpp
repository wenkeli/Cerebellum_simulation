/*
 * psh.cpp
 *
 *  Created on: Jul 6, 2011
 *      Author: consciousness
 */

#include "../../includes/datamodules/psh.h"

PSHData::PSHData(ifstream &infile)
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

	stimNumBins=750/binTimeSize;
	postStimNumBins=totalNumBins-preStimNumBins-stimNumBins;

	data=new unsigned int *[totalNumBins];
	data[0]=new unsigned int[totalNumBins*numCells];
	for(int i=1; i<totalNumBins; i++)
	{
		data[i]=&(data[0][numCells*i]);
	}

	infile.read((char *)data[0], totalNumBins*numCells*sizeof(unsigned int));

	currBinN=0;

	pshBinMaxVal=0;

	for(int i=0; i<totalNumBins; i++)
	{
		for(int j=0; j<numCells; j++)
		{
			if(data[i][j]>pshBinMaxVal)
			{
				pshBinMaxVal=data[i][j];
			}
		}
	}
}

PSHData::~PSHData()
{
	delete[] data[0];
	delete[] data;
}

void PSHData::exportPSH(ofstream &outfile)
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
	outfile.write((char *)data[0], numCells*totalNumBins*sizeof(unsigned int));
}

unsigned int PSHData::getCellNum()
{
	return numCells;
}

unsigned int PSHData::getNumTrials()
{
	return numTrials;
}

unsigned int PSHData::getPreStimNumBins()
{
	return preStimNumBins;
}

unsigned int PSHData::getStimNumBins()
{
	return stimNumBins;
}

unsigned int PSHData::getPostStimNumBins()
{
	return postStimNumBins;
}

unsigned int PSHData::getTotalNumBins()
{
	return totalNumBins;
}

unsigned int PSHData::getBinTimeSize()
{
	return binTimeSize;
}

unsigned int PSHData::getPSHBinMaxVal()
{
	return pshBinMaxVal;
}

const unsigned int *PSHData::getDataRow(unsigned int binN)
{
	return data[binN];
}
const unsigned int **PSHData::getData()
{
	return (const unsigned int **)data;
}

QPixmap *PSHData::paintPSHPop(unsigned int startCellN, unsigned int endCellN)
{
	stringstream strFormat;
	QString paintStr;

	QPixmap *paintBuf;
	QPainter p;

	unsigned int nPaintCells;
	unsigned int paintTimeWidth;
	unsigned int cellTickStep;
	unsigned int timeTickStep;
	int timeTickStartNum;

	nPaintCells=endCellN-startCellN;
	if(nPaintCells>1024)
	{
		nPaintCells=1024;
	}
	if(startCellN>=numCells)
	{
		startCellN=numCells-1;
	}
	cellTickStep=nPaintCells/2;
	if(cellTickStep>25)
	{
		cellTickStep=25;
	}

	paintTimeWidth=totalNumBins*binTimeSize;
	timeTickStartNum=-(preStimNumBins*binTimeSize);
	timeTickStep=paintTimeWidth/10;
	if(timeTickStep>50)
	{
		timeTickStep=50;
	}

	paintBuf=new QPixmap(paintTimeWidth+100, nPaintCells+100);
	paintBuf->fill(Qt::black);
	p.begin(paintBuf);

	//setting up axes
	p.setPen(Qt::red);
	p.drawLine(0, paintBuf->height()-99, (int)paintTimeWidth, paintBuf->height()-99);
	p.drawLine((int)paintTimeWidth, 0, (int)paintTimeWidth, paintBuf->height()-99);

	p.setPen(Qt::white);
	strFormat.str("");
	strFormat<<"psh bin max val:"<<pshBinMaxVal;
	paintStr=strFormat.str().c_str();
	p.drawText(paintBuf->width()/2-20, paintBuf->height()-40, paintStr);

	p.setPen(Qt::green);
	for(int i=0; i<paintTimeWidth; i+=timeTickStep)
	{
		p.drawLine(i, paintBuf->height()-98, i, paintBuf->height()-93);
		strFormat.str("");
		strFormat<<i+timeTickStartNum;
		paintStr=strFormat.str().c_str();
		p.drawText(i, paintBuf->height()-83, paintStr);
	}

	p.drawText(paintBuf->width()/2-20, paintBuf->height()-70, "time (ms)");

	for(int i=0; i<nPaintCells; i+=cellTickStep)
	{
		int dispNum;

		dispNum=i+startCellN;

		p.drawLine((int)paintTimeWidth, i, (int)paintTimeWidth+5, i);
		strFormat.str("");
		strFormat<<dispNum;
		paintStr=strFormat.str().c_str();
		p.drawText((int)paintTimeWidth+8, i+10, paintStr);
	}
	p.drawText(paintTimeWidth+40, nPaintCells+2, "cell #");
	//end axes

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

			greyVal=(int)(((float)data[i][j]/pshBinMaxVal)*255);
			paintColor.setRgb(greyVal, greyVal, greyVal, 255);

			p.setPen(paintColor);
			p.drawLine(binTStart, j-startCellN, binTEnd, j-startCellN);
		}
	}

	p.setPen(Qt::blue);
	p.drawLine(preStimNumBins*binTimeSize, 0,
			preStimNumBins*binTimeSize, nPaintCells);
	p.drawLine((preStimNumBins+stimNumBins)*binTimeSize, 0,
			(preStimNumBins+stimNumBins)*binTimeSize, nPaintCells);

	p.end();

	return paintBuf;
}

QPixmap *PSHData::paintPSHInd(unsigned int cellN)
{
	QPixmap *paintBuf;

	float *psh;
//	stringstream strFormat;
//	QString paintStr;
//	QPainter p;
//
//	unsigned int paintSpC;
//	unsigned int paintTimeWidth;
//	unsigned int spCTickStep;
//	unsigned int spCPaintStep;
//	unsigned int spCNumTicks;
//	unsigned int spCDispNum;
//	unsigned int timeTickStep;
//	int timeTickStartNum;
//
	if(cellN>=numCells)
	{
		cellN=numCells-1;
	}

	psh=new float[totalNumBins];
//
//
//	paintSpC=500;
//	spCNumTicks=21;
//	spCPaintStep=paintSpC/spCNumTicks;
//	spCTickStep=pshBinMaxVal/spCNumTicks;
//
//	if(spCTickStep<1)
//	{
//		spCTickStep=1;
//	}
//
//	paintTimeWidth=totalNumBins*binTimeSize;
//	timeTickStep=paintTimeWidth/10;
//	timeTickStartNum=-(preStimNumBins*binTimeSize);
//	if(timeTickStep>50)
//	{
//		timeTickStep=50;
//	}
//
//	paintBuf=new QPixmap(paintTimeWidth+100, paintSpC+100);
//	paintBuf->fill(Qt::black);
//	p.begin(paintBuf);
//
//	//setting up axes
//	p.setPen(Qt::red);
//	p.drawLine(0, paintBuf->height()-99, (int)paintTimeWidth, paintBuf->height()-99);
//	p.drawLine((int)paintTimeWidth, 0, (int)paintTimeWidth, paintBuf->height()-99);
//
//	p.setPen(Qt::white);
//	strFormat.str("");
//	strFormat<<"psh bin max val:"<<pshBinMaxVal;
//	paintStr=strFormat.str().c_str();
//	p.drawText(paintBuf->width()/2-20, paintBuf->height()-40, paintStr);
//
//	p.setPen(Qt::green);
//	for(int i=0; i<paintTimeWidth; i+=timeTickStep)
//	{
//		p.drawLine(i, paintBuf->height()-98, i, paintBuf->height()-93);
//		strFormat.str("");
//		strFormat<<i+timeTickStartNum;
//		paintStr=strFormat.str().c_str();
//		p.drawText(i, paintBuf->height()-83, paintStr);
//	}
//
//	p.drawText(paintBuf->width()/2-20, paintBuf->height()-70, "time (ms)");
//
//	spCDispNum=0;
//	for(int i=paintSpC; i>=0; i-=spCPaintStep)
//	{
//
//		p.drawLine((int)paintTimeWidth, i, (int)paintTimeWidth+5, i);
//		strFormat.str("");
//		strFormat<<spCDispNum;
//		paintStr=strFormat.str().c_str();
//		p.drawText((int)paintTimeWidth+8, i+10, paintStr);
//		spCDispNum+=spCTickStep;
//	}
//	p.drawText(paintTimeWidth+40, paintSpC+2, "spike count");
//	//end axes
//
//	p.fillRect((int)preStimNumBins*binTimeSize, 0,
//			(int)stimNumBins*binTimeSize, (int)paintSpC, Qt::blue);
//	for(int i=0; i<totalNumBins; i++)
//	{
//		int binX;
//		int binHeight;
//		int binY;
//		binX=i*binTimeSize;
//
//		binHeight=paintSpC*((float)data[i][cellN])/pshBinMaxVal;
//		binY=paintSpC-binHeight;
//
//		p.fillRect(binX, binY, (int)binTimeSize, binHeight, Qt::white);
//	}
//
//	p.end();
//
	for(int i=0; i<totalNumBins; i++)
	{
		psh[i]=data[i][cellN];
	}

	paintBuf=paintPSH(psh);
	delete[] psh;
	return paintBuf;
}

QPixmap *PSHData::paintPSH(float *psh)
{
	QPixmap *paintBuf;
	stringstream strFormat;
	QString paintStr;
	QPainter p;

	unsigned int paintSpC;
	unsigned int paintTimeWidth;
	unsigned int spCTickStep;
	unsigned int spCPaintStep;
	unsigned int spCNumTicks;
	unsigned int spCDispNum;
	unsigned int timeTickStep;
	int timeTickStartNum;

	paintSpC=500;
	spCNumTicks=21;
	spCPaintStep=paintSpC/spCNumTicks;
	spCTickStep=pshBinMaxVal/spCNumTicks;

	if(spCTickStep<1)
	{
		spCTickStep=1;
	}

	paintTimeWidth=totalNumBins*binTimeSize;
	timeTickStep=paintTimeWidth/10;
	timeTickStartNum=-(preStimNumBins*binTimeSize);
	if(timeTickStep>50)
	{
		timeTickStep=50;
	}

	paintBuf=new QPixmap(paintTimeWidth+100, paintSpC+100);
	paintBuf->fill(Qt::white);
	p.begin(paintBuf);

	//setting up axes
	p.setPen(Qt::red);
	p.drawLine(0, paintBuf->height()-99, (int)paintTimeWidth, paintBuf->height()-99);
	p.drawLine((int)paintTimeWidth, 0, (int)paintTimeWidth, paintBuf->height()-99);

//	p.setPen(Qt::white);
//	strFormat.str("");
//	strFormat<<"psh bin max val:"<<pshBinMaxVal;
//	paintStr=strFormat.str().c_str();
//	p.drawText(paintBuf->width()/2-20, paintBuf->height()-40, paintStr);

	p.setPen(Qt::green);
	for(int i=0; i<paintTimeWidth; i+=timeTickStep)
	{
		p.drawLine(i, paintBuf->height()-98, i, paintBuf->height()-93);
		strFormat.str("");
		strFormat<<i+timeTickStartNum;
		paintStr=strFormat.str().c_str();
		p.drawText(i, paintBuf->height()-83, paintStr);
	}

	p.drawText(paintBuf->width()/2-20, paintBuf->height()-70, "time (ms)");

	spCDispNum=0;
	for(int i=paintSpC; i>=0; i-=spCPaintStep)
	{

		p.drawLine((int)paintTimeWidth, i, (int)paintTimeWidth+5, i);
		strFormat.str("");
		strFormat<<spCDispNum;
		paintStr=strFormat.str().c_str();
		p.drawText((int)paintTimeWidth+8, i+10, paintStr);
		spCDispNum+=spCTickStep;
	}
	p.drawText(paintTimeWidth+40, paintSpC+2, "spike count");
	//end axes

	p.fillRect((int)preStimNumBins*binTimeSize, 0,
			(int)stimNumBins*binTimeSize, (int)paintSpC, Qt::gray);
	for(int i=0; i<totalNumBins; i++)
	{
		int binX;
		int binHeight;
		int binY;
		binX=i*binTimeSize;

		binHeight=paintSpC*(psh[i])/pshBinMaxVal;
		binY=paintSpC-binHeight;

		p.fillRect(binX, binY, (int)binTimeSize, binHeight, Qt::black);
	}

	p.end();

	return paintBuf;
}
