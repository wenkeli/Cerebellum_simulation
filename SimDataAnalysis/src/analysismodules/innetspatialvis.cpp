/*
 * innetspatialvis.cpp
 *
 *  Created on: May 1, 2012
 *      Author: consciousness
 */

#include "../../includes/analysismodules/innetspatialvis.h"

using namespace std;
InNetSpatialVis::InNetSpatialVis(PSHData *gr, PSHData *go)
{
	grPSH=gr;
	goPSH=go;

	numTrials=gr->getNumTrials();
	totalNumBins=gr->getTotalNumBins();

	numGR=gr->getCellNum();
	numGO=go->getCellNum();

	grPSHMaxVal=gr->getPSHBinMaxVal();
	goPSHMaxVal=go->getPSHBinMaxVal();

	bufs=new QPixmap*[totalNumBins];

	for(int i=0; i<totalNumBins; i++)
	{
		QPixmap *paintBuf;
		QPainter p;

		const unsigned int *grPSHRow;
		const unsigned int *goPSHRow;

		unsigned int grGridX;
	//	unsigned int goGridX;
		unsigned int grGridY;

		cout<<"painting bufs #"<<i<<endl;

		grGridX=2048;
		grGridY=512;

		grPSHRow=grPSH->getDataRow(i);
		goPSHRow=goPSH->getDataRow(i);

		paintBuf=new QPixmap(grGridX, grGridY);

	//	cout<<"here"<<endl;

		p.begin(paintBuf);
		for(int j=0; j<numGR; j++)
		{
			QColor paintColor;
			int greyVal;
			int paintX;
			int paintY;

			greyVal=(int)(((float)grPSHRow[j]/numTrials)*255);
			greyVal=(greyVal>255)*255+(greyVal<=255)*greyVal;
			paintColor.setRgb(greyVal, greyVal, greyVal, 255);

			paintX=j%grGridX;
			paintY=j/grGridX;
			p.setPen(paintColor);
			p.drawPoint(paintX, paintY);
		}
		p.end();

		bufs[i]=paintBuf;
	}
}

InNetSpatialVis::~InNetSpatialVis()
{
	for(int i=0; i<totalNumBins; i++)
	{
		delete bufs[i];
	}
	delete[] bufs;
}

QPixmap *InNetSpatialVis::paintSpatial(unsigned int binN)
{
	QPixmap *paintBuf;

	unsigned int grGridX;
//	unsigned int goGridX;
	unsigned int grGridY;

	grGridX=2048;
	grGridY=512;

	paintBuf=new QPixmap(*bufs[binN]);

	return paintBuf;
}




