#include "../../includes/gui/spikeratesdispw.h"
#include "../../includes/gui/moc_spikeratesdispw.h"

#define HISTNUMBINS 100

SpikeRatesDispW::SpikeRatesDispW(QWidget *parent, int numCs, unsigned int *spikeS, QMutex *accessSpikeSumL, int nBs, string title)
    : QWidget(parent)
{
//	float timeS;
//	float spikeRates[NUMGR];
//	float minSpikeRate, maxSpikeRate;
//	float binEdges[HISTNUMBINS+1];
//	float binWidth;
//	int count[HISTNUMBINS];
//
//	int numCells;
//	unsigned int *spikeCounts;
//	QMutex *accessSpikeCountsLock;
//	int numBins;
//
//	ui.setupUi(this);
//	numCells=numCs;
//	spikeCounts=spikeS;
//	accessSpikeCountsLock=accessSpikeSumL;
//	windowTitle=title;
//
//	this->setAttribute(Qt::WA_DeleteOnClose);
//	this->setFixedSize(600, 600);
//	backBuf=new QPixmap(this->width(), this->height());
//	backBuf->fill(Qt::black);
//
//	numBins=nBs;
//	if(numBins>HISTNUMBINS)
//	{
//		numBins=HISTNUMBINS;
//	}
//
////	simDispTypeLock.lock();
////	if(simDispType==1)
////	{
////		numCells=NUMGR;
////		numBins=HISTNUMBINS;
////		spikeCounts=(unsigned int *)spikeSumGR;
////	}
////	else
////	{
////		numCells=NUMGO;
////		numBins=25;
////		spikeCounts=(unsigned int *)spikeSumGO;
////	}
////	simDispTypeLock.unlock();
//
////	msCountLock.lock();
//	accessSpikeCountsLock->lock();
//	timeS=msCount/1000.0f;
////	msCountLock.unlock();
//
////	spikeSumGOLock.lock();
////	spikeSumGRLock.lock();
//
//	maxSpikeRate=0;
//	minSpikeRate=1000;
//	for(int i=0; i<numCells; i++)
//	{
//		spikeRates[i]=spikeCounts[i]/timeS;
//		if(spikeRates[i]>maxSpikeRate)
//		{
//			maxSpikeRate=spikeRates[i];
//		}
//		if(spikeRates[i]<minSpikeRate)
//		{
//			minSpikeRate=spikeRates[i];
//		}
//	}
//	accessSpikeCountsLock->unlock();
////	spikeSumGOLock.unlock();
////	spikeSumGRLock.unlock();
//
//	if(minSpikeRate>=maxSpikeRate)
//	{
////		cout<<minSpikeRate<<"  "<<maxSpikeRate<<endl;
//		return;
//	}
//
//	binWidth=(maxSpikeRate-minSpikeRate)/((float)numBins);
//	binEdges[0]=minSpikeRate;
//	for(int i=0; i<numBins; i++)
//	{
//		binEdges[i+1]=binEdges[i]+binWidth;
//		count[i]=0;
//	}
//
//	for(int i=0; i<numCells; i++)
//	{
//		for(int j=0; j<numBins; j++)
//		{
//			if(spikeRates[i]>=binEdges[j] && spikeRates[i]<binEdges[j+1])
//			{
//				count[j]++;
//				break;
//			}
//		}
//		if(spikeRates[i]>=binEdges[numBins])
//		{
//			count[numBins-1]++;
//		}
//	}
//
//	paintBuf(binEdges, count, numBins);
}

SpikeRatesDispW::~SpikeRatesDispW()
{
	delete backBuf;
}

void SpikeRatesDispW::paintBuf(float *binEdges, int *count, int binNum)
{
	stringstream formatStr;
	int graphWidth, graphHeight;
	int maxCount=0;
	int binWidth;
	float totalBinSpan;
	QPainter p;
	p.begin(backBuf);
	p.setPen(Qt::red);

	graphWidth=backBuf->width()-99;
	graphHeight=backBuf->height()-99;

	p.drawLine(99, 0, 99, graphHeight);
	p.drawLine(99, graphHeight, backBuf->width(), graphHeight);
	p.drawText(backBuf->width()/2, graphHeight+80, windowTitle.c_str());
	for(int i=0; i<binNum; i++)
	{
		if(count[i]>maxCount)
		{
			maxCount=count[i];
		}
	}

//	cout<<maxCount<<endl;
//	cout<<binEdges[0]<<" "<<binEdges[binNum]<<endl;

	p.setPen(Qt::green);
	totalBinSpan=binEdges[binNum]-binEdges[0];
	for(int i=1; i<=4; i++)
	{
		int drawY, drawX;

		drawY=graphHeight-(graphHeight*i)/5;
		drawX=99+(graphWidth*i)/5;
//		cout<<drawX<<" "<<drawY<<endl;

		p.drawLine(94, drawY, 99, drawY);
		p.drawLine(drawX, graphHeight, drawX, graphHeight+5);

		formatStr.str("");
		formatStr<<(maxCount*i)/5;
		p.drawText(20, drawY, formatStr.str().c_str());

		formatStr.str("");
		formatStr<<(totalBinSpan*i)/5.0f+binEdges[0];
		p.drawText(drawX, graphHeight+30, formatStr.str().c_str());
	}

	p.setPen(Qt::white);
	binWidth=graphWidth/binNum;
	for(int i=0; i<binNum; i++)
	{
		int xPos, yPos;
		xPos=i*binWidth+99;
		yPos=graphHeight*(1.0f-(((float)count[i])/maxCount));

//		cout<<xPos<<" "<<yPos<<" "<<binWidth<<" "<<graphHeight-yPos<<endl;

		p.fillRect(xPos, yPos, binWidth, graphHeight-yPos, Qt::white);
	}
}

void SpikeRatesDispW::paintEvent(QPaintEvent *event)
{
	QPainter painter(this);
	if(backBuf==NULL)
	{
		this->close();
		return;
	}

	painter.drawPixmap(event->rect(), *backBuf, event->rect());
}
