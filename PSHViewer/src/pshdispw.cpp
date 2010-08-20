#include "../includes/pshdispw.h"
#include "../includes/moc_pshdispw.h"

PSHDispw::PSHDispw(QWidget *parent, int t, int cT, int cN)
    : QWidget(parent)
{
	stringstream windowTitle;
	string cellTypeText[3]={"Mossy fiber", "Golgi Cell", "Granule Cell"};

	ui.setupUi(this);
	type=t;
	cellT=cT;
	cellN=cN;

	this->setAttribute(Qt::WA_DeleteOnClose);

	if(type==0)
	{
		wW=ALLVIEWPW+100;
		wH=ALLVIEWPH+100;
		windowTitle<<"all "<<cellTypeText[cellT]<<"s PSH starting at cell #"<<cN*ALLVIEWPH<<"(GR only)";

	}
	else
	{
		wW=SINGLEVIEWPW+100;
		wH=SINGLEVIEWPH+100;
		windowTitle<<"single "<<cellTypeText[cellT]<<" PSH for cell #"<<cN;
	}

	backBuf=new QPixmap(wW, wH);
	this->setFixedSize(wW, wH);
	backBuf->fill(Qt::black);
	this->setWindowTitle(windowTitle.str().c_str());

	paintPSH();
}

PSHDispw::~PSHDispw()
{
	delete backBuf;
}

void PSHDispw::paintPSH()
{
	stringstream strFormat;
	QString paintStr;
//	cout<<"type: "<<type<<" cellT:"<<cellT<<" cellN"<<cellN<<endl;
	QPainter p;

	p.begin(backBuf);

	if(type==0)
	{
		p.setPen(Qt::red);
		p.drawLine(0, wH-99, ALLVIEWPW, wH-99);
		p.drawLine(ALLVIEWPW, 0, ALLVIEWPW, wH-99);

		p.setPen(Qt::green);
		for(int i=0; i<wW; i+=100)
		{
			p.drawLine(i, wH-98, i, wH-93);
			strFormat.str("");
			strFormat<<i;
			paintStr=strFormat.str().c_str();
			p.drawText(i, wH-83, paintStr);
		}
		p.drawText(wW/2-20, wH-70, "time (ms)");

		p.setPen(Qt::green);
		for(int i=0; i<ALLVIEWPH; i+=25)
		{
			int dispNum=i;
			if(cellT==2)
			{
				dispNum=ALLVIEWPH*cellN+i;
			}
			p.drawLine(ALLVIEWPW, i, ALLVIEWPW+5, i);
			strFormat.str("");
			strFormat<<dispNum;
			paintStr=strFormat.str().c_str();
			p.drawText(ALLVIEWPW+8, i, paintStr);
		}
		p.drawText(ALLVIEWPW+40, ALLVIEWPH/2, "cell #");

		p.setPen(Qt::white);
		if(cellT==0)
		{
			strFormat.str("");
			strFormat<<"MF bin Max value: "<<pshMFMax;
			paintStr=strFormat.str().c_str();
			p.drawText(wW/2-20, wH-40, paintStr);

			for(int i=0; i<ALLVIEWPW; i++)
			{
				QColor paintColor;
				int binN, greyVal;

				binN=(int)(i/((float)ALLVIEWPW/NUMBINS));
				for(int j=0; j<NUMMF; j++)
				{
					greyVal=(int)(((float)pshMF[binN][j]/pshMFMax)*255);
					paintColor.setRgb(greyVal, greyVal, greyVal, 255);
					p.setPen(paintColor);
					p.drawPoint(i, j);
				}
			}
		}
		else if(cellT==1)
		{
			strFormat.str("");
			strFormat<<"GO bin Max value: "<<pshGOMax;
			paintStr=strFormat.str().c_str();
			p.drawText(wW/2-20, wH-40, paintStr);

			for(int i=0; i<ALLVIEWPW; i++)
			{
				QColor paintColor;
				int binN, greyVal;

				binN=(int)(i/((float)ALLVIEWPW/NUMBINS));
				for(int j=0; j<NUMMF; j++)
				{
					greyVal=(int)(((float)pshGO[binN][j]/pshGOMax)*255);
					paintColor.setRgb(greyVal, greyVal, greyVal, 255);
					p.setPen(paintColor);
					p.drawPoint(i, j);
				}
			}
		}
		else
		{
			int i=0;
			strFormat.str("");
			strFormat<<"GR bin Max value: "<<pshGRMax;
			paintStr=strFormat.str().c_str();
			p.drawText(wW/2-20, wH-40, paintStr);


			//for(int i=0; i<ALLVIEWPW; i++)
			{
				QColor paintColor;
				int binN, greyVal;
				int startCellN, endCellN;

				startCellN=cellN*ALLVIEWPH;
				endCellN=(cellN+1)*ALLVIEWPH;
				if(startCellN>=pshValGR.size())
				{
					startCellN=(cellN-1)*ALLVIEWPH;
					endCellN=pshValGR.size();
				}
				if(endCellN>=pshValGR.size())
				{
					endCellN=pshValGR.size();
				}

				cout<<startCellN<<" "<<endCellN<<" "<<pshValGR.size()<<endl;

				for(int i=startCellN; i<endCellN; i++)
				{
					for(int j=0; j<ALLVIEWPW; j++)
					{
						binN=(int)(j/((float)ALLVIEWPW/NUMBINS));
						greyVal=(int)(((float)pshValGR[i][binN]/pshGRMax)*255);
						paintColor.setRgb(greyVal, greyVal, greyVal, 255);
						p.setPen(paintColor);
						p.drawPoint(j, i%ALLVIEWPH);
					}
				}

//				for(int j=0; j<NUMGR; j++)
//				{
//					int maxVal;
//					maxVal=0;
//
//					if(i>=ALLVIEWPH)
//					{
//						break;
//					}
//					for(int k=0; k<NUMBINS; k++)
//					{
//						if(pshGR[k][j]>maxVal)
//						{
//							maxVal=pshGR[k][j];
//						}
//					}
//					if(maxVal<(pshGRMax/3))
//					{
//						continue;
//					}
//
//					for(int k=0; k<ALLVIEWPW; k++)
//					{
//						binN=(int)(k/((float)ALLVIEWPW/NUMBINS));
//						greyVal=(int)(((float)pshGR[binN][j]/pshGRMax)*255);
//						paintColor.setRgb(greyVal, greyVal, greyVal, 255);
//						p.setPen(paintColor);
//						p.drawPoint(k, i);
//					}
//					i++;
//				}

//				binN=(int)(i/((float)ALLVIEWPW/NUMBINS));
//				for(int j=cellN*ALLVIEWPH; j<(cellN+1)*ALLVIEWPH; j++)
//				{
//					greyVal=(int)(((float)pshGR[binN][j]/pshGRMax)*255);
//					paintColor.setRgb(greyVal, greyVal, greyVal, 255);
//					p.setPen(paintColor);
//					p.drawPoint(i, j%ALLVIEWPH);
//				}
			}
		}
	}
	else
	{
		int yMaxVal=1;
		int yInc=2;
		p.setPen(Qt::red);
		p.drawLine(99, 0, 99, wH-99);
		p.drawLine(99, wH-99, wW, wH-99);

		p.setPen(Qt::green);
		for(int i=0; i<SINGLEVIEWPW; i+=100)
		{
			p.drawLine(i+100, wH-98, i+100, wH-93);
			strFormat.str("");
			strFormat<<i;
			paintStr=strFormat.str().c_str();
			p.drawText(i+100, wH-83, paintStr);
		}
		p.drawText(wW/2+30, wH-70, "time (ms)");
		p.drawText(5, SINGLEVIEWPH/2, "# of spikes");

		p.setPen(Qt::white);
		if(cellT==0)
		{
			if(cellN>=NUMMF)
			{
				cellN=NUMMF-1;
			}

			for(int i=0; i<NUMBINS; i++)
			{
				if(pshMF[i][cellN]>yMaxVal)
				{
					yMaxVal=pshMF[i][cellN];
				}
			}

			for(int i=0; i<NUMBINS; i++)
			{
				int yPos, xPos;
				yPos=SINGLEVIEWPH-(int)(((float)(pshMF[i][cellN])/yMaxVal)*SINGLEVIEWPH);
				xPos=(int)(i*((float)SINGLEVIEWPW/NUMBINS))+100;

				p.fillRect(xPos, yPos, SINGLEVIEWPW/NUMBINS, SINGLEVIEWPH-yPos, Qt::white);
			}
		}
		else if(cellT==1)
		{
			if(cellN>=NUMGO)
			{
				cellN=NUMGO-1;
			}
			for(int i=0; i<NUMBINS; i++)
			{
				if(pshGO[i][cellN]>yMaxVal)
				{
					yMaxVal=pshGO[i][cellN];
				}
			}
			for(int i=0; i<NUMBINS; i++)
			{
				int yPos, xPos;
				yPos=SINGLEVIEWPH-(int)(((float)(pshGO[i][cellN])/yMaxVal)*SINGLEVIEWPH);
				xPos=(int)(i*((float)SINGLEVIEWPW/NUMBINS))+100;

				p.fillRect(xPos, yPos, SINGLEVIEWPW/NUMBINS, SINGLEVIEWPH-yPos, Qt::white);
			}
		}
		else
		{
			if(cellN>=pshValGR.size())
			{
				cellN=pshValGR.size()-1;
			}

			for(int i=0; i<NUMBINS; i++)
			{
				if(pshValGR[cellN][i]>yMaxVal)
				{
					yMaxVal=pshValGR[cellN][i];
				}
			}
			for(int i=0; i<NUMBINS; i++)
			{
				int yPos, xPos;
				yPos=SINGLEVIEWPH-(int)(((float)(pshValGR[cellN][i])/2000)*SINGLEVIEWPH);//yMaxVal
				xPos=(int)(i*((float)SINGLEVIEWPW/NUMBINS))+100;

				p.fillRect(xPos, yPos, SINGLEVIEWPW/NUMBINS, SINGLEVIEWPH-yPos, Qt::white);
			}

//			for(int i=0; i<NUMBINS; i++)
//			{
//				if(pshGR[i][cellN]>yMaxVal)
//				{
//					yMaxVal=pshGR[i][cellN];
//				}
//			}
//			for(int i=0; i<NUMBINS; i++)
//			{
//				int yPos, xPos;
//				yPos=SINGLEVIEWPH-(int)(((float)(pshGR[i][cellN])/yMaxVal)*SINGLEVIEWPH);
//				xPos=(int)(i*((float)SINGLEVIEWPW/NUMBINS))+100;
//
//				p.fillRect(xPos, yPos, SINGLEVIEWPW/NUMBINS, SINGLEVIEWPH-yPos, Qt::white);
//			}
		}

		strFormat.str("");
		strFormat<<"max # of spikes: "<<pshGRMax;
		paintStr=strFormat.str().c_str();
		p.drawText(wW/2-50, wH-40, paintStr);

		if(yMaxVal<10)
		{
			yInc=1;
		}
		else
		{
			yInc=ceil((float)pshGRMax/10);//yMaxVal
		}
		p.setPen(Qt::green);
		for(int i=0; i<pshGRMax; i+=yInc)//yMaxVal
		{
			int yPos;
			yPos=SINGLEVIEWPH-(int)(((float)i/pshGRMax)*SINGLEVIEWPH);//yMaxVal
			p.drawLine(95, yPos, 99, yPos);
			strFormat.str("");
			strFormat<<i;
			paintStr=strFormat.str().c_str();
			p.drawText(75, yPos, paintStr);
		}
	}
	p.end();
}

void PSHDispw::paintEvent(QPaintEvent *event)
{
	QPainter painter(this);
	if(backBuf==NULL)
	{
		this->close();
		return;
	}

	painter.drawPixmap(event->rect(), *backBuf, event->rect());
}
