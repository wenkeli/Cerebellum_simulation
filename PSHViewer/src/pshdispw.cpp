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
		wW=ALLVIEWPW;
		wH=ALLVIEWPH+100;
		windowTitle<<"all "<<cellTypeText[cellT]<<"s PSH starting at cell #"<<cN<<"(GR only)";

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
		p.drawLine(0, wH-99, wW, wH-99);

		for(int i=0; i<wW; i+=100)
		{
			p.setPen(Qt::blue);
			p.drawLine(i, wH-98, i, wH-93);
			strFormat.str("");
			strFormat<<i;
			paintStr=strFormat.str().c_str();
			p.setPen(Qt::green);
			p.drawText(i, wH-83, paintStr);
		}
		p.drawText(wW/2-20, wH-70, "time (ms)");


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
			strFormat.str("");
			strFormat<<"GR bin Max value: "<<pshGRMax;
			paintStr=strFormat.str().c_str();
			p.drawText(wW/2-20, wH-40, paintStr);

			for(int i=0; i<ALLVIEWPW; i++)
			{
				QColor paintColor;
				int binN, greyVal;

				binN=(int)(i/((float)ALLVIEWPW/NUMBINS));
				for(int j=cellN*ALLVIEWPH; j<(cellN+1)*ALLVIEWPH; j++)
				{
					greyVal=(int)(((float)pshGR[binN][j]/pshGRMax)*255);
					paintColor.setRgb(greyVal, greyVal, greyVal, 255);
					p.setPen(paintColor);
					p.drawPoint(i, j%ALLVIEWPH);
				}
			}
		}
	}
	else
	{
		if(cellT==0)
		{

		}
		else if(cellT==1)
		{

		}
		else
		{

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
