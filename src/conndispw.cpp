#include "../includes/conndispw.h"
#include "../includes/moc_conndispw.h"

ConnDispW::ConnDispW(QWidget *parent)
    : QWidget(parent)
{
	ui.setupUi(this);
}

ConnDispW::~ConnDispW()
{

}

void ConnDispW::setBounds(int s, int e)
{
	start=s;
	end=e;
}

void ConnDispW::setDispT(ConnDispT t)
{
	dispT=t;
}

void ConnDispW::paintEvent(QPaintEvent *event)
{
	QPainter painter(this);
	stringstream windowTitle;
	CRandomMother randomGen(0);

	if(start>end || start<0)
	{
		this->close();
		return;
	}

	if((dispT==MFGO || dispT==MFGR) && end>=NUMMF)
	{
		this->close();
		return;
	}
	if(dispT==GRGO && end>=NUMGR)
	{
		this->close();
		return;
	}
	if(dispT==GRGO && end>=NUMGO)
	{
		this->close();
		return;
	}

	//draw initializations

	this->resize(GRX, GRY);
	painter.setPen(Qt::white);

	if(dispT==MFGR)
	{
		windowTitle<<"Mossy fibers (#"<<start<<" to "<<end<<") to granule cells connections";
		this->setWindowTitle(windowTitle.str().c_str());
		windowTitle.str("");
		for(int i=start; i<=end; i++)
		{
			for(int j=0; j<MFGRSYNPERMF; j++)
			{
				int grNum=conMFtoGR[i][j][0];
				int x=grNum%GRX;
				int y=grNum/GRX;
				if(y>=GRY)
				{
					this->close();
					return;
				}
				painter.drawPoint(x, y);
			}
			painter.setPen(QColor(randomGen.iRandom(100, 255), randomGen.iRandom(100, 255), randomGen.iRandom(100, 255)));
		}
		return;
	}
	if(dispT==MFGO)
	{
		float scaleX=(float) GOX/GRX;
		float scaleY=(float) GOY/GRY;
		windowTitle<<"Mossy fibers (#"<<start<<" to "<<end<<") to golgi cells connections";
		this->setWindowTitle(windowTitle.str().c_str());
		windowTitle.str("");

		for(int i=start; i<=end; i++)
		{
			for(int j=0; j<MFGOSYNPERMF; j++)
			{
				int goNum=conMFtoGO[i][j][0];
				int goPosX=goNum%GOX;
				int goPosY=(int) goNum/GOX;
				int grPosX=(int) ((goPosX+0.5)/scaleX);
				int grPosY=(int) ((goPosY+0.5)/scaleY);
				painter.drawPoint(grPosX, grPosY);
			}
			painter.setPen(QColor(randomGen.iRandom(100, 255), randomGen.iRandom(100, 255), randomGen.iRandom(100, 255)));
		}

		return;
	}
	if(dispT==GRGO)
	{
		windowTitle<<"Granule cells (#"<<start<<" to "<<end<<") to golgi cells connections";
		this->setWindowTitle(windowTitle.str().c_str());
		windowTitle.str("");
		//draw
		for(int i=start; i<end; i++)
		{
			for(int j=0; j<GRGOSYNPERGR; j++)
			{

			}
			painter.setPen(QColor(randomGen.iRandom(100, 255), randomGen.iRandom(100, 255), randomGen.iRandom(100, 255)));
		}
		return;
	}
	if(dispT==GOGR)
	{
		windowTitle<<"Golgi cells (#"<<start<<" to "<<end<<") to granule cells connections";
		this->setWindowTitle(windowTitle.str().c_str());
		windowTitle.str("");
		//draw


		/*
		//draw debug, shows GRs with incomplete connections
		for(int i=0; i<incompGRs.size(); i++)
		{
			int grNum=incompGRs[i];
			int x=grNum%GRX;
			int y=grNum/GRX;
			if(y>=GRY)
			{
				this->close();
				return;
			}
			painter.drawPoint(x, y);
		}

		float scaleX=(float) GOX/GRX;
		float scaleY=(float) GOY/GRY;

		painter.setPen(Qt::red);
		for(int i=0; i<NUMGO; i++)
		{
			int goPosX=i%GOX;
			int goPosY=(int) i/GOX;

			int grPosX=(int) ((goPosX+0.5)/scaleX);
			int grPosY=(int) ((goPosY+0.5)/scaleY);
			painter.drawPoint(grPosX, grPosY);
		}*/

		return;
	}
}
