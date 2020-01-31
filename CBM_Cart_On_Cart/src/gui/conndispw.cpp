/*
 * see conndispw.h for descriptions of the functions and private memebers
 */

#include "../../includes/gui/conndispw.h"
#include "../../includes/gui/moc_conndispw.h"

ConnDispW::ConnDispW(QWidget *parent, int s, int e, ConnDispT t)
	: QWidget(parent)
{
	stringstream windowTitle; //formatting string
	ui.setupUi(this);
	start=s;
	end=e;
	dispT=t;
	backBuf=new QPixmap(GRX, GRY); //construct a pixmap with width GRX and height GRY pixels
	backBuf->fill(Qt::black); //fill the pixmap with black color
	this->setAttribute(Qt::WA_DeleteOnClose); //when the window is closed, delete the object
	//error checking
	valid=false;
//	if(!initialized) //check if connections are made
//	{
//		return;
//	}
	//make sure the starting number is not greater than the end number
	//and make sure neither are negative
	if(start>end || start<0)
	{
		return;
	}
	//make sure the ending cell index is no greater than the max index of cell/fibers
	//depending on the type of connections displayed
	if((dispT==MFGO || dispT==MFGR) && end>=NUMMF)
	{
		return;
	}
	if(dispT==GRGO && end>=NUMGR)
	{
		return;
	}
	if(dispT==GRGO && end>=NUMGO)
	{
		return;
	}
	//end error check

	valid=true; //if pass all error checking, set to true
	this->setFixedSize(backBuf->width(), backBuf->height()); //set the fixed window size to the same as the backBuf
	//setting window titles depending on what kind of connections is displayed
	if(dispT==MFGR)
	{
		windowTitle<<"Mossy fibers (#"<<start<<" to "<<end<<") to granule cells connections";
		this->setWindowTitle(windowTitle.str().c_str());
	}
	if(dispT==MFGO)
	{
		windowTitle<<"Mossy fibers (#"<<start<<" to "<<end<<") to golgi cells connections";
		this->setWindowTitle(windowTitle.str().c_str());
	}
	if(dispT==GRGO)
	{
		windowTitle<<"Granule cells to golgi cells connections for golgi cells #"<<start<<" to "<<end;
		this->setWindowTitle(windowTitle.str().c_str());
	}
	if(dispT==GOGR)
	{
		windowTitle<<"Golgi cells (#"<<start<<" to "<<end<<") to granule cells connections";
		this->setWindowTitle(windowTitle.str().c_str());
	}
	paintBuf(); //draw the backbuffer
}

ConnDispW::~ConnDispW()
{
	delete backBuf;
}

void ConnDispW::paintBuf()
{
//	QPainter painter;
//	CRandomSFMT0 randomGen(time(NULL)); //used for setting random colors
//
//	if(!valid || backBuf==NULL)
//	{
//		return;
//	}
//
//	painter.begin(backBuf); //set the painter to draw on the buffer
//	painter.setPen(Qt::white); //always draw the first connections in white
//	if(dispT==MFGR)
//	{
//		for(int i=start; i<=end; i++)
//		{
//			for(int j=0; j<numGROutPerMF[i]; j++)//for(int j=0; j<numSynMFtoGR; j++)
//			{
//				int grNum=conMFOutGRForMF[i][j]/4;
//				//int grNum=conMFtoGR[i][j]/4;
//				int x=grNum%GRX;
//				int y=grNum/GRX;
//				if(y>=GRY)
//				{
//					return;
//				}
//				painter.drawPoint(x, y);
//			}
//			painter.setPen(QColor(randomGen.IRandom(100, 255), randomGen.IRandom(100, 255), randomGen.IRandom(100, 255)));
//		}
//		return;
//	}
//	if(dispT==MFGO)
//	{
//		float scaleX=(float) GOX/GRX;
//		float scaleY=(float) GOY/GRY;
//		for(int i=start; i<=end; i++)
//		{
//			for(int j=0; j<numGOOutPerMF[i]; j++)//for(int j=0; j<numSynMFtoGO[i]; j++)
//			{
////				int goNum=conMFtoGO[i][j];
//				int goNum=conMFOutGOForMF[i][j];
////				if(goNum>=NUMGO)
////					cout<<goNum<<endl;
//				int goPosX=goNum%GOX;
//				int goPosY=(int) goNum/GOX;
//				int grPosX=(int) ((goPosX+0.5)/scaleX); //+0.5 is for shifting the map 0.5/scaleX away from the edge
//				int grPosY=(int) ((goPosY+0.5)/scaleY);
//				painter.drawPoint(grPosX, grPosY);
//			}
//			painter.setPen(QColor(randomGen.IRandom(100, 255), randomGen.IRandom(100, 255), randomGen.IRandom(100, 255)));
//		}
//		return;
//	}
//	//this is different from others, instead of displaying which GO cells a GR cell connects to,
//	//it displays for each GO cell, which GR cells are connected to it
//	if(dispT==GRGO)
//	{
//		float scaleX=(float) GOX/GRX;
//		float scaleY=(float) GOY/GRY;
//		for(int i=start; i<=end; i++)
//		{
//			for(int j=0; j<NUMGR; j++)
//			{
//				for(int k=0; k<numGOOutPerGR[j]; k++)
//				{
//					painter.fillRect((int)(((i%GOX)+0.5)/scaleX-1), (int)((((int)i/GOX)+0.5)/scaleY-1), 3, 3, Qt::red);
//
//					if(conGROutGOForGR[j][k]==i)
//					{
//						int grPosX=j%GRX;
//						int grPosY=j/GRX;
//						if(grPosY>=GRY)
//						{
//							return;
//						}
//						painter.drawPoint(grPosX, grPosY);
//						break;
//					}
//				}
//			}
//			painter.setPen(QColor(randomGen.IRandom(100, 255), randomGen.IRandom(100, 255), randomGen.IRandom(100, 255)));
//		}
//		return;
//	}
//	if(dispT==GOGR)
//	{
//		float scaleX=(float) GOX/GRX;
//		float scaleY=(float) GOY/GRY;
//		for(int i=start; i<=end; i++)
//		{
//			painter.fillRect((int)(((i%GOX)+0.5)/scaleX-1), (int)((((int)i/GOX)+0.5)/scaleY-1), 3, 3, Qt::red);
//			for(int j=0; j<numGROutPerGO[i]; j++) //for(int j=0; j<numSynGOtoGR[i]; j++)
//			{
////				int grNum=conGOtoGR[i][j]/4;
//				int grNum=conGOOutGRForGO[i][j]/4;
//				int grPosX=grNum%GRX;
//				int grPosY=grNum/GRX;
//				if(grPosY>=GRY)
//				{
//					//cout<<"here"<<endl;
//					return;
//				}
//				painter.drawPoint(grPosX, grPosY);
//			}
//			painter.setPen(QColor(randomGen.IRandom(100, 255), randomGen.IRandom(100, 255), randomGen.IRandom(100, 255)));
//		}
//#ifdef DEBUG //draw debug for GOGR
//		// shows GRs with incomplete connections
//		for(unsigned int i=0; i<incompGRsGOGR.size(); i++)
//		{
//			int grNum=incompGRsGOGR[i];
//			int x=grNum%GRX;
//			int y=grNum/GRX;
//			if(y>=GRY)
//			{
//				this->close();
//				return;
//			}
//			painter.drawPoint(x, y);
//		}
//
//		//draw the GO cell grid
//		painter.setPen(Qt::red);
//		for(int i=0; i<NUMGO; i++)
//		{
//			int goPosX=i%GOX;
//			int goPosY=(int) i/GOX;
//
//			int grPosX=(int) ((goPosX+0.5)/scaleX);
//			int grPosY=(int) ((goPosY+0.5)/scaleY);
//			painter.drawPoint(grPosX, grPosY);
//		}
//
//		//draw the GO cells that have incomplete connections
//		painter.setPen(Qt::cyan);
//		for(unsigned int i=0; i<incompGOsGOGR.size(); i++)
//		{
//			int goPosX=incompGOsGOGR[i]%GOX;
//			int goPosY=(int) incompGOsGOGR[i]/GOX;
//
//			int grPosX=(int) ((goPosX+0.5)/scaleX);
//			int grPosY=(int) ((goPosY+0.5)/scaleY);
//			painter.drawPoint(grPosX, grPosY);
//		}
//#endif
//		return;
//	}
}


void ConnDispW::paintEvent(QPaintEvent *event)
{
	QPainter painter(this);
	if(!valid || backBuf==NULL)
	{
		this->close();
		return;
	}
	//copy the region that needs to be redrawn from backBuf to the screen
	painter.drawPixmap(event->rect(), *backBuf, event->rect());
}
