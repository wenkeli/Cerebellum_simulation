#include "../../includes/gui/simdispw.h"
#include "../../includes/gui/moc_simdispw.h"
#include "../../includes/externalmodules/cartpole.h"
#include <cmath>

SimDispW::SimDispW(QWidget *parent)
    : QWidget(parent)
{
	ui.setupUi(this);
	this->setAttribute(Qt::WA_DeleteOnClose); //when the window is closed, delete the object
	this->setFixedSize(2000, 1024);
	backBuf=new QPixmap(this->width(), this->height());
	backBuf->fill(Qt::black);

}

SimDispW::~SimDispW()
{
	delete backBuf;
}

//QPixmap *SimDispW::getBackBuf()
//{
//	return backBuf;
//}


void SimDispW::drawRaster(vector<bool> aps, int t)
{
	QPainter p;

	p.begin(backBuf);
	p.setWindow(0,0, 5000, 1024);
	p.setViewport(0, 0, backBuf->width(), backBuf->height());

	p.setPen(Qt::white);
	for(int i=0; i<NUMMF; i++)
	{
		if(aps[i])
		{
			p.drawPoint(t, i);
		}
	}


#if defined CARTPOLE
        drawCartPoleViz(p,t);
#endif

	p.scale(backBuf->width()/5000.0, backBuf->height()/1024.0);
	QRect updateArea(p.worldTransform().mapRect(QRect(t, 0, 1, NUMMF)));
	if(updateArea.width()<1)
	{
		updateArea.setWidth(1);
	}
	p.end();

	this->update(updateArea);
}

void SimDispW::drawPSH(vector<unsigned short> greyVals, int t, bool inCS)
{
	QPainter p;
	QColor pshC;

	p.begin(backBuf);
	p.setWindow(0,0, 5000, 1024);
	p.setViewport(0, 0, backBuf->width(), backBuf->height());

	if(inCS)
	{
		for(int i=0; i<NUMMF; i++)
		{
			pshC.setRgb(greyVals[i], greyVals[i], 255, 255);
			p.setPen(pshC);
			p.drawPoint(t, i);
		}
	}
	else
	{
		for(int i=0; i<NUMMF; i++)
		{
			pshC.setRgb(greyVals[i], greyVals[i], 127, 255);
			p.setPen(pshC);
			p.drawPoint(t, i);
		}
	}

	p.scale(backBuf->width()/5000.0, backBuf->height()/1024.0);
	QRect updateArea(p.worldTransform().mapRect(QRect(t, 0, 1, NUMMF)));
	if(updateArea.width()<1)
	{
		updateArea.setWidth(1);
	}
	p.end();

	this->update(updateArea);
}

void SimDispW::drawSCBCPCActs(SCBCPCActs acts, int t)
{
	QPainter p;

	p.begin(backBuf);
	p.setWindow(0,0, 5000, 1024);
	p.setViewport(0, 0, backBuf->width(), backBuf->height());

	p.setPen(Qt::red);
	for(int i=0; i<NUMPC; i++)
	{
		p.drawPoint(t, (int)(12*(i-(acts.vPC[i]/100.0f)))+NUMSC+NUMBC);//
		if(acts.apPC[i])
		{
			p.drawLine(t, (12*i)+NUMSC+NUMBC, t, (int)(12*(i-(acts.vPC[i]/100)))+NUMSC+NUMBC);
		}
	}

	p.setPen(Qt::white);
	for(int i=0; i<NUMSC; i++)
	{
		if(acts.apSC[i])
		{
			p.drawPoint(t, i);
		}
	}
	p.setPen(Qt::green);
	for(int i=0; i<NUMBC; i++)
	{
		if(acts.apBC[i])
		{
			p.drawPoint(t, i+NUMSC);
		}
	}

	p.scale(backBuf->width()/5000.0, backBuf->height()/1024.0);
	QRect updateArea(p.worldTransform().mapRect(QRect(t, 0, 1, NUMMF)));
	if(updateArea.width()<1)
	{
		updateArea.setWidth(1);
	}
	p.end();

	this->update(updateArea);
}

void SimDispW::drawIONCPCActs(IONCPCActs acts, int t)
{
	QPainter p;

	p.begin(backBuf);
	p.setWindow(0,0, 5000, 1024);
	p.setViewport(0, 0, backBuf->width(), backBuf->height());

	p.setPen(Qt::red);
	for(int i=0; i<NUMPC; i++)
	{
		p.drawPoint(t, (int)(12*(i-(acts.vPC[i]/100.0f)))+NUMSC+NUMBC);//
		if(acts.apPC[i])
		{
			p.drawLine(t, (12*i)+NUMSC+NUMBC, t, (int)(12*(i-(acts.vPC[i]/100)))+NUMSC+NUMBC);
		}
	}

	p.setPen(Qt::white);
	for(int i=0; i<NUMIO; i++)
	{
		p.drawPoint(t, 120*(i-acts.vIO[i]/100.0f));
		if(acts.apIO[i])
		{
			p.drawLine(t, 120*i, t, 120*(i-acts.vIO[i]/100.0f));
		}
	}

	p.setPen(Qt::green);
	for(int i=0; i<NUMNC; i++)
	{
		p.drawPoint(t, 15*(i-acts.vNC[i]/100.0f)+480);
		if(acts.apNC[i])
		{
			p.drawLine(t, 15*i+480, t, 15*(i-acts.vNC[i]/100.0f)+480);
		}
	}

	p.setPen(Qt::white);
//	testOutputLock.lock();

#if defined EYELID
	simMZDispNumLock.lock();
	p.drawPoint(t, (int)(512*outputMod[simMZDispNum]->exportOutput()));
	simMZDispNumLock.unlock();
#endif        

#if defined CARTPOLE
        drawCartPoleViz(p,t);
#endif

	p.scale(backBuf->width()/5000.0, backBuf->height()/1024.0);
	QRect updateArea(p.worldTransform().mapRect(QRect(t, 0, 1, NUMMF)));
	updateArea.setWidth((updateArea.width()<=1)+(updateArea.width()>1)*updateArea.width());

	p.end();

	this->update(updateArea);
}

void SimDispW::drawTotalAct(int y, int t)
{
	QPainter p;

	p.begin(backBuf);
	p.setWindow(0,0, 5000, 1024);
	p.setViewport(0, 0, backBuf->width(), backBuf->height());

	p.setPen(Qt::red);
	p.drawPoint(t, y);

	p.scale(backBuf->width()/5000.0, backBuf->height()/1024.0);
	QRect updateArea(p.worldTransform().mapRect(QRect(t, 0, 1, NUMMF)));
	if(updateArea.width()<1)
	{
		updateArea.setWidth(1);
	}
	p.end();

	this->update(updateArea);
}

void SimDispW::drawCSBackground(int t)
{
	QPainter p;

	p.begin(backBuf);
	p.setWindow(0,0, 5000, 1024);
	p.setViewport(0, 0, backBuf->width(), backBuf->height());

	p.setPen(Qt::blue);
	p.drawLine(t, 0, t, NUMMF);

	p.scale(backBuf->width()/5000.0, backBuf->height()/1024.0);
	QRect updateArea(p.worldTransform().mapRect(QRect(t, 0, 1, NUMMF)));
	if(updateArea.width()<1)
	{
		updateArea.setWidth(1);
	}
	p.end();

	this->update(updateArea);
}

void SimDispW::drawBlankDisp()
{
	QPainter p;
	backBuf->fill(Qt::black);
	p.begin(backBuf);
	p.setWindow(0,0, 5000, 1024);
	p.setViewport(0, 0, backBuf->width(), backBuf->height());

#if defined EYELID
	p.setPen(Qt::yellow);
	p.drawLine(1500, 0, 1500, 1024);
	p.drawLine(2500, 0, 2500, 1024);
	p.setPen(Qt::red);
	p.drawLine(USONSET, 0, USONSET, 1024);
#endif

	p.scale(backBuf->width()/5000.0, backBuf->height()/1024.0);
	p.end();
	this->update();
}

void SimDispW::drawCartPoleViz(QPainter& p, int t) {
	simMZDispNumLock.lock();
	p.setPen(Qt::red);
	p.drawPoint(t, (int)(512*outputMod[0]->exportOutput()));
	p.setPen(Qt::blue);
	p.drawPoint(t, (int)(512*outputMod[1]->exportOutput()));

        // This is not really necessary given the graphical visualization
	// if (!((CartPole*)externalMod)->isInTimeout())
	// {
        //   int min = 200; 
        //   int max = 400;

        //   // Draw bounds
        //   p.setPen(Qt::yellow);
        //   static bool on = false;
        //   static int cnt = 0;
        //   if (cnt++ % 10 == 0)
        //     on = !on;
        //   if (on) { p.drawPoint(t,min); p.drawPoint(t,max); }

        //   // Draw zero point
        //   p.setPen(Qt::blue);
        //   p.drawPoint(t, (int)((max-min)/2 + min)); 

        //   // Pole angle visualization
        //   p.setPen(Qt::red);
        //   float pole_ang_min = ((CartPole*)externalMod)->getMinPoleAngle();
        //   float pole_ang_max = ((CartPole*)externalMod)->getMaxPoleAngle();
        //   float curr_ang     = ((CartPole*)externalMod)->getPoleAngle();
        //   float ang_perc     = (curr_ang - pole_ang_min) / (pole_ang_max - pole_ang_min);
        //   float ang_remap    = ang_perc * (max-min) + min;
	//   p.drawPoint(t, (int) ang_remap);

        //   // Cart Position Visualization
        //   p.setPen(Qt::yellow);
        //   float cart_min = ((CartPole*)externalMod)->getMinCartPos();
        //   float cart_max = ((CartPole*)externalMod)->getMaxCartPos();
        //   float curr_pos = ((CartPole*)externalMod)->getCartRelativePos();
        //   float perc = (curr_pos - cart_min) / (cart_max - cart_min);
        //   float remapped = perc * (max-min) + min;
        //   p.drawPoint(t, (int) remapped);
	// }
	simMZDispNumLock.unlock();
}

void SimDispW::paintEvent(QPaintEvent *event)
{
	QPainter p(this);
	p.setViewTransformEnabled(true);
	p.setWindow(0, 0, backBuf->width(), backBuf->height());
	p.setViewport(0, 0, this->width(), this->height());
	//p.setWorldMatrixEnabled(false);

	p.drawPixmap(event->rect(), *backBuf, event->rect());
	//cout<<event->rect().x()<<" "<<event->rect().y()<<" "<<event->rect().width()<<" "<<event->rect().height()<<endl;
	//p.drawPixmap(0,0, *backBuf);
}
