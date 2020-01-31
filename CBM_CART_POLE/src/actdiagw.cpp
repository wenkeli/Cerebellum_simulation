#include "../includes/actdiagw.h"
#include "../includes/moc_actdiagw.h"

ActDiagW::ActDiagW(QWidget *parent)
    : QWidget(parent)
{
	ui.setupUi(this);
	this->setAttribute(Qt::WA_DeleteOnClose);
	this->setFixedSize(GRX, GRY);
	backBuf=new QPixmap(this->width(), this->height());
	backBuf->fill(Qt::black);

	scaleX=(float)GOX/GRX;
	scaleY=(float)GOY/GRY;
}

ActDiagW::~ActDiagW()
{
	delete backBuf;
}

void ActDiagW::drawActivity(vector<unsigned char> grAPs, vector<bool> goAPs)
{
	QPainter p;
	backBuf->fill(Qt::black);

	p.begin(backBuf);
	p.setPen(Qt::green);
	for(int i=0; i<NUMGR; i++)
	{
		if(grAPs[i]&0x01)
		{
			p.drawPoint(i%GRX, i/GRX);
		}
	}

	for(int i=0; i<NUMGO; i++)
	{
		if(goAPs[i])
		{
			p.fillRect((int)(((i%GOX)+0.5)/scaleX-1), (int)((((int)i/GOX)+0.5)/scaleY-1), 3, 3, Qt::red);
		}
	}
	p.end();
	this->update();
}

void ActDiagW::paintEvent(QPaintEvent *event)
{
	QPainter p(this);

	p.drawPixmap(event->rect(), *backBuf, event->rect());
}
