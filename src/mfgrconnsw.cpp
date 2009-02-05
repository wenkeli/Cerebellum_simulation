#include "../includes/mfgrconnsw.h"
#include "../includes/moc_mfgrconnsw.h"

MFGRConnsW::MFGRConnsW(QWidget *parent)
    : QWidget(parent)
{
	ui.setupUi(this);
	resize(GRX,GRY);
}

MFGRConnsW::~MFGRConnsW()
{

}

void MFGRConnsW::paintEvent(QPaintEvent* event)
{
	QPainter painter(this);

	painter.setPen(Qt::white);

	if(dispMFNum>NUMMF-1 || dispMFNum<0)
	{
		this->close();
		return;
	}
	if(!connsMade)
	{
		this->close();
		return;
	}

	for(int i=0; i<MFGRSYNPERMF; i++)
	{
		int grNum=conMFtoGR[dispMFNum][i][0];
		int x=grNum%GRX;
		int y=grNum/GRX;
		if(y>=GRY)
		{
			this->close();
			return;
		}
		painter.drawPoint(x, y);
	}
}
