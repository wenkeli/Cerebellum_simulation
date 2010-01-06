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
		wH=ALLVIEWPH;
		windowTitle<<"all "<<cellTypeText[cellT]<<"s PSH starting at cell #"<<cN<<"(GR only)";

	}
	else
	{
		wW=SINGLEVIEWPW;
		wH=SINGLEVIEWPH;
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
	cout<<"type: "<<type<<" cellT:"<<cellT<<" cellN"<<cellN<<endl;
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
