#include "../includes/pshdispw.h"
#include "../includes/moc_pshdispw.h"

PSHDispw::PSHDispw(QWidget *parent, QPixmap *buf, QString wt)
    : QWidget(parent)
{
	ui.setupUi(this);

	this->setAttribute(Qt::WA_DeleteOnClose);

	this->setWindowTitle(wt);

	this->setFixedSize(buf->width(), buf->height());
	this->setAutoFillBackground(true);

	backBuf=buf;

	this->show();
	this->update();
}

void PSHDispw::switchBuf(QPixmap *newBuf)
{
	delete backBuf;
	backBuf=newBuf;
	this->setFixedSize(newBuf->width(), newBuf->height());
	this->update();
}

PSHDispw::~PSHDispw()
{
	delete backBuf;
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
