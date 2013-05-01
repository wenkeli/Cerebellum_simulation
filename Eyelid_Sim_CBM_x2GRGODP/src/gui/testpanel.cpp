#include "../../includes/gui/testpanel.h"
#include "../../includes/gui/moc/moc_testpanel.h"

TestPanel::TestPanel(QWidget *parent)
    : QWidget(parent)
{
	ui.setupUi(this);

	this->setFixedSize(300, 300);
	this->setAutoFillBackground(true);
}

TestPanel::~TestPanel()
{

}
