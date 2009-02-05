#include "../includes/mfgrmainp.h"
#include "../includes/moc_mfgrmainp.h"
#include "../includes/mfgrconnsw.h"

MFGRmainP::MFGRmainP(QWidget *parent)
    : QWidget(parent)
{
	ui.setupUi(this);
	ui.mfNumSel->setRange(0,NUMMF-1);
	this->setWindowTitle("display Mossy fiber to granule cell connections");
}

MFGRmainP::~MFGRmainP()
{

}

void MFGRmainP::drawMFGRConns()
{
	MFGRConnsW *connW=new MFGRConnsW();

	if(!connsMade)
	{
		ui.statusBox->textCursor().insertText("Error: connections not initialized. Can not display.\n");
		ui.statusBox->repaint();
		return;
	}

	dispMFNum=ui.mfNumSel->value();
	if(dispMFNum>NUMMF-1 || dispMFNum<0)
	{
		ui.statusBox->textCursor().insertText("Error: Mossy fiber # must be between 0 and 1023\n");
		ui.statusBox->repaint();
		return;
	}
	ui.statusBox->textCursor().insertText("drawing...\n");
	ui.statusBox->repaint();

	connW->show();

	ui.statusBox->textCursor().insertText("done!\n");
	ui.statusBox->repaint();
}
