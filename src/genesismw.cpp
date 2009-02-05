#include "../includes/genesismw.h"
#include "../includes/moc_genesismw.h"
#include "../includes/mfgrmainp.h"

GenesisMW::GenesisMW(QWidget *parent)
    : QMainWindow(parent)
{
	ui.setupUi(this);
}

GenesisMW::~GenesisMW()
{

}

QTextBrowser *GenesisMW::getStatusBox()
{
	return ui.statusBox;
}

void GenesisMW::makeConns()
{
	genesis(ui.statusBox);
}

void GenesisMW::showMFGRMainP()
{
	if(!connsMade)
	{
		ui.statusBox->textCursor().insertText("connections not initialized. Please run \"make connections\" to initialize.\n");
		return;
	}
	MFGRmainP *mainP=new MFGRmainP();
	mainP->show();
}
