#include "../includes/genesismw.h"
#include "../includes/moc_genesismw.h"

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
