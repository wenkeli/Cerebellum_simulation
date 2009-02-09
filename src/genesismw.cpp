#include "../includes/genesismw.h"
#include "../includes/moc_genesismw.h"
#include "../includes/dispdialoguep.h"

GenesisMW::GenesisMW(QWidget *parent)
    : QMainWindow(parent)
{
	ui.setupUi(this);
	this->setWindowTitle("Cerebellum connections");
}

GenesisMW::~GenesisMW()
{

}

QTextBrowser *GenesisMW::getStatusBox()
{
	return ui.statusBox;
}

void GenesisMW::setApp(QApplication *a)
{
	app=a;
	connect(ui.quitButton, SIGNAL(clicked()), app, SLOT(quit()));
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

	//call display main panel (DispDialogueP)
	DispDialogueP *panel=new DispDialogueP();
	panel->setDispT(MFGR);
	panel->show();
	//(DispDialogue calls ConnDispW)
}

void GenesisMW::showMFGOMainP()
{
	if(!connsMade)
	{
		ui.statusBox->textCursor().insertText("connections not initialized. Please run \"make connections\" to initialize.\n");
		return;
	}

	//call display main panel (DispDialogueP)
	DispDialogueP *panel=new DispDialogueP();
	panel->setDispT(MFGO);
	panel->show();
}

void GenesisMW::showGRGOMainP()
{
	if(!connsMade)
	{
		ui.statusBox->textCursor().insertText("connections not initialized. Please run \"make connections\" to initialize.\n");
		return;
	}

	//call display main panel (DispDialogueP)
	DispDialogueP *panel=new DispDialogueP();
	panel->setDispT(GRGO);
	panel->show();
}

void GenesisMW::showGOGRMainP()
{
	if(!connsMade)
	{
		ui.statusBox->textCursor().insertText("connections not initialized. Please run \"make connections\" to initialize.\n");
		return;
	}

	//call display main panel (DispDialogueP)
	DispDialogueP *panel=new DispDialogueP();
	panel->setDispT(GOGR);
	panel->show();
}

