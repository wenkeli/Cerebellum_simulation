#include "../../includes/gui/mainw.h"
#include "../../includes/gui/moc_mainw.h"

MainW::MainW(QWidget *parent, QApplication *a)
    : QMainWindow(parent)
{
	this->setAttribute(Qt::WA_DeleteOnClose); //when the window is closed, delete the object
	if(a==NULL)
	{
		return;
	}
	ui.setupUi(this);
	this->setWindowTitle("Cerebellum simulation");
	app=a;
	connect(ui.quitButton, SIGNAL(clicked()), app, SLOT(quit()));
	connect(this, SIGNAL(destroyed()), app, SLOT(quit()));

	ui.viewMFGRButton->setEnabled(false);
	ui.viewMFGOButton->setEnabled(false);
	ui.viewGRGOButton->setEnabled(false);
	ui.viewGOGRButton->setEnabled(false);
	ui.runSimButton->setEnabled(false);
}

MainW::~MainW()
{

}

QTextBrowser *MainW::getStatusBox()
{
	return ui.statusBox;
}

void MainW::makeConns()
{
	ui.connectButton->setEnabled(false);
	ui.loadSimButton->setEnabled(false);
	newSim();
	initialized=true;

	ui.viewMFGRButton->setEnabled(true);
	ui.viewMFGOButton->setEnabled(true);
	ui.viewGRGOButton->setEnabled(true);
	ui.viewGOGRButton->setEnabled(true);
	ui.runSimButton->setEnabled(true);
}

void MainW::loadSim()
{
	stringstream dummy;
	QString fileName;

	fileName=QFileDialog::getOpenFileName(this, "Please select the sim state file to open", "/", "");
	simIn.open(fileName.toStdString().c_str(), ios::binary);
	if(!simIn.good() || !simIn.is_open())
	{
		cerr<<"error opening file "<<fileName.toStdString()<<endl;
		return;
	}

	ui.connectButton->setEnabled(false);
	ui.loadSimButton->setEnabled(false);

	readSimIn(simIn);

	simIn.close();
	initialized=true;

	ui.viewMFGRButton->setEnabled(true);
	ui.viewMFGOButton->setEnabled(true);
	ui.viewGRGOButton->setEnabled(true);
	ui.viewGOGRButton->setEnabled(true);
	ui.runSimButton->setEnabled(true);
}

void MainW::showMFGRMainP()
{
//	if(!initialized)
//	{
//		ui.statusBox->textCursor().insertText("connections not initialized. Please run \"make connections\" to initialize.\n");
//		return;
//	}
//
//	//call display main panel (DispDialogueP)
//	DispDialogueP *panel=new DispDialogueP(NULL, MFGR);
//	panel->show();
//	//(DispDialogue calls ConnDispW)
}

void MainW::showMFGOMainP()
{
//	if(!initialized)
//	{
//		ui.statusBox->textCursor().insertText("connections not initialized. Please run \"make connections\" to initialize.\n");
//		return;
//	}
//
//	//call display main panel (DispDialogueP)
//	DispDialogueP *panel=new DispDialogueP(NULL, MFGO);
//	//panel->setDispT(MFGO);
//	panel->show();
}

void MainW::showGRGOMainP()
{
//	if(!initialized)
//	{
//		ui.statusBox->textCursor().insertText("connections not initialized. Please run \"make connections\" to initialize.\n");
//		return;
//	}
//
//	//call display main panel (DispDialogueP)
//	DispDialogueP *panel=new DispDialogueP(NULL, GRGO);
//	//panel->setDispT(GRGO);
//	panel->show();
}

void MainW::showGOGRMainP()
{
//	if(!initialized)
//	{
//		ui.statusBox->textCursor().insertText("connections not initialized. Please run \"make connections\" to initialize.\n");
//		return;
//	}
//
//	//call display main panel (DispDialogueP)
//	DispDialogueP *panel=new DispDialogueP(NULL, GOGR);
//	//panel->setDispT(GOGR);
//	panel->show();
}

void MainW::runSimulation()
{
	SimCtrlP *simPanel;

	if(!initialized)
	{
		ui.statusBox->textCursor().insertText("connections not initialized. Please run \"make connections\" to initialize.\n");
		return;
	}
	ui.runSimButton->setDisabled(true);

	simPanel=new SimCtrlP(NULL);
	simPanel->show();
}
