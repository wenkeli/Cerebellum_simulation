#include "../../includes/gui/mainw.h"
#include "../../includes/gui/moc/moc_mainw.h"

using namespace std;

MainW::MainW(QApplication *app, QWidget *parent)
    : QWidget(parent)
{
	QStringList args;
	ui.setupUi(this);

	args=app->arguments();

	conPFileName=args[1].toStdString();
	actPFileName=args[2].toStdString();

	this->setAttribute(Qt::WA_DeleteOnClose);

	connect(ui.quitButton, SIGNAL(clicked()), app, SLOT(quit()));
	connect(this, SIGNAL(destroyed()), app, SLOT(quit()));

	cout<<"conPF "<<conPFileName<<endl;
	cout<<"actPF "<<actPFileName<<endl;

	ui.numTrialsBox->setMaximum(1000000000);
	ui.numTrialsBox->setValue(1050);
	ui.itiBox->setMaximum(1000000);
	ui.itiBox->setValue(5000);
	ui.csStartBox->setMaximum(ui.itiBox->maximum());
	ui.csStartBox->setValue(2000);
	ui.csEndBox->setMaximum(ui.itiBox->maximum());
	ui.csEndBox->setValue(2750);
	ui.csPEndBox->setMaximum(ui.itiBox->maximum());
	ui.csPEndBox->setValue(2040);
	ui.csTrialStartNBox->setMaximum(ui.numTrialsBox->maximum());
	ui.csTrialStartNBox->setValue(5);
	ui.dataTrialStartNBox->setMaximum(ui.numTrialsBox->maximum());
	ui.dataTrialStartNBox->setValue(45);
	ui.numDataTrialsBox->setMaximum(ui.numTrialsBox->maximum());
	ui.numDataTrialsBox->setValue(1000);

	ui.fracCtxtMFBox->setMaximum(1);
	ui.fracCtxtMFBox->setDecimals(4);
	ui.fracCtxtMFBox->setValue(0.03);
	ui.mfBGFreqMinBox->setMaximum(1000);
	ui.mfBGFreqMinBox->setDecimals(4);
	ui.mfBGFreqMinBox->setValue(1);
	ui.mfBGFreqMaxBox->setMaximum(1000);
	ui.mfBGFreqMaxBox->setDecimals(4);
	ui.mfBGFreqMaxBox->setValue(10);
	ui.ctxtMFFreqMinBox->setMaximum(1000);
	ui.ctxtMFFreqMinBox->setDecimals(4);
	ui.ctxtMFFreqMinBox->setValue(30);
	ui.ctxtMFFreqMaxBox->setMaximum(1000);
	ui.ctxtMFFreqMaxBox->setDecimals(4);
	ui.ctxtMFFreqMaxBox->setValue(60);

	ui.csMFBGFreqMinBox->setMaximum(1000);
	ui.csMFBGFreqMinBox->setDecimals(4);
	ui.csMFBGFreqMinBox->setValue(1);
	ui.csMFBGFreqMaxBox->setMaximum(1000);
	ui.csMFBGFreqMaxBox->setDecimals(4);
	ui.csMFBGFreqMaxBox->setValue(5);
	ui.fracCSTMFBox->setMaximum(1);
	ui.fracCSTMFBox->setDecimals(4);
	ui.fracCSTMFBox->setValue(0.025);
	ui.csTMFFreqMinBox->setMaximum(1000);
	ui.csTMFFreqMinBox->setDecimals(4);
	ui.csTMFFreqMinBox->setValue(40);
	ui.csTMFFreqMaxBox->setMaximum(1000);
	ui.csTMFFreqMaxBox->setDecimals(4);
	ui.csTMFFreqMaxBox->setValue(50);
	ui.fracCSPMFBox->setMaximum(1);
	ui.fracCSPMFBox->setDecimals(4);
	ui.fracCSPMFBox->setValue(0.0);
	ui.csPMFFreqMinBox->setMaximum(1000);
	ui.csPMFFreqMinBox->setDecimals(4);
	ui.csPMFFreqMinBox->setValue(120);
	ui.csPMFFreqMaxBox->setMaximum(1000);
	ui.csPMFFreqMaxBox->setDecimals(4);
	ui.csPMFFreqMaxBox->setValue(130);

	ui.inputNetCellTBox->setDisabled(true);
	ui.showINetBox->setDisabled(true);
	ui.showINetBox->setChecked(false);
	ui.showINetSpatialBox->setDisabled(true);
	ui.showINetSpatialBox->setChecked(false);
	ui.showBCBox->setDisabled(true);
	ui.showBCBox->setChecked(false);
	ui.showSCBox->setDisabled(true);
	ui.showSCBox->setChecked(false);
	ui.showPCBox->setDisabled(true);
	ui.showPCBox->setChecked(false);
	ui.showIOBox->setDisabled(true);
	ui.showIOBox->setChecked(false);
	ui.showNCBox->setDisabled(true);
	ui.showNCBox->setChecked(false);

	ui.gpuStartNBox->setMinimum(0);
	ui.gpuStartNBox->setMaximum(128);
	ui.gpuStartNBox->setValue(0);

	ui.numGPUP2Box->setMinimum(0);
	ui.numGPUP2Box->setMaximum(8);
	ui.numGPUP2Box->setValue(2);


//	inputNetSpatialView->hide();

//	pcTView->drawBlank(Qt::blue);
//	pcTView->drawVertLine(500, Qt::white);

}

MainW::~MainW()
{
	delete compThread;
	delete manager;
	delete inputNetSpatialView;
	delete inputNetTView;
	delete scTView;
	delete bcTView;
	delete pcTView;
	delete ncTView;
	delete ioTView;
}


void MainW::run()
{
	vector<int> xDims;
	vector<int> yDims;
	vector<int> sizes;
	vector<QColor> colors;

	vector<int> csLineTs;
	vector<QColor> csLineColors;

	QColor temp(255, 165, 0);

	ui.runButton->setDisabled(true);

	manager=new ECManagementDelay(conPFileName, actPFileName, time(0),
			ui.numTrialsBox->value(), ui.itiBox->value(),
			ui.csStartBox->value(), ui.csEndBox->value(), ui.csPEndBox->value(),
			ui.csTrialStartNBox->value(), ui.dataTrialStartNBox->value(), ui.numDataTrialsBox->value(),
			ui.fracCSTMFBox->value(), ui.fracCSPMFBox->value(), ui.fracCtxtMFBox->value(),
			ui.mfBGFreqMinBox->value(), ui.csMFBGFreqMinBox->value(), ui.ctxtMFFreqMinBox->value(),
			ui.csTMFFreqMinBox->value(), ui.csPMFFreqMinBox->value(),
			ui.mfBGFreqMaxBox->value(), ui.csMFBGFreqMaxBox->value(), ui.ctxtMFFreqMaxBox->value(),
			ui.csTMFFreqMaxBox->value(), ui.csPMFFreqMaxBox->value(),
			ui.gpuStartNBox->value(), ui.numGPUP2Box->value());

	conParams=manager->getConParams();

	xDims.push_back(conParams->getGRX());
	xDims.push_back(conParams->getGOX());
	xDims.push_back(conParams->getGLX());

	yDims.push_back(conParams->getGRY());
	yDims.push_back(conParams->getGOY());
	yDims.push_back(conParams->getGLY());

	sizes.push_back(2);
	sizes.push_back(10);
	sizes.push_back(2);

	colors.push_back(Qt::green);
	colors.push_back(temp);
	colors.push_back(Qt::blue);

	for(int i=0; i<7; i++)
	{
		csLineTs.push_back(2000+i*250);
		csLineColors.push_back(Qt::yellow);
	}
//
	inputNetSpatialView=new ActSpatialView(xDims, yDims, sizes, colors, "/mnt/FastData/movie/");

	inputNetTView=new ActTemporalView(conParams->getNumGO(), 1, manager->getInterTrialI(),
			2000, conParams->getNumGO(),
			csLineTs, csLineColors, Qt::white, "inputNet");
	scTView=new ActTemporalView(conParams->getNumSC(), 1, manager->getInterTrialI(),
			2000, conParams->getNumSC(),
			csLineTs, csLineColors, Qt::white, "stellate");
	bcTView=new ActTemporalView(conParams->getNumBC(), 1, manager->getInterTrialI(),
			2000, conParams->getNumBC(),
			csLineTs, csLineColors, Qt::green, "basket");
	pcTView=new ActTemporalView(conParams->getNumPC(), 8, manager->getInterTrialI(),
			2000, conParams->getNumPC()*8,
			csLineTs, csLineColors, Qt::red, "purkinje");
	ncTView=new ActTemporalView(conParams->getNumNC(), 16, manager->getInterTrialI(),
			2000, conParams->getNumNC()*16,
			csLineTs, csLineColors, Qt::green, "nucleus");
	ioTView=new ActTemporalView(conParams->getNumIO(), 32, manager->getInterTrialI(),
			2000, conParams->getNumIO()*32,
			csLineTs, csLineColors, Qt::white, "inferior olive");
//
	itc=new InterThreadComm();

	compThread=new SimThread(this, manager,
			inputNetSpatialView,
			inputNetTView,
			scTView,
			bcTView,
			pcTView,
			ncTView,
			ioTView,
			itc);

	inputNetTView->hide();
	inputNetSpatialView->hide();
	scTView->hide();
	pcTView->hide();
	bcTView->hide();
	ncTView->hide();
	ioTView->hide();

	ui.numTrialsBox->setDisabled(true);
	ui.itiBox->setDisabled(true);
	ui.csStartBox->setDisabled(true);
	ui.csEndBox->setDisabled(true);
	ui.csPEndBox->setDisabled(true);
	ui.csTrialStartNBox->setDisabled(true);
	ui.dataTrialStartNBox->setDisabled(true);
	ui.numDataTrialsBox->setDisabled(true);

	ui.fracCtxtMFBox->setDisabled(true);
	ui.mfBGFreqMinBox->setDisabled(true);
	ui.mfBGFreqMaxBox->setDisabled(true);
	ui.ctxtMFFreqMinBox->setDisabled(true);
	ui.ctxtMFFreqMaxBox->setDisabled(true);

	ui.csMFBGFreqMinBox->setDisabled(true);
	ui.csMFBGFreqMaxBox->setDisabled(true);
	ui.fracCSTMFBox->setDisabled(true);
	ui.csTMFFreqMinBox->setDisabled(true);
	ui.csTMFFreqMaxBox->setDisabled(true);
	ui.fracCSPMFBox->setDisabled(true);
	ui.csPMFFreqMinBox->setDisabled(true);
	ui.csPMFFreqMaxBox->setDisabled(true);

	ui.inputNetCellTBox->setEnabled(true);
	ui.showINetBox->setEnabled(true);
	ui.showINetSpatialBox->setEnabled(true);
	ui.showBCBox->setEnabled(true);
	ui.showSCBox->setEnabled(true);
	ui.showPCBox->setEnabled(true);
	ui.showIOBox->setEnabled(true);
	ui.showNCBox->setEnabled(true);

	compThread->start(QThread::TimeCriticalPriority);
}

void MainW::updateInNetCellT(int cellT)
{
	itc->accessDispParamLock.lock();
	itc->inNetDispCellT=cellT;
	itc->accessDispParamLock.unlock();
}

void MainW::showINetAct(int checked)
{
	showActCommon(checked, 0, inputNetTView);
}

void MainW::showINetSpatial(int checked)
{
	showActCommon(checked, 1, inputNetSpatialView);
}

void MainW::showBCAct(int checked)
{
	showActCommon(checked, 2, bcTView);
}

void MainW::showSCAct(int checked)
{
	showActCommon(checked, 3, scTView);
}

void MainW::showPCAct(int checked)
{
	showActCommon(checked, 4, pcTView);
}

void MainW::showIOAct(int checked)
{
	showActCommon(checked, 5, ioTView);
}

void MainW::showNCAct(int checked)
{
	showActCommon(checked, 6, ncTView);
}

void MainW::showActCommon(int checked, int checkedIndex, QWidget *view)
{
	if(checked==Qt::Unchecked)
	{
		view->hide();

		itc->accessDispParamLock.lock();
		itc->showActPanels[checkedIndex]=false;
		itc->accessDispParamLock.unlock();
	}
	else
	{
		view->show();

		itc->accessDispParamLock.lock();
		itc->showActPanels[checkedIndex]=true;
		itc->accessDispParamLock.unlock();
	}
}

