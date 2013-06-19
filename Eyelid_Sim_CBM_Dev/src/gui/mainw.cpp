#include "../../includes/gui/mainw.h"
#include "../../includes/gui/moc/moc_mainw.h"

using namespace std;

MainW::MainW(QApplication *app, QWidget *parent)
    : QWidget(parent)
{
	vector<int> xDims;
	vector<int> yDims;
	vector<int> sizes;
	vector<QColor> colors;

	vector<int> csLineTs;
	vector<QColor> csLineColors;

	QColor temp(255, 165, 0);

	ui.setupUi(this);

	string conPF;
	string actPF;
	QStringList args;

	args=app->arguments();

	conPF=args[1].toStdString();
	actPF=args[2].toStdString();

	this->setAttribute(Qt::WA_DeleteOnClose);

	connect(ui.quitButton, SIGNAL(clicked()), app, SLOT(quit()));
	connect(this, SIGNAL(destroyed()), app, SLOT(quit()));

	cout<<"conPF "<<conPF<<endl;
	cout<<"actPF "<<actPF<<endl;

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
	ui.fracCtxtMFBox->setValue(0.03);
	ui.fracCtxtMFBox->setDecimals(4);
	ui.mfBGFreqMinBox->setMaximum(1000);
	ui.mfBGFreqMinBox->setValue(1);
	ui.mfBGFreqMinBox->setDecimals(4);
	ui.mfBGFreqMaxBox->setMaximum(1000);
	ui.mfBGFreqMaxBox->setValue(10);
	ui.mfBGFreqMaxBox->setDecimals(4);
	ui.ctxtMFFreqMinBox->setMaximum(1000);
	ui.ctxtMFFreqMinBox->setValue(30);
	ui.ctxtMFFreqMinBox->setDecimals(4);
	ui.ctxtMFFreqMaxBox->setMaximum(1000);
	ui.ctxtMFFreqMaxBox->setValue(60);
	ui.ctxtMFFreqMaxBox->setDecimals(4);

	ui.csMFBGFreqMinBox->setMaximum(1000);
	ui.csMFBGFreqMinBox->setValue(1);
	ui.csMFBGFreqMinBox->setDecimals(4);
	ui.csMFBGFreqMaxBox->setMaximum(1000);
	ui.csMFBGFreqMaxBox->setValue(5);
	ui.csMFBGFreqMaxBox->setDecimals(4);
	ui.fracCSTMFBox->setMaximum(1);
	ui.fracCSTMFBox->setValue(0.025);
	ui.fracCSTMFBox->setDecimals(4);
	ui.csTMFFreqMinBox->setMaximum(1000);
	ui.csTMFFreqMinBox->setValue(40);
	ui.csTMFFreqMinBox->setDecimals(4);
	ui.csTMFFreqMaxBox->setMaximum(1000);
	ui.csTMFFreqMaxBox->setValue(50);
	ui.csTMFFreqMaxBox->setDecimals(4);
	ui.fracCSPMFBox->setMaximum(1);
	ui.fracCSPMFBox->setValue(0.0);
	ui.fracCSPMFBox->setDecimals(4);
	ui.csPMFFreqMinBox->setMaximum(1000);
	ui.csPMFFreqMinBox->setValue(120);
	ui.csPMFFreqMinBox->setDecimals(4);
	ui.csPMFFreqMaxBox->setMaximum(1000);
	ui.csPMFFreqMaxBox->setValue(130);
	ui.csPMFFreqMaxBox->setDecimals(4);

	manager=new ECManagementDelay(conPF, actPF, time(0), 12550, 10000, 2000, 3500, 2040,
			5, 11545, 1000, 0.0075, 0.0, 0.015,
			1, 1, 30, 40, 120, 10, 5, 60, 50, 130);

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


//	inputNetSpatialView->hide();

//	pcTView->drawBlank(Qt::blue);
//	pcTView->drawVertLine(500, Qt::white);

	ui.showINetBox->setChecked(false);
	ui.showINetSpatialBox->setChecked(false);
	ui.showBCBox->setChecked(false);
	ui.showSCBox->setChecked(false);
	ui.showPCBox->setChecked(false);
	ui.showIOBox->setChecked(false);
	ui.showNCBox->setChecked(false);
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

