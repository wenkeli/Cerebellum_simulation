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

	ui.setupUi(this);

	dataFileOut.open("dataout", ios::binary);

	this->setAttribute(Qt::WA_DeleteOnClose);

	connect(ui.quitButton, SIGNAL(clicked()), app, SLOT(quit()));
	connect(this, SIGNAL(destroyed()), app, SLOT(quit()));

//	manager=new ECManagementBase(10000, 5000);
	manager=new ECManagementDelay(&dataFileOut, 2000, 5000, 2000, 2750, 2040, 10, 1000, 1000, 0.01, 0.012, 0.03,
			1, 1, 30, 40, 120, 10, 5, 60, 50, 130);


	xDims.push_back(manager->getGRX());
	xDims.push_back(manager->getGOX());
	xDims.push_back(manager->getGLX());

	yDims.push_back(manager->getGRY());
	yDims.push_back(manager->getGOY());
	yDims.push_back(manager->getGLY());

	sizes.push_back(1);
	sizes.push_back(5);
	sizes.push_back(2);

	colors.push_back(Qt::green);
	colors.push_back(Qt::red);
	colors.push_back(Qt::blue);
//
	inputNetSpatialView=new ActSpatialView(xDims, yDims, sizes, colors);

	inputNetTView=new ActTemporalView(manager->getNumGO(), 1, manager->getInterTrialI(),
			manager->getInterTrialI()/2, manager->getNumGO(), Qt::white);
	scTView=new ActTemporalView(manager->getNumSC(), 1, manager->getInterTrialI(),
			manager->getInterTrialI()/2, manager->getNumSC(), Qt::white);
	bcTView=new ActTemporalView(manager->getNumBC(), 1, manager->getInterTrialI(),
			manager->getInterTrialI()/2, manager->getNumBC(), Qt::green);
	pcTView=new ActTemporalView(manager->getNumPC(), 8, manager->getInterTrialI(),
			manager->getInterTrialI()/2, manager->getNumPC()*8, Qt::red);
	ncTView=new ActTemporalView(manager->getNumNC(), 16, manager->getInterTrialI(),
			manager->getInterTrialI()/2, manager->getNumNC()*16, Qt::green);
	ioTView=new ActTemporalView(manager->getNumIO(), 32, manager->getInterTrialI(),
			manager->getInterTrialI()/2, manager->getNumIO()*32, Qt::white);
//
	compThread=new SimThread(this, manager,
			inputNetSpatialView,
			inputNetTView,
			scTView,
			bcTView,
			pcTView,
			ncTView,
			ioTView);

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
//	inputNetSpatialView->show();
//	inputNetSpatialView->update();

	bcTView->show();
	bcTView->update();

	scTView->show();
	scTView->update();

	pcTView->show();
	pcTView->update();

	ncTView->show();
	ncTView->update();

	ioTView->show();
	ioTView->update();

	inputNetTView->show();
	inputNetTView->update();

	compThread->start(QThread::TimeCriticalPriority);
}
