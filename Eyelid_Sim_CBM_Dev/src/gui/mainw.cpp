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

	this->setAttribute(Qt::WA_DeleteOnClose);

	connect(ui.quitButton, SIGNAL(clicked()), app, SLOT(quit()));
	connect(this, SIGNAL(destroyed()), app, SLOT(quit()));

	manager=new ECManagement(1000, 5000);

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
	pcTView=new ActTemporalView(manager->getNumPC(), 24, manager->getInterTrialI(),
			manager->getInterTrialI()/2, manager->getNumPC()*24, Qt::red);
	ncTView=new ActTemporalView(manager->getNumNC(), 24, manager->getInterTrialI(),
			manager->getInterTrialI()/2, manager->getNumNC()*24, Qt::green);
	ioTView=new ActTemporalView(manager->getNumIO(), 64, manager->getInterTrialI(),
			manager->getInterTrialI()/2, manager->getNumIO()*64, Qt::white);
//
	compThread=new SimThread(this, manager, inputNetSpatialView, pcTView);
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
	inputNetSpatialView->show();
	inputNetSpatialView->update();

	pcTView->show();
	pcTView->update();

	compThread->start(QThread::TimeCriticalPriority);
}
