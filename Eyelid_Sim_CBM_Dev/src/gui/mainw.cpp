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

	manager=new ECManagementDelay(conPF, actPF, 10, 50000, 20000, 2000, 2500, 2040,
			5, 1000, 1000, 0.025, 0.03, 0.03,
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
//
	inputNetSpatialView=new ActSpatialView(xDims, yDims, sizes, colors, "/mnt/FastData/movie/");

	inputNetTView=new ActTemporalView(conParams->getNumGO(), 1, manager->getInterTrialI(),
			manager->getInterTrialI()/8, conParams->getNumGO(), Qt::white, "inputNet");
	scTView=new ActTemporalView(conParams->getNumSC(), 1, manager->getInterTrialI(),
			manager->getInterTrialI()/8, conParams->getNumSC(), Qt::white, "stellate");
	bcTView=new ActTemporalView(conParams->getNumBC(), 1, manager->getInterTrialI(),
			manager->getInterTrialI()/8, conParams->getNumBC(), Qt::green, "basket");
	pcTView=new ActTemporalView(conParams->getNumPC(), 8, manager->getInterTrialI(),
			manager->getInterTrialI()/8, conParams->getNumPC()*8, Qt::red, "purkinje");
	ncTView=new ActTemporalView(conParams->getNumNC(), 16, manager->getInterTrialI(),
			manager->getInterTrialI()/8, conParams->getNumNC()*16, Qt::green, "nucleus");
	ioTView=new ActTemporalView(conParams->getNumIO(), 32, manager->getInterTrialI(),
			manager->getInterTrialI()/8, conParams->getNumIO()*32, Qt::white, "inferior olive");
//
	compThread=new SimThread(this, manager,
			inputNetSpatialView,
			inputNetTView,
			scTView,
			bcTView,
			pcTView,
			ncTView,
			ioTView);

	inputNetSpatialView->hide();

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
	inputNetSpatialView->show();
	inputNetSpatialView->update();

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
//
	inputNetTView->show();
	inputNetTView->update();

	compThread->start(QThread::TimeCriticalPriority);
}
