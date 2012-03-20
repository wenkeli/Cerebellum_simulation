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

	manager=new ECManagement(10, 5000);

	xDims.push_back(manager->getGRX());
	xDims.push_back(manager->getGOX());

	yDims.push_back(manager->getGRY());
	yDims.push_back(manager->getGOY());

	sizes.push_back(1);
	sizes.push_back(9);

	colors.push_back(Qt::green);
	colors.push_back(Qt::red);
//
	spatialView=new ActSpatialView(xDims, yDims, sizes, colors);
//
	compThread=new SimThread(this, manager, spatialView);
}

MainW::~MainW()
{
	delete compThread;
	delete manager;
	delete spatialView;
}


void MainW::run()
{
	spatialView->show();
	spatialView->update();

	compThread->start(QThread::TimeCriticalPriority);
}
