#include "../../includes/gui/mainw.h"
#include "../../includes/gui/moc/moc_mainw.h"

using namespace std;

MainW::MainW(QApplication *app, QWidget *parent)
    : QWidget(parent)
{
	ui.setupUi(this);

	this->setAttribute(Qt::WA_DeleteOnClose);

	connect(ui.quitButton, SIGNAL(clicked()), app, SLOT(quit()));
	connect(this, SIGNAL(destroyed()), app, SLOT(quit()));
}

MainW::~MainW()
{
	delete spatialView;
}

void MainW::run()
{
	CBMSimCore *simCore;
	MFPoissonRegen *mf;

	ActSpatialView *panel;

	float *freqs;

	int t;

	vector<int> cellGridX;
	vector<int> cellGridY;
	vector<int> cellSize;
	vector<QColor> cellColor;

	simCore=new CBMSimCore(1);
	mf=new MFPoissonRegen(simCore->getNumMF());
	freqs=new float[simCore->getNumMF()];

	cellGridX.push_back(2048);
	cellGridX.push_back(64);

	cellGridY.push_back(512);
	cellGridY.push_back(16);

	cellSize.push_back(1);
	cellSize.push_back(9);

	cellColor.push_back(Qt::green);
	cellColor.push_back(Qt::red);

	panel=new ActSpatialView(cellGridX, cellGridY, cellSize, cellColor);

	for(int i=0; i<simCore->getNumMF(); i++)
	{
		freqs[i]=5;
	}

	cerr<<"starting run"<<endl;

	panel->show();
	panel->update();

	for(int i=0; i<10; i++)
	{
		t=time(0);
		cerr<<"iteration #"<<i<<": ";
		cerr.flush();
		for(int j=0; j<5000; j++)
		{
			const bool *mfAct;
			mfAct=mf->calcActivity(freqs);

			simCore->updateMFInput(mfAct);
			simCore->updateErrDrive(0, 0);
			simCore->calcActivity();
		}
		cerr<<time(0)-t<<" sec"<<endl;
	}

//	panel->close();

	delete simCore;
	delete mf;
	delete[] freqs;
	delete panel;
}
