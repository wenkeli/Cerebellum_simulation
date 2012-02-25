#include "../../includes/gui/mainw.h"
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
	CBMSimCore *simCore1;
	MFPoissonRegen *mf1;

	ActSpatialView *panel;

	float *freqs;

	int t;

	vector<int> cellGridX;
	vector<int> cellGridY;
	vector<int> cellSize;
	vector<int>

//	simCore1=new CBMSimCore(1);
//	mf1=new MFPoissonRegen(simCore1->getNumMF());
//	freqs=new float[simCore->getNumMF()];



	for(int i=0; i<simCore->getNumMF(); i++)
	{
		freqs[i]=5;
	}

	cerr<<"starting run"<<endl;

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

	delete simCore;
	delete mf;
	delete[] freqs;
}
