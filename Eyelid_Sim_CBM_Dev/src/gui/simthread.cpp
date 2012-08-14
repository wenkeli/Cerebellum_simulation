/*
 * simthread.cpp
 *
 *  Created on: Mar 2, 2012
 *      Author: consciousness
 */


#include "../../includes/gui/simthread.h"
#include "../../includes/gui/moc/moc_simthread.h"

using namespace std;

SimThread::SimThread(QObject *parent, ECManagement *ecsim,
		ActSpatialView *sview, ActTemporalView *pcTV)
	: QThread(parent)
{
	management=ecsim;
	spatialView=sview;
	pcTView=pcTV;

	qRegisterMetaType<std::vector<bool> >("std::vector<bool>");
	qRegisterMetaType<std::vector<float> >("std::vector<float>");
	qRegisterMetaType<QColor>("QColor");

	connect(this, SIGNAL(updateSpatialW(std::vector<bool>, int, bool)),
			spatialView, SLOT(drawActivity(std::vector<bool>, int, bool)),
			Qt::QueuedConnection);
	connect(this, SIGNAL(updatePCTW(std::vector<bool>, std::vector<float>, int)),
			pcTView, SLOT(drawVmRaster(std::vector<bool>, std::vector<float>, int)),
			Qt::QueuedConnection);
	connect(this, SIGNAL(blankPCTW(QColor)), pcTView, SLOT(drawBlank(QColor)),
			Qt::QueuedConnection);
}


SimThread::~SimThread()
{
}


void SimThread::run()
{
	simLoop();
}

void SimThread::lockAccessData()
{
	accessDataLock.lock();
}

void SimThread::unlockAccessData()
{
	accessDataLock.unlock();
}

void SimThread::simLoop()
{
	vector<bool> apGRVis;
	vector<bool> apGOVis;
	vector<bool> apGLVis;

	vector<bool> apPCVis;
	vector<float> vmPCVis;

	const bool *apGR;
	const bool *apGO;
	const bool *apGL;
	const bool *apPC;
	const float *vmPC;

	int numGR;
	int numGO;
	int numGL;
	int numPC;
	int iti;

	numGR=management->getNumGR();
	numGO=management->getNumGO();
	numGL=management->getNumGL();
	numPC=management->getNumPC();
	iti=management->getInterTrialI();

	apGRVis.resize(numGR);
	apGOVis.resize(numGO);
	apGLVis.resize(numGL);
	apPCVis.resize(numPC);
	vmPCVis.resize(numPC);

	timer.start();

	while(true)
	{
		int runLen;
		int currentTrial;
		int currentTime;
		bool notDone;

		lockAccessData();

		notDone=management->runStep();
		if(!notDone)
		{
			break;
		}
		currentTime=management->getCurrentTime();
		if(currentTime>=iti)
		{
			runLen=timer.restart();
			currentTrial=management->getCurrentTrialN();

			emit(blankPCTW(Qt::black));

			cerr<<"run time for trial #"<<currentTrial<<": "<<runLen<<" ms"<<endl;
		}

//		apGR=management->exportAPGR();
//		for(int i=0; i<numGR; i++)
//		{
//			apGRVis[i]=apGR[i];
//		}
//		emit(updateSpatialW(apGRVis, 0, true));

		apGO=management->exportAPGO();
		for(int i=0; i<numGO; i++)
		{
			apGOVis[i]=apGO[i];
		}
		emit(updateSpatialW(apGOVis, 1, true));

//		apGL=management->exportAPGL();
//		for(int i=0; i<numGL; i++)
//		{
//			apGLVis[i]=apGL[i];
//		}
//		emit(updateSpatialW(apGLVis, 2, false));

		apPC=management->exportAPPC();
		vmPC=management->exportVmPC();
		for(int i=0; i<numPC; i++)
		{
			apPCVis[i]=apPC[i];
			vmPCVis[i]=(vmPC[i]+80)/80;
		}
		emit(updatePCTW(apPCVis, vmPCVis, currentTime));

		unlockAccessData();
	}
}
