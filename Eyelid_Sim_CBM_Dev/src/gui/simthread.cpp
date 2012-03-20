/*
 * simthread.cpp
 *
 *  Created on: Mar 2, 2012
 *      Author: consciousness
 */


#include "../../includes/gui/simthread.h"
#include "../../includes/gui/moc/moc_simthread.h"

using namespace std;

SimThread::SimThread(QObject *parent, ECManagement *ecsim, ActSpatialView *sview)
	: QThread(parent)
{
	management=ecsim;
	spatialView=sview;

	qRegisterMetaType<std::vector<bool> >("std::vector<bool>");

	connect(this, SIGNAL(updateSpatialW(std::vector<bool>, int, bool)),
			spatialView, SLOT(drawActivity(std::vector<bool>, int, bool)),
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

	const bool *apGR;
	const bool *apGO;

	int numGR;
	int numGO;
	int iti;

	numGR=management->getNumGR();
	numGO=management->getNumGO();
	iti=management->getInterTrialI();

	apGRVis.resize(numGR);
	apGOVis.resize(numGO);

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

			cerr<<"run time for trial #"<<currentTrial<<": "<<runLen<<" ms"<<endl;
		}

//		apGR=management->exportAPGR();
//		for(int i=0; i<numGR; i++)
//		{
//			apGRVis[i]=apGR[i];
//		}
//		emit(updateSpatialW(apGRVis, 0, true));
//
//		apGO=management->exportAPGO();
//		for(int i=0; i<numGO; i++)
//		{
//			apGOVis[i]=apGO[i];
//		}
//		emit(updateSpatialW(apGOVis, 1, false));

		unlockAccessData();
	}
}
