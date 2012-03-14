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

	qRegisterMetaType<vector<bool> >("vector<bool>");

	connect(this, SIGNAL(updateSpatialW(vector<bool>, int, bool)),
			spatialView, SLOT(drawActivity(vector<bool>, int, bool)),
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

	bool finished;
	int trialN;

	numGR=management->getNumGR();
	numGO=management->getNumGO();

	apGRVis.resize(numGR);
	apGOVis.resize(numGO);

	finished=false;

	while(!finished)
	{
		lockAccessData();

		finished=management->runStep();

		apGR=management->exportAPGR();
		for(int i=0; i<numGR; i++)
		{
			apGRVis[i]=apGR[i];
		}
		emit(updateSpatialW(apGRVis, 0, true));

		apGO=management->exportAPGO();
		for(int i=0; i<numGO; i++)
		{
			apGOVis[i]=apGO[i];
		}
		emit(updateSpatialW(apGOVis, 1, false));

		unlockAccessData();
	}
}
