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

	bool notDone;
	int trialN;
	int count;
	count=0;

	numGR=management->getNumGR();
	numGO=management->getNumGO();

	apGRVis.resize(numGR);
	apGOVis.resize(numGO);

	notDone=true;

	while(true)
	{
		lockAccessData();

		notDone=management->runStep();
		if(!notDone)
		{
			break;
		}

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
//		cout<<"here"<<endl;

//		cout<<"iteration: "<<count<<endl;
//		count++;

		unlockAccessData();
	}
}
