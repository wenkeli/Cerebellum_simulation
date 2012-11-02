/*
 * simthread.cpp
 *
 *  Created on: Mar 2, 2012
 *      Author: consciousness
 */


#include "../../includes/gui/simthread.h"
#include "../../includes/gui/moc/moc_simthread.h"

using namespace std;

SimThread::SimThread(QObject *parent, ECManagementBase *ecsim,
		ActSpatialView *inputNetSV,
		ActTemporalView *inputNetTV,
		ActTemporalView *scTV,
		ActTemporalView *bcTV,
		ActTemporalView *pcTV,
		ActTemporalView *ncTV,
		ActTemporalView *ioTV)
	: QThread(parent)
{
	management=ecsim;
	inputNetSView=inputNetSV;

	inputNetTView=inputNetTV;
	scTView=scTV;
	bcTView=bcTV;
	pcTView=pcTV;
	ioTView=ioTV;
	ncTView=ncTV;

	qRegisterMetaType<std::vector<bool> >("std::vector<bool>");
	qRegisterMetaType<std::vector<float> >("std::vector<float>");
	qRegisterMetaType<QColor>("QColor");

	connect(this, SIGNAL(updateSpatialW(std::vector<bool>, int, bool)),
			inputNetSView, SLOT(drawActivity(std::vector<bool>, int, bool)),
			Qt::QueuedConnection);
	connect(this, SIGNAL(spatialFrameDump()),
			inputNetSView, SLOT(saveBuf()),
			Qt::QueuedConnection);


	connect(this, SIGNAL(blankTW(QColor)), inputNetTView, SLOT(drawBlank(QColor)),
			Qt::QueuedConnection);
	connect(this, SIGNAL(blankTW(QColor)), scTView, SLOT(drawBlank(QColor)),
				Qt::QueuedConnection);
	connect(this, SIGNAL(blankTW(QColor)), bcTView, SLOT(drawBlank(QColor)),
					Qt::QueuedConnection);
	connect(this, SIGNAL(blankTW(QColor)), pcTView, SLOT(drawBlank(QColor)),
					Qt::QueuedConnection);
	connect(this, SIGNAL(blankTW(QColor)), ncTView, SLOT(drawBlank(QColor)),
					Qt::QueuedConnection);
	connect(this, SIGNAL(blankTW(QColor)), ioTView, SLOT(drawBlank(QColor)),
					Qt::QueuedConnection);

	connect(this, SIGNAL(updateINTW(std::vector<bool>, int)),
			inputNetTView, SLOT(drawRaster(std::vector<bool>, int)),
			Qt::QueuedConnection);
	connect(this, SIGNAL(updateBCTW(std::vector<bool>, int)),
			bcTView, SLOT(drawRaster(std::vector<bool>, int)),
			Qt::QueuedConnection);
	connect(this, SIGNAL(updateSCTW(std::vector<bool>, int)),
			scTView, SLOT(drawRaster(std::vector<bool>, int)),
			Qt::QueuedConnection);
	connect(this, SIGNAL(updatePCTW(std::vector<bool>, std::vector<float>, int)),
			pcTView, SLOT(drawVmRaster(std::vector<bool>, std::vector<float>, int)),
			Qt::QueuedConnection);
	connect(this, SIGNAL(updateNCTW(std::vector<bool>, std::vector<float>, int)),
			ncTView, SLOT(drawVmRaster(std::vector<bool>, std::vector<float>, int)),
			Qt::QueuedConnection);
	connect(this, SIGNAL(updateIOTW(std::vector<bool>, std::vector<float>, int)),
			ioTView, SLOT(drawVmRaster(std::vector<bool>, std::vector<float>, int)),
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
	vector<bool> apMFVis;

	vector<bool> apSCVis;
	vector<bool> apBCVis;

	vector<bool> apPCVis;
	vector<float> vmPCVis;
	vector<bool> apNCVis;
	vector<float> vmNCVis;
	vector<bool> apIOVis;
	vector<float> vmIOVis;

	const bool *apGR;
	const bool *apGO;
	const bool *apGL;
	const bool *apMF;

	const bool *apSC;
	const bool *apBC;

	const bool *apPC;
	const float *vmPC;
	const bool *apNC;
	const float *vmNC;
	const bool *apIO;
	const float *vmIO;

	int numGR;
	int numGO;
	int numGL;
	int numMF;
	int numSC;
	int numBC;
	int numPC;
	int numNC;
	int numIO;
	int iti;

	numGR=management->getNumGR();
	numGO=management->getNumGO();
	numGL=management->getNumGL();
	numMF=management->getNumMF();

	numSC=management->getNumSC();
	numBC=management->getNumBC();
	numPC=management->getNumPC();
	numNC=management->getNumNC();
	numIO=management->getNumIO();
	iti=management->getInterTrialI();

	apGRVis.resize(numGR);
	apGOVis.resize(numGO);
	apGLVis.resize(numGL);
	apMFVis.resize(numMF);

	apSCVis.resize(numSC);
	apBCVis.resize(numBC);
	apPCVis.resize(numPC);
	vmPCVis.resize(numPC);
	apNCVis.resize(numNC);
	vmNCVis.resize(numNC);
	apIOVis.resize(numIO);
	vmIOVis.resize(numIO);

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

			emit(blankTW(Qt::black));

			cerr<<"run time for trial #"<<currentTrial<<": "<<runLen<<" ms"<<endl;
		}

		if(currentTrial==2)
		{
			if(currentTime>1750 && currentTime<3000)
			{
				emit(spatialFrameDump());
			}
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

//		apGL=management->exportAPGL();
//		for(int i=0; i<numGL; i++)
//		{
//			apGLVis[i]=apGL[i];
//		}
//		emit(updateSpatialW(apGLVis, 2, false));

		apGO=management->exportAPGO();
		for(int i=0; i<numGO; i++)
		{
			apGOVis[i]=apGO[i];
		}
		emit(updateINTW(apGOVis, currentTime));
//
//		apSC=management->exportAPSC();
//		for(int i=0; i<numSC; i++)
//		{
//			apSCVis[i]=apSC[i];
//		}
//		emit(updateSCTW(apSCVis, currentTime));
//
//		apBC=management->exportAPBC();
//		for(int i=0; i<numBC; i++)
//		{
//			apBCVis[i]=apBC[i];
//		}
//		emit(updateBCTW(apBCVis, currentTime));
//
//		apPC=management->exportAPPC();
//		vmPC=management->exportVmPC();
//		for(int i=0; i<numPC; i++)
//		{
//			apPCVis[i]=apPC[i];
//			vmPCVis[i]=(vmPC[i]+80)/80;
//		}
//		emit(updatePCTW(apPCVis, vmPCVis, currentTime));
//
//		apNC=management->exportAPNC();
//		vmNC=management->exportVmNC();
//		for(int i=0; i<numNC; i++)
//		{
//			apNCVis[i]=apNC[i];
//			vmNCVis[i]=(vmNC[i]+80)/80;
//		}
//		emit(updateNCTW(apNCVis, vmNCVis, currentTime));
//
//		apIO=management->exportAPIO();
//		vmIO=management->exportVmIO();
//		for(int i=0; i<numIO; i++)
//		{
//			apIOVis[i]=apIO[i];
//			vmIOVis[i]=(vmIO[i]+80)/80;
//		}
//		emit(updateIOTW(apIOVis, vmIOVis, currentTime));

		unlockAccessData();
	}
}
