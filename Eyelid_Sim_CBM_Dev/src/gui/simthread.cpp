/*
 * simthread.cpp
 *
 *  Created on: Mar 2, 2012
 *      Author: consciousness
 */


#include "../../includes/gui/simthread.h"
#include "../../includes/gui/moc/moc_simthread.h"

#include <math.h>

using namespace std;

SimThread::SimThread(QObject *parent, ECManagementBase *ecsim,
		ActSpatialView *inputNetSV,
		ActTemporalView *inputNetTV,
		ActTemporalView *scTV,
		ActTemporalView *bcTV,
		ActTemporalView *pcTV,
		ActTemporalView *ncTV,
		ActTemporalView *ioTV,
		InterThreadComm *interThreadData)
	: QThread(parent)
{
	management=ecsim;

	conParams=management->getConParams();

	inputNet=management->getInputNet();

	mZone=management->getMZone();

	inputNetSView=inputNetSV;

	inputNetTView=inputNetTV;
	scTView=scTV;
	bcTView=bcTV;
	pcTView=pcTV;
	ioTView=ioTV;
	ncTView=ncTV;

	qRegisterMetaType<std::vector<ct_uint8_t> >("std::vector<ct_uint8_t>");
	qRegisterMetaType<std::vector<float> >("std::vector<float>");
	qRegisterMetaType<QColor>("QColor");

	connect(this, SIGNAL(updateSpatialW(std::vector<ct_uint8_t>, int, bool)),
			inputNetSView, SLOT(drawActivity(std::vector<ct_uint8_t>, int, bool)),
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

	connect(this, SIGNAL(updateINTW(std::vector<ct_uint8_t>, int)),
			inputNetTView, SLOT(drawRaster(std::vector<ct_uint8_t>, int)),
			Qt::QueuedConnection);
	connect(this, SIGNAL(updateBCTW(std::vector<ct_uint8_t>, int)),
			bcTView, SLOT(drawRaster(std::vector<ct_uint8_t>, int)),
			Qt::QueuedConnection);
	connect(this, SIGNAL(updateSCTW(std::vector<ct_uint8_t>, int)),
			scTView, SLOT(drawRaster(std::vector<ct_uint8_t>, int)),
			Qt::QueuedConnection);
	connect(this, SIGNAL(updatePCTW(std::vector<ct_uint8_t>, std::vector<float>, int)),
			pcTView, SLOT(drawVmRaster(std::vector<ct_uint8_t>, std::vector<float>, int)),
			Qt::QueuedConnection);
	connect(this, SIGNAL(updateNCTW(std::vector<ct_uint8_t>, std::vector<float>, int)),
			ncTView, SLOT(drawVmRaster(std::vector<ct_uint8_t>, std::vector<float>, int)),
			Qt::QueuedConnection);
	connect(this, SIGNAL(updateIOTW(std::vector<ct_uint8_t>, std::vector<float>, int)),
			ioTView, SLOT(drawVmRaster(std::vector<ct_uint8_t>, std::vector<float>, int)),
			Qt::QueuedConnection);

	itc=interThreadData;
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
	vector<ct_uint8_t> apGRVis;
	vector<ct_uint8_t> apGOVis;
	vector<ct_uint8_t> apGLVis;
	vector<ct_uint8_t> apMFVis;

	vector<ct_uint8_t> apSCVis;
	vector<ct_uint8_t> apBCVis;

	vector<ct_uint8_t> apPCVis;
	vector<float> vmPCVis;
	vector<ct_uint8_t> apNCVis;
	vector<float> vmNCVis;
	vector<ct_uint8_t> apIOVis;
	vector<float> vmIOVis;

	const ct_uint8_t *apGR;
	const ct_uint8_t *apGO;
	const ct_uint8_t *apGL;
	const ct_uint8_t *apMF;

	const ct_uint8_t *apSC;
	const ct_uint8_t *apBC;

	const ct_uint8_t *apPC;
	const float *vmPC;
	const ct_uint8_t *apNC;
	const float *vmNC;
	const ct_uint8_t *apIO;
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

	int inNetDispCellT;

	numGR=conParams->getNumGR();
	numGO=conParams->getNumGO();
	numGL=conParams->getNumGL();
	numMF=conParams->getNumMF();

	numSC=conParams->getNumSC();
	numBC=conParams->getNumBC();
	numPC=conParams->getNumPC();
	numNC=conParams->getNumNC();
	numIO=conParams->getNumIO();
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
		const float *gESumGR;
		const float *vmGR;

		lockAccessData();

		notDone=management->runStep();
		if(!notDone)
		{
			break;
		}
		currentTime=management->getCurrentTime();
		if(currentTime>=(iti-1))
		{
			runLen=timer.restart();
			currentTrial=management->getCurrentTrialN();

			emit(blankTW(Qt::black));

			cerr<<"run time for trial #"<<currentTrial<<": "<<runLen<<" ms"<<endl;
		}

//		if(currentTrial==2)
//		{
//			if(currentTime>1750 && currentTime<3000)
//			{
//				emit(spatialFrameDump());
//			}
//		}

//		apGR=inputNet->exportAPGR();
//		for(int i=0; i<numGR; i++)
//		{
//			apGRVis[i]=apGR[i];
//		}
//		emit(updateSpatialW(apGRVis, 0, true));
//
//
//		apGO=inputNet->exportAPGO();
//		for(int i=0; i<numGO; i++)
//		{
//			apGOVis[i]=apGO[i];
//		}
//		emit(updateSpatialW(apGOVis, 1, false));

//		apGL=management->exportAPGL();
//		for(int i=0; i<numGL; i++)
//		{
//			apGLVis[i]=apGL[i];
//		}
//		emit(updateSpatialW(apGLVis, 2, false));

		itc->accessDispParamLock.lock();
		inNetDispCellT=itc->inNetDispCellT;
		itc->accessDispParamLock.unlock();
		if(inNetDispCellT==0)
		{
			apGO=management->exportAPMF();
		}
		else if(inNetDispCellT==1)
		{
			apGO=inputNet->exportAPGO();
		}
		else
		{
			apGO=inputNet->exportAPGR();
		}
//		apGO=management->exportAPMF();
//		gESumGR=inputNet->exportGESumGR();
//		cout<<"gESumGR "<<gESumGR[0]<<" "<<gESumGR[100]<<" "<<gESumGR[200]<<gESumGR[1048575]<<endl;
//		vmGR=inputNet->exportVmGR();
//		cout<<"vmGR "<<vmGR[0]<<" "<<vmGR[100]<<" "<<vmGR[200]<<" "<<vmGR[1048575]<<endl;
//		int numBadGRs=0;
//		for(int i=0; i<numGR; i++)
//		{
////			if(vmGR[i]==numeric_limits<float>::infinity() || vmGR[i]==-numeric_limits<float>::infinity() ||
////					vmGR[i]==numeric_limits<float>::quiet_NaN())
//			if(isnanf(vmGR[i]) || isinff(vmGR[i]))
//			{
//				numBadGRs++;
//			}
//		}

//		cout<<"numBadGRs "<<numBadGRs<<endl;
//		apGO=inputNet->exportAPGR();
//		}
		for(int i=0; i<numGO; i++)
		{
			apGOVis[i]=apGO[i];
		}
		emit(updateINTW(apGOVis, currentTime));

		apSC=inputNet->exportAPSC();
		for(int i=0; i<numSC; i++)
		{
			apSCVis[i]=apSC[i];
		}
		emit(updateSCTW(apSCVis, currentTime));

		apBC=mZone->exportAPBC();
		for(int i=0; i<numBC; i++)
		{
			apBCVis[i]=apBC[i];
		}
		emit(updateBCTW(apBCVis, currentTime));

		apPC=mZone->exportAPPC();
		vmPC=mZone->exportVmPC();
		for(int i=0; i<numPC; i++)
		{
			apPCVis[i]=apPC[i];
			vmPCVis[i]=(vmPC[i]+80)/80;
		}
		emit(updatePCTW(apPCVis, vmPCVis, currentTime));

		apNC=mZone->exportAPNC();
		vmNC=mZone->exportVmNC();
		for(int i=0; i<numNC; i++)
		{
			apNCVis[i]=apNC[i];
			vmNCVis[i]=(vmNC[i]+80)/80;
		}
		emit(updateNCTW(apNCVis, vmNCVis, currentTime));

		apIO=mZone->exportAPIO();
		vmIO=mZone->exportVmIO();
		for(int i=0; i<numIO; i++)
		{
			apIOVis[i]=apIO[i];
			vmIOVis[i]=(vmIO[i]+80)/80;
		}
		emit(updateIOTW(apIOVis, vmIOVis, currentTime));

		unlockAccessData();
	}

	((ECManagementDelay*)management)->writeDataToFile();
}
