/*
 * simthread.cpp
 *
 *  Created on: Feb 16, 2009
 *      Author: wen
 */

#include "../../includes/gui/simthread.h"
#include "../../includes/gui/moc_simthread.h"
#include "../../includes/globalvars.h"

SimThread::SimThread(QObject *parent, SimDispW *panel, ActDiagW *actW) : QThread(parent)
{
	dispW=panel;
	activityW=actW;
	qRegisterMetaType<SCBCPCActs>("SCBCPCActs");
	qRegisterMetaType<IONCPCActs>("IONCPCActs");
	qRegisterMetaType<vector<bool> >("vector<bool>");
	qRegisterMetaType<vector<unsigned short> >("vector<unsigned short>");
	qRegisterMetaType<vector<unsigned char> >("vector<unsigned char>");
	connect(this, SIGNAL(updateRaster(vector<bool>, int)), dispW, SLOT(drawRaster(vector<bool>, int)), Qt::QueuedConnection);
	connect(this, SIGNAL(updatePSH(vector<unsigned short>, int, bool)), dispW, SLOT(drawPSH(vector<unsigned short>, int, bool)), Qt::QueuedConnection);
	connect(this, SIGNAL(updateSCBCPCActs(SCBCPCActs, int)), dispW, SLOT(drawSCBCPCActs(SCBCPCActs, int)), Qt::QueuedConnection);
	connect(this, SIGNAL(updateIONCPCActs(IONCPCActs, int)), dispW, SLOT(drawIONCPCActs(IONCPCActs, int)), Qt::QueuedConnection);
	connect(this, SIGNAL(updateTotalAct(int, int)), dispW, SLOT(drawTotalAct(int, int)), Qt::QueuedConnection);
	connect(this, SIGNAL(updateCSBackground(int)), dispW, SLOT(drawCSBackground(int)), Qt::QueuedConnection);
	connect(this, SIGNAL(updateBlankDisp()), dispW, SLOT(drawBlankDisp()), Qt::QueuedConnection);
	connect(this, SIGNAL(updateActW(vector<bool>, vector<bool>)), activityW, SLOT(drawActivity(vector<bool>, vector<bool>)), Qt::QueuedConnection);
}

SimThread::~SimThread()
{
	cleanSim();
}

void SimThread::run()
{
	if(dispW==NULL || !initialized)
	{
		return;
	}

//	initCUDA();
	cudaSetDevice(0);


	simLoopNew();
}

void SimThread::simLoopNew()
{
	int trialRunTime;
	int nRuns=0;

	IONCPCActs inpActs;
	SCBCPCActs sbpActs;
	vector<bool> apRaster(NUMMF, false);
	vector<bool> grActs(NUMGR, 0);
	vector<bool> goActs(NUMGO, 0);

	if(dispW==NULL)
	{
		return;
	}

	cout<<"pre-run to stabilize network"<<endl;
	trialRunTime=time(NULL);
	for(int i=5000; i<10000; i++)
	{
		calcCellActivities(i, *randGen);

//		cout<<i<<endl;
		if(i%500==0)
		{
			cout<<(i/500-10)*10<<" % complete"<<endl;
		}
	}
	cout<<"pre-run completed in "<<time(NULL)-trialRunTime<<" seconds"<<endl;

	while(true) //for(int trials=0; trials<1000; trials++)//
	{
		bool calcPSH;
		bool dispRaster;
		bool dispGRGOAct;
		bool calcSpikeHist;

		int dispType;

		simStopLock.lock();
		if(simStop)
		{
			simStopLock.unlock();
			break;
		}
		simStopLock.unlock();

		simPSHCheckLock.lock();
		calcPSH=simPSHCheck;
		//reset bin counter for PSH
		if(calcPSH)
		{
			mfPSH->resetCurrentBinN();
			goPSH->resetCurrentBinN();
			grPSH->resetCurrentBinN();
			scPSH->resetCurrentBinN();

			for(int i=0; i<NUMMZONES; i++)
			{
				bcPSH[i]->resetCurrentBinN();
				pcPSH[i]->resetCurrentBinN();
				ioPSH[i]->resetCurrentBinN();
				ncPSH[i]->resetCurrentBinN();
			}
		}
		simPSHCheckLock.unlock();

		simCalcSpikeHistLock.lock();
		calcSpikeHist=simCalcSpikeHist;
		simCalcSpikeHistLock.unlock();

		simDispRasterLock.lock();
		dispRaster=simDispRaster;
		simDispRasterLock.unlock();

		trialRunTime=time(NULL);
		nRuns++;

		if(dispRaster)
		{
			emit updateBlankDisp();
		}

		for(short i=0; i<TRIALTIME; i++)
		{
			simDispActsLock.lock();
			dispGRGOAct=simDispActs;
			simDispActsLock.unlock();

			simStopLock.lock();
			if(simStop)
			{
				simStopLock.unlock();
				break;
			}
			simStopLock.unlock();

			if(i%(TRIALTIME/5)==0)
			{
				cout<<i<<"ms ";
				cout.flush();
			}

			simPauseLock.lock();
			calcCellActivities(i, *randGen);

//			if(calcSpikeHist)
//			{

//			}

			if(i%(NUMBINSINAPBUF*PSHBINWIDTH)==0
					&& i>csOnset[0]-(PSHPRESTIMNUMBINS*PSHBINWIDTH)
					&& i<=csOnset[0]+((PSHSTIMNUMBINS+PSHPOSTSTIMNUMBINS)*PSHBINWIDTH)
					&& calcPSH)
			{
				accessPSHLock.lock();
				mfPSH->updatePSH();
				goPSH->updatePSH();
				grPSH->updatePSH();
				scPSH->updatePSH();

				for(int i=0; i<NUMMZONES; i++)
				{
					bcPSH[i]->updatePSH();
					pcPSH[i]->updatePSH();
					ioPSH[i]->updatePSH();
					ncPSH[i]->updatePSH();
				}
				accessPSHLock.unlock();
			}

			if(dispGRGOAct)
			{
				inputNetwork->exportActGODisp(goActs, inputNetwork->numGO);
				inputNetwork->exportActGRDisp(grActs, inputNetwork->numGR);
				emit updateActW(grActs, goActs);
			}

			if(dispRaster)
			{
				simDispTypeLock.lock();
				dispType=simDispType;
				simDispTypeLock.unlock();

				if(dispType==0)
				{
					mfMod->exportActDisp(apRaster, 1024);
					emit updateRaster(apRaster, i);
				}
				else if(dispType==1)
				{
					inputNetwork->exportActGRDisp(apRaster, 1024);
					emit updateRaster(apRaster, i);
				}
				else if(dispType==2)
				{
					inputNetwork->exportActGODisp(apRaster, 1024);
					emit updateRaster(apRaster, i);
				}
				else if(dispType==3)
				{
					inputNetwork->exportActSCDisp(sbpActs);
					simMZDispNumLock.lock();
					zones[simMZDispNum]->exportActsPCBCDisp(sbpActs);
					simMZDispNumLock.unlock();
					emit updateSCBCPCActs(sbpActs, i);
				}
				else
				{
					simMZDispNumLock.lock();
					zones[simMZDispNum]->exportActsIONCPCDisp(inpActs);
					simMZDispNumLock.unlock();
					emit updateIONCPCActs(inpActs, i);
				}
			}

			simPauseLock.unlock();
		}
		pfSynWeightPCLock.lock();
		for(int i=0; i<NUMMZONES; i++)
		{
			zones[i]->cpyPFPCSynWCUDA();
		}
		pfSynWeightPCLock.unlock();

		cout<<endl<<"trial #"<<nRuns<<" "<<"trial run time: "<<time(NULL)-trialRunTime<<endl;
	}
}
