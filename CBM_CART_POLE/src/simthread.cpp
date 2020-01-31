/*
 * simthread.cpp
 *
 *  Created on: Feb 16, 2009
 *      Author: wen
 */

#include "../includes/simthread.h"
#include "../includes/moc_simthread.h"

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
	connect(this, SIGNAL(updateActW(vector<unsigned char>, vector<bool>)), activityW, SLOT(drawActivity(vector<unsigned char>, vector<bool>)), Qt::QueuedConnection);
}


void SimThread::run()
{
	if(dispW==NULL || !initialized)
	{
		return;
	}

	initCUDA();

//	simLoop();
	simLoopNew();
}

void SimThread::simLoop()
{
	CRandomSFMT0 randGen(time(NULL));
	int trialTime;

	IONCPCActs inpActs;
	SCBCPCActs sbpActs;
	vector<bool> apRaster(NUMMF, false);
	vector<unsigned short> pshGreyVals(NUMMF, 0);
	vector<unsigned char> grActs(NUMGR, 0);
	vector<bool> goActs(NUMGO, 0);

	if(dispW==NULL)
	{
		return;
	}

	cout<<"pre-run to stabilize network"<<endl;
	trialTime=time(NULL);
	for(int i=5000; i<15000; i++)
	{
		calcCellActivities(i, randGen);
		if(i%1000==0)
		{
			cout<<(i/1000-5)*10<<" % complete"<<endl;
		}
	}
	cout<<"pre-run completed in "<<time(NULL)-trialTime<<" seconds"<<endl;

	numTrials=0;
	while(true)//for(nTrials=0; nTrials<2001; nTrials++)
	{
		bool pshOn;

		float avgPFPCInNCS[NUMPC];
		float avgPFPCInCS[NUMPC];
		float avgPCAPNCS[NUMPC];
		float avgPCAPCS[NUMPC];
		float avgSCAPNCS;
		float avgSCAPCS;
		float avgGRAPNCS;
		float avgGRAPCS;

		memset(avgPFPCInNCS, 0, NUMPC*sizeof(float));
		memset(avgPFPCInCS, 0, NUMPC*sizeof(float));
		memset(avgPCAPNCS, 0, NUMPC*sizeof(float));
		memset(avgPCAPCS, 0, NUMPC*sizeof(float));
		avgGRAPNCS=0;
		avgGRAPCS=0;

		simStopLock.lock();
		if(simStop)
		{
			simStopLock.unlock();
			break;
		}
		simStopLock.unlock();

		simPSHCheckLock.lock();
		pshOn=simPSHCheck;
		simPSHCheckLock.unlock();

		accessPSHLock.lock();
		numTrials++;
		accessPSHLock.unlock();

		trialTime=time(NULL);

		updateMFCSOn();

		emit updateBlankDisp();

		for(short i=0; i<TRIALTIME; i++)
		{
			unsigned char *drawAP;
			int dispType;
			int activeCells;
			int totalCells;

			int activeSC;

			float cellProps;
			unsigned short *pshDraw, *pshMaxDraw;
			float drawScale;

			simStopLock.lock();
			if(simStop)
			{
				simStopLock.unlock();
				break;
			}
			simStopLock.unlock();

			simPauseLock.lock();

			calcCellActivities(i, randGen);

			accessSpikeSumLock.lock();
			for(int j=0; j<NUMGR; j++)
			{
				spikeSumGR[j]+=apOutGR[j]&0x01;
//				if(apOutGR[j]&0x01)
//				{
//					spikeSumGR[j]++;
//				}
			}

//			spikeSumGOLock.lock();
			for(int j=0; j<NUMGO; j++)
			{
				spikeSumGO[j]+=apGO[j];
//				if(apGO[j])
//				{
//					spikeSumGO[j]++;
//				}
			}
//			spikeSumGOLock.unlock();
//			spikeSumSCLock.lock();
			for(int j=0; j<NUMSC; j++)
			{
				spikeSumSC[j]+=apSC[j];
//				if(apSC[j])
//				{
//					spikeSumSC[j]++;
//				}
			}
//			spikeSumSCLock.unlock();
//			spikeSumBCLock.lock();
			for(int j=0; j<NUMBC; j++)
			{
				spikeSumBC[j]+=apBC[j];
//				if(apBC[j])
//				{
//					spikeSumBC[j]++;
//				}
			}
//			spikeSumBCLock.unlock();
//			spikeSumPCLock.lock();
			for(int j=0; j<NUMPC; j++)
			{
				spikeSumPC[j]+=apPC[j];
//				if(apPC[j])
//				{
//					spikeSumPC[j]++;
//				}
			}
//			spikeSumPCLock.unlock();
//			spikeSumNCLock.lock();
			for(int j=0; j<NUMNC; j++)
			{
				spikeSumNC[j]+=apNC[j];
//				if(apNC[j])
//				{
//					spikeSumNC[j]++;
//				}
			}
//			spikeSumNCLock.unlock();

//			msCountLock.lock();
			msCount++;
//			msCountLock.unlock();
			accessSpikeSumLock.unlock();

			simDispTypeLock.lock();
			dispType=simDispType;
			simDispTypeLock.unlock();
			drawAP=apOutGR;
			totalCells=NUMGR;
			if(dispType==0)
			{
				drawAP=(unsigned char *)apMF;
				totalCells=NUMMF;
				pshDraw=(unsigned short *)pshMF;
				pshMaxDraw=&pshMFMax;
				drawScale=3;
			}
			else if(dispType==1)
			{
				drawAP=apOutGR;
				totalCells=NUMGR;
				pshDraw=(unsigned short *)pshGR;
				pshMaxDraw=&pshGRMax;
				drawScale=50;
			}
			else if(dispType==2)
			{
				drawAP=(unsigned char *)apGO;
				totalCells=NUMGO;
				pshDraw=(unsigned short *)pshGO;
				pshMaxDraw=&pshGOMax;
				drawScale=3;
			}

			activeCells=0;
			for(int j=0; j<totalCells; j++)
			{
				activeCells=activeCells+(drawAP[j]&0x01);
			}
			cellProps=(float)activeCells/(float)totalCells;

			activeSC=0;
			for(int j=0; j<NUMSC; j++)
			{
				activeSC=activeSC+apSC[j];
			}

			if(i>=csOnset[activeCS] && i<csOnset[activeCS]+csDuration[activeCS])
			{
				for(int j=0; j<NUMPC; j++)
				{
					avgPFPCInCS[j]=avgPFPCInCS[j]+inputSumPFPC[j];
					avgPCAPCS[j]=avgPCAPCS[j]+apPC[j];
				}
				avgGRAPCS=avgGRAPCS+activeCells;
				avgSCAPCS=avgSCAPCS+activeSC;
				//				cout<<"CSON---------------------"<<endl;
			}
			else
			{
				for(int j=0; j<NUMPC; j++)
				{
					avgPFPCInNCS[j]=avgPFPCInNCS[j]+inputSumPFPC[j];
					avgPCAPNCS[j]=avgPCAPNCS[j]+apPC[j];
				}
				avgGRAPNCS=avgGRAPNCS+activeCells;
				avgSCAPNCS=avgSCAPNCS+activeSC;
			}
			//			cout<<activeCells<<" ";
			//			for(int j=0; j<8; j++)
			//			{
			//				cout<<inputSumPFPC[j]<<" ";
			//			}
			//			cout<<endl;
			//
			//			for(int j=0; j<8; j++)
			//			{
			//				cout<<apPC[j]<<" ";
			//			}
			//			cout<<endl<<"v: ";
			//
			//			for(int j=0; j<8; j++)
			//			{
			//				cout<<vPC[j]<<" ";
			//			}
			//			cout<<endl<<"thresh: ";;
			//
			//			for(int j=0; j<8; j++)
			//			{
			//				cout<<threshPC[j]<<" ";
			//			}
			//			cout<<endl;

			if(i>=csOnset[activeCS]-100 && i<csOnset[activeCS]+csDuration[activeCS]+100)
			{
				//				if(v>0)
				if(pshOn)
				{
					int greyVal;
					//do peri-stimulus histogram stats here
					int binN;
					QColor pshC;

					accessPSHLock.lock();
					binN=(int)((i-csOnset[activeCS]+100)/PSHBINWIDTH);//((float)csDuration[activeCS]+200/PSHNUMBINS));

					for(int j=0; j<NUMGR; j++)
					{
						pshGR[binN][j]+=(apOutGR[j]&0x01);
						pshGRMax=(pshGR[binN][j]>pshGRMax)*pshGR[binN][j]+(!(pshGR[binN][j]>pshGRMax))*pshGRMax;
					}

					for(int j=0; j<NUMGO; j++)
					{
						pshGO[binN][j]+=apGO[j];
						pshGOMax=(pshGO[binN][j]>pshGOMax)*pshGO[binN][j]+(!(pshGO[binN][j]>pshGOMax))*pshGOMax;
					}

					for(int j=0; j<NUMMF; j++)
					{
						pshMF[binN][j]+=apMF[j];
						pshMFMax=(pshMF[binN][j]>pshMFMax)*pshMF[binN][j]+(!(pshMF[binN][j]>pshMFMax))*pshMFMax;
					}
					for(int j=0; j<NUMPC; j++)
					{
						pshPC[binN][j]+=apPC[j];
						pshPCMax=(pshPC[binN][j]>pshPCMax)*pshPC[binN][j]+(!(pshPC[binN][j]>pshPCMax))*pshPCMax;
					}
					for(int j=0; j<NUMBC; j++)
					{
						pshBC[binN][j]+=apBC[j];
						pshBCMax=(pshBC[binN][j]>pshBCMax)*pshBC[binN][j]+(!(pshBC[binN][j]>pshBCMax))*pshBCMax;
					}
					for(int j=0; j<NUMSC; j++)
					{
						pshSC[binN][j]+=apSC[j];
						pshSCMax=(pshSC[binN][j]>pshSCMax)*pshSC[binN][j]+(!(pshSC[binN][j]>pshSCMax))*pshSCMax;
					}
					//end
					//draw peri-stimulus histogram stats
					if(dispType<3)
					{
						for(int j=0; j<NUMMF; j++)
						{
							pshGreyVals[j]=(unsigned short)(((float)pshDraw[binN*totalCells+j]/(*pshMaxDraw))*255);
						}
						//						cout<<"t="<<i-csOnset[activeCS]+100<<" "<<"binN="<<binN<<endl;

						if(i>=csOnset[activeCS] && i<csOnset[activeCS]+csDuration[activeCS])
						{
							emit updatePSH(pshGreyVals, i, true);
						}
						else
						{
							emit updatePSH(pshGreyVals, i, false);
						}
					}
					else if(i>=csOnset[activeCS] && i<csOnset[activeCS]+csDuration[activeCS])
					{
						emit updateCSBackground(i);
					}
					accessPSHLock.unlock();
				}
				else if(i>=csOnset[activeCS] && i<csOnset[activeCS]+csDuration[activeCS])
				{
					emit updateCSBackground(i);
				}
			}

			simDispTypeLock.lock();
			//			p.setPen(Qt::white);
			if(simDispType<3)
			{
				for(int j=0; j<NUMMF; j++)
				{
					apRaster[j]=(bool)(drawAP[j]&0x01);
				}
				emit updateRaster(apRaster, i);
				//				for(int j=0; j<NUMMF; j++)
				//				{
				//					if(drawAP[j])
				//					{
				//						p.drawPoint(i, j);
				//					}
				//				}
			}
			else
			{
				//				p.setPen(Qt::red);
				//				for(int j=0; j<NUMPC; j++)
				//				{
				//					p.drawPoint(i, (int)(12*(j-(vPC[j]/100.0f)))+NUMSC+NUMBC);//
				//					if(apPC[j])
				//					{
				//						p.drawLine(i, (12*j)+NUMSC+NUMBC, i, (int)(12*(j-(vPC[j]/100)))+NUMSC+NUMBC);
				//					}
				//				}
				if(simDispType==3)
				{
					for(int j=0; j<NUMPC; j++)
					{
						sbpActs.apPC[j]=apPC[j];
						sbpActs.vPC[j]=vPC[j];
					}
					for(int j=0; j<NUMBC; j++)
					{
						sbpActs.apBC[j]=apBC[j];
					}
					for(int j=0; j<NUMSC; j++)
					{
						sbpActs.apSC[j]=apSC[j];
					}
					emit updateSCBCPCActs(sbpActs, i);
					//					p.setPen(Qt::white);
					//					for(int j=0; j<NUMSC; j++)
					//					{
					//						if(apSC[j])
					//						{
					//							p.drawPoint(i, j);
					//						}
					//					}
					//					p.setPen(Qt::green);
					//					for(int j=0; j<NUMBC; j++)
					//					{
					//						if(apBC[j])
					//						{
					//							p.drawPoint(i, j+NUMSC);
					//						}
					//					}
				}
				else
				{
					for(int j=0; j<NUMPC; j++)
					{
						inpActs.apPC[j]=apPC[j];
						inpActs.vPC[j]=vPC[j];
					}
					for(int j=0; j<NUMIO; j++)
					{
						inpActs.apIO[j]=apIO[j];
						inpActs.vIO[j]=vIO[j];
					}
					for(int j=0; j<NUMNC; j++)
					{
						inpActs.apNC[j]=apNC[j];
						inpActs.vNC[j]=vNC[j];
					}

					emit updateIONCPCActs(inpActs, i);
					//					p.setPen(Qt::white);
					//					for(int j=0; j<NUMIO; j++)
					//					{
					//						p.drawPoint(i, 120*(j-vIO[j]/100.0f));
					//						if(apIO[j])
					//						{
					//							p.drawLine(i, 120*j, i, 120*(j-vIO[j]/100.0f));
					//						}
					//					}
					//
					//					p.setPen(Qt::green);
					//					for(int j=0; j<NUMNC; j++)
					//					{
					//						p.drawPoint(i, 15*(j-vNC[j]/100.0f)+480);
					//						if(apNC[j])
					//						{
					//							p.drawLine(i, 15*j+480, i, 15*(j-vNC[j]/100.0f)+480);
					//						}
					//					}
					//					cout<<vNC[0]<<endl;
				}
			}
			simDispTypeLock.unlock();

			//draw overall cell activity
			//			p.setPen(Qt::red);
			//			p.drawPoint(i, (int)((cellProps*NUMMF*drawScale)));

			emit updateTotalAct((int)(cellProps*NUMMF*drawScale), i);

			simDispActsLock.lock();
			if(simDispActs)
			{
				for(int i=0; i<NUMGR; i++)
				{
					grActs[i]=apOutGR[i];
				}
				for(int i=0; i<NUMGO; i++)
				{
					goActs[i]=apGO[i];
				}
				emit updateActW(grActs, goActs);
			}
			simDispActsLock.unlock();

			memset(apMF, false, NUMMF*sizeof(bool));
			memset(apOutGR, 0, NUMGR*sizeof(unsigned char));
			memset(apGO, false, NUMGO*sizeof(bool));

			simPauseLock.unlock();
		}

		pfSynWeightPCLock.lock();
		cudaMemcpy(pfSynWeightPC, pfSynWeightPCGPU, NUMGR*sizeof(float), cudaMemcpyDeviceToHost);
		//		float tempmax=0;
		//		for(int i=0; i<NUMGR; i++)
		//		{
		//			if(pfSynWeightPC[i]>tempmax)
		//				tempmax=pfSynWeightPC[i];
		//		}
		//		cout<<"max pf syn weight: "<<tempmax<<endl;
		pfSynWeightPCLock.unlock();

		cout<<"trial run time: "<<time(NULL)-trialTime<<endl;

		for(int j=0; j<NUMPC; j++)
		{
			avgPFPCInCS[j]=avgPFPCInCS[j]/csDuration[activeCS];
			avgPCAPCS[j]=avgPCAPCS[j]/csDuration[activeCS];
			avgPFPCInNCS[j]=avgPFPCInNCS[j]/(TRIALTIME-csDuration[activeCS]);
			avgPCAPNCS[j]=avgPCAPNCS[j]/(TRIALTIME-csDuration[activeCS]);
			cout<<"PC#"<<j<<" PFPC(CS):"<<avgPFPCInCS[j]<<" PFPC(NCS):"<<avgPFPCInNCS[j]<<" %diff:"<<(avgPFPCInCS[j]-avgPFPCInNCS[j])/avgPFPCInNCS[j]<<" | PCAP(CS):"
					<<avgPCAPCS[j]*1000<<" PCAP(NCS):"<<avgPCAPNCS[j]*1000<<" %diff:"<<(avgPCAPCS[j]-avgPCAPNCS[j])/avgPCAPNCS[j]<<endl;
		}
		avgGRAPCS=avgGRAPCS/csDuration[activeCS];
		avgGRAPNCS=avgGRAPNCS/(TRIALTIME-csDuration[activeCS]);
		avgSCAPCS=avgSCAPCS/csDuration[activeCS];
		avgSCAPNCS=avgSCAPNCS/(TRIALTIME-csDuration[activeCS]);
		cout<<"#GR active: (CS): "<<avgGRAPCS<<" (NCS): "<<avgGRAPNCS<<" %diff: "<<(avgGRAPCS-avgGRAPNCS)/avgGRAPNCS<<endl;
		cout<<"# SC active: (CS): "<<avgSCAPCS<<" (NCS): "<<avgSCAPNCS<<" %diff: "<<(avgSCAPCS-avgSCAPNCS)/avgSCAPNCS<<endl;
		accessPSHLock.lock();
		cout<<"trial #"<<numTrials<<endl;
		accessPSHLock.unlock();
	}
}

void SimThread::simLoopNew()
{
	CRandomSFMT0 randGen(time(NULL));
	int trialRunTime;
	int nRuns=0;

	IONCPCActs inpActs;
	SCBCPCActs sbpActs;
	vector<bool> apRaster(NUMMF, false);
	vector<unsigned char> grActs(NUMGR, 0);
	vector<bool> goActs(NUMGO, 0);

	if(dispW==NULL)
	{
		return;
	}

	cout<<"pre-run to stabilize network"<<endl;
	trialRunTime=time(NULL);
	for(int i=5000; i<15000; i++)
	{
		calcCellActivities(i, randGen);
		if(i%1000==0)
		{
			cout<<(i/1000-5)*10<<" % complete"<<endl;
		}
	}
	cout<<"pre-run completed in "<<time(NULL)-trialRunTime<<" seconds"<<endl;

	accessPSHLock.lock();
	numTrials=0;
	accessPSHLock.unlock();

	while(true)
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
		simPSHCheckLock.unlock();

		simCalcSpikeHistLock.lock();
		calcSpikeHist=simCalcSpikeHist;
		simCalcSpikeHistLock.unlock();

		simDispRasterLock.lock();
		dispRaster=simDispRaster;
		simDispRasterLock.unlock();


		if(calcPSH)
		{
			accessPSHLock.lock();
			numTrials++;
			accessPSHLock.unlock();
		}

		trialRunTime=time(NULL);
		nRuns++;

		updateMFCSOn();

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
			calcCellActivities(i, randGen);

			if(calcSpikeHist)
			{
				accessSpikeSumLock.lock();
				for(int j=0; j<NUMGR; j++)
				{
					spikeSumGR[j]+=apOutGR[j]&0x01;
				}

				for(int j=0; j<NUMGO; j++)
				{
					spikeSumGO[j]+=apGO[j];
				}

				for(int j=0; j<NUMSC; j++)
				{
					spikeSumSC[j]+=apSC[j];
				}

				for(int j=0; j<NUMBC; j++)
				{
					spikeSumBC[j]+=apBC[j];
				}

				for(int j=0; j<NUMPC; j++)
				{
					spikeSumPC[j]+=apPC[j];
				}

				for(int j=0; j<NUMNC; j++)
				{
					spikeSumNC[j]+=apNC[j];
				}

				msCount++;
				accessSpikeSumLock.unlock();
			}

			if(i>=csOnset[activeCS]-100 && i<csOnset[activeCS]+csDuration[activeCS]+100 && calcPSH)
			{
				int binN;
				accessPSHLock.lock();
				binN=(int)((i-csOnset[activeCS]+100)/PSHBINWIDTH);//((float)csDuration[activeCS]+200/PSHNUMBINS));

				for(int j=0; j<NUMGR; j++)
				{
					pshGR[binN][j]+=(apOutGR[j]&0x01);
//					pshGRMax=(pshGR[binN][j]>pshGRMax)*pshGR[binN][j]+(!(pshGR[binN][j]>pshGRMax))*pshGRMax;
				}

				for(int j=0; j<NUMGO; j++)
				{
					pshGO[binN][j]+=apGO[j];
//					pshGOMax=(pshGO[binN][j]>pshGOMax)*pshGO[binN][j]+(!(pshGO[binN][j]>pshGOMax))*pshGOMax;
				}

				for(int j=0; j<NUMMF; j++)
				{
					pshMF[binN][j]+=apMF[j];
//					pshMFMax=(pshMF[binN][j]>pshMFMax)*pshMF[binN][j]+(!(pshMF[binN][j]>pshMFMax))*pshMFMax;
				}
				for(int j=0; j<NUMPC; j++)
				{
					pshPC[binN][j]+=apPC[j];
//					pshPCMax=(pshPC[binN][j]>pshPCMax)*pshPC[binN][j]+(!(pshPC[binN][j]>pshPCMax))*pshPCMax;
				}
				for(int j=0; j<NUMBC; j++)
				{
					pshBC[binN][j]+=apBC[j];
//					pshBCMax=(pshBC[binN][j]>pshBCMax)*pshBC[binN][j]+(!(pshBC[binN][j]>pshBCMax))*pshBCMax;
				}
				for(int j=0; j<NUMSC; j++)
				{
					pshSC[binN][j]+=apSC[j];
//					pshSCMax=(pshSC[binN][j]>pshSCMax)*pshSC[binN][j]+(!(pshSC[binN][j]>pshSCMax))*pshSCMax;
				}
				accessPSHLock.unlock();
			}

			if(dispGRGOAct)
			{
				for(int i=0; i<NUMGR; i++)
				{
					grActs[i]=apOutGR[i];
				}
				for(int i=0; i<NUMGO; i++)
				{
					goActs[i]=apGO[i];
				}
				emit updateActW(grActs, goActs);
			}

			if(dispRaster)
			{
				unsigned char *apPointers[3]={(unsigned char *)apMF, apOutGR, (unsigned char *)apGO};

				simDispTypeLock.lock();
				dispType=simDispType;
				simDispTypeLock.unlock();

				if(i>=csOnset[activeCS] && i<csOnset[activeCS]+csDuration[activeCS])
				{
					emit updateCSBackground(i);
				}

				if(dispType<3)
				{
					for(int j=0; j<NUMMF; j++)
					{
						apRaster[j]=(bool)(apPointers[dispType][j]&0x01);
					}
					emit updateRaster(apRaster, i);
				}
				else if(dispType==3)
				{
					for(int j=0; j<NUMPC; j++)
					{
						sbpActs.apPC[j]=apPC[j];
						sbpActs.vPC[j]=vPC[j];
					}
					for(int j=0; j<NUMBC; j++)
					{
						sbpActs.apBC[j]=apBC[j];
					}
					for(int j=0; j<NUMSC; j++)
					{
						sbpActs.apSC[j]=apSC[j];
					}
					emit updateSCBCPCActs(sbpActs, i);
				}
				else
				{
					for(int j=0; j<NUMPC; j++)
					{
						inpActs.apPC[j]=apPC[j];
						inpActs.vPC[j]=vPC[j];
					}
					for(int j=0; j<NUMIO; j++)
					{
						inpActs.apIO[j]=apIO[j];
						inpActs.vIO[j]=vIO[j];
					}
					for(int j=0; j<NUMNC; j++)
					{
						inpActs.apNC[j]=apNC[j];
						inpActs.vNC[j]=vNC[j];
					}
					emit updateIONCPCActs(inpActs, i);
				}
			}

			simPauseLock.unlock();
		}
		pfSynWeightPCLock.lock();
		cudaMemcpy(pfSynWeightPC, pfSynWeightPCGPU, NUMGR*sizeof(float), cudaMemcpyDeviceToHost);
		pfSynWeightPCLock.unlock();

		cout<<endl<<"trial #"<<nRuns<<" "<<"trial run time: "<<time(NULL)-trialRunTime<<endl;
	}
}
