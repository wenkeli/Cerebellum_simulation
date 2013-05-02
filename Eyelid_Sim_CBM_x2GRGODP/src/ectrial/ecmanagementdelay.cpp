/*
 * ecmanagementdelay.cpp
 *
 *  Created on: Sep 4, 2012
 *      Author: consciousness
 */

#include "../../includes/ectrial/ecmanagementdelay.h"
using namespace std;

ECManagementDelay::ECManagementDelay(string conParamFile, string actParamFile, string actParamFile1,
		int randSeed, int numT, int iti, int csOn, int csOff, int csPOff,
		int csStartTN, int dataStartTN, int nDataT,
		float fracCSTMF, float fracCSPMF, float fracCtxtMF,
		float bgFreqMin, float csBGFreqMin, float ctxtFreqMin, float csTFreqMin, float csPFreqMin,
		float bgFreqMax, float csBGFreqMax, float ctxtFreqMax, float csTFreqMax, float csPFreqMax)
		:ECManagementBase(conParamFile, actParamFile, actParamFile1, numT, iti, randSeed)
{
	CRandomSFMT0 randGen(randSeed);

	int numCSTMF;
	int numCSPMF;
	int numCtxtMF;

	bool *isCSTonic;
	bool *isCSPhasic;
	bool *isContext;

	rSeed=randSeed;

	csOnTime=csOn;
	csOffTime=csOff;
	csPOffTime=csPOff;

	csStartTrialN=csStartTN;
	dataStartTrialN=dataStartTN;
	numDataTrials=nDataT;

	fracCSTonicMF=fracCSTMF;
	fracCSPhasicMF=fracCSPMF;
	fracContextMF=fracCtxtMF;

	backGFreqMin=bgFreqMin;
	csBackGFreqMin=csBGFreqMin;
	contextFreqMin=ctxtFreqMin;
	csTonicFreqMin=csTFreqMin;
	csPhasicFreqMin=csPFreqMin;

	backGFreqMax=bgFreqMax;
	csBackGFreqMax=csBGFreqMax;
	contextFreqMax=ctxtFreqMax;
	csTonicFreqMax=csTFreqMax;
	csPhasicFreqMax=csPFreqMax;

	mfs=new PoissonRegenCells(numMF, rSeed, 4, 1);

	mfFreqBG=new float[numMF];
	mfFreqInCSTonic=new float[numMF];
	mfFreqInCSPhasic=new float[numMF];


	isCSTonic=new bool[numMF];
	isCSPhasic=new bool[numMF];
	isContext=new bool[numMF];

	for(int i=0; i<numMF; i++)
	{
		mfFreqBG[i]=randGen.Random()*(backGFreqMax-backGFreqMin)+backGFreqMin;
		mfFreqInCSTonic[i]=mfFreqBG[i];
		mfFreqInCSPhasic[i]=mfFreqBG[i];

		isCSTonic[i]=false;
		isCSPhasic[i]=false;
		isContext[i]=false;
	}

	numCSTMF=fracCSTonicMF*numMF;
	numCSPMF=fracCSPhasicMF*numMF;
	numCtxtMF=fracContextMF*numMF;

	for(int i=0; i<numCSTMF; i++)
	{
		while(true)
		{
			int mfInd;

			mfInd=randGen.IRandom(0, numMF-1);

			if(isCSTonic[mfInd])
			{
				continue;
			}

			isCSTonic[mfInd]=true;
			break;
		}
	}

	for(int i=0; i<numCSPMF; i++)
	{
		while(true)
		{
			int mfInd;

			mfInd=randGen.IRandom(0, numMF-1);

			if(isCSPhasic[mfInd] || isCSTonic[mfInd])
			{
				continue;
			}

			isCSPhasic[mfInd]=true;
			break;
		}
	}

	for(int i=0; i<numCtxtMF; i++)
	{
		while(true)
		{
			int mfInd;

			mfInd=randGen.IRandom(0, numMF-1);

			if(isContext[mfInd] || isCSPhasic[mfInd] || isCSTonic[mfInd])
			{
				continue;
			}

			isContext[mfInd]=true;
			break;
		}
	}

	for(int i=0; i<numMF; i++)
	{
		if(isContext[i])
		{
			mfFreqBG[i]=randGen.Random()*(contextFreqMax-contextFreqMin)+contextFreqMin;
			mfFreqInCSTonic[i]=mfFreqBG[i];
			mfFreqInCSPhasic[i]=mfFreqBG[i];
		}

		if(isCSTonic[i])
		{
			mfFreqInCSTonic[i]=randGen.Random()*(csTonicFreqMax-csTonicFreqMin)+csTonicFreqMin;
			mfFreqInCSPhasic[i]=mfFreqInCSTonic[i];
		}

		if(isCSPhasic[i])
		{
			mfFreqInCSPhasic[i]=randGen.Random()*(csPhasicFreqMax-csPhasicFreqMin)+csPhasicFreqMin;
		}
	}

	simState->getActivityParams(0)->showParams(cout);
	simState->getConnectivityParams()->showParams(cout);

	eyelidFunc=new EyelidIntegrator(simState->getConnectivityParams()->getNumNC(),
			simState->getActivityParams(0)->getMSPerTimeStep(), 10.5, 0.012, -0.13, 0.13, 100);

	{
		EyelidOutParams eyelidParams;
		PSHParams pp;
		RasterParams rp;
		map<string, PSHParams> pshParams;
		map<string, RasterParams> rasterParams;

		eyelidParams.numTimeStepSmooth=4;

		pp.numCells=simState->getConnectivityParams()->getNumGO();
		pp.numTimeStepsPerBin=10;
		pshParams["go0"]=pp;
//		pp.numCells=simState->getConnectivityParams()->getNumGR();
//		pshParams["gr0"]=pp;
		pp.numCells=simState->getConnectivityParams()->getNumSC();
		pshParams["sc0"]=pp;
		pp.numCells=simState->getConnectivityParams()->getNumBC();
		pshParams["bc0"]=pp;
		pp.numCells=simState->getConnectivityParams()->getNumPC();
		pshParams["pc0"]=pp;
		pp.numCells=simState->getConnectivityParams()->getNumIO();
		pshParams["io0"]=pp;

		pp.numCells=simState->getConnectivityParams()->getNumGO();
		pp.numTimeStepsPerBin=10;
		pshParams["go1"]=pp;
//		pp.numCells=simState->getConnectivityParams()->getNumGR();
//		pshParams["gr1"]=pp;
		pp.numCells=simState->getConnectivityParams()->getNumSC();
		pshParams["sc1"]=pp;
		pp.numCells=simState->getConnectivityParams()->getNumBC();
		pshParams["bc1"]=pp;
		pp.numCells=simState->getConnectivityParams()->getNumPC();
		pshParams["pc1"]=pp;
		pp.numCells=simState->getConnectivityParams()->getNumIO();
		pshParams["io1"]=pp;

		rasterParams.clear();
		rp.numCells=simState->getConnectivityParams()->getNumGO();
		rasterParams["go0"]=rp;
		rp.numCells=simState->getConnectivityParams()->getNumSC();
		rasterParams["sc0"]=rp;
		rp.numCells=simState->getConnectivityParams()->getNumBC();
		rasterParams["bc0"]=rp;
		rp.numCells=simState->getConnectivityParams()->getNumPC();
		rasterParams["pc0"]=rp;
		rp.numCells=simState->getConnectivityParams()->getNumNC();
		rasterParams["nc0"]=rp;
		rp.numCells=simState->getConnectivityParams()->getNumIO();
		rasterParams["io0"]=rp;

		rp.numCells=simState->getConnectivityParams()->getNumGO();
		rasterParams["go1"]=rp;
		rp.numCells=simState->getConnectivityParams()->getNumSC();
		rasterParams["sc1"]=rp;
		rp.numCells=simState->getConnectivityParams()->getNumBC();
		rasterParams["bc1"]=rp;
		rp.numCells=simState->getConnectivityParams()->getNumPC();
		rasterParams["pc1"]=rp;
		rp.numCells=simState->getConnectivityParams()->getNumNC();
		rasterParams["nc1"]=rp;
		rp.numCells=simState->getConnectivityParams()->getNumIO();
		rasterParams["io1"]=rp;


		data=new ECTrialsData(500, csOff-csOn, 500, simState->getActivityParams(0)->getMSPerTimeStep(),
				numDataTrials, pshParams, rasterParams, eyelidParams);
	}

	grPCPlastSet=false;
	grPCPlastReset=true;

	gIncGRtoGOSet=false;
	gIncGRtoGOReset=true;

	delete[] isCSTonic;
	delete[] isCSPhasic;
	delete[] isContext;
}

ECManagementDelay::~ECManagementDelay()
{
	delete mfs;
	delete[] mfFreqBG;
	delete[] mfFreqInCSTonic;
	delete[] mfFreqInCSPhasic;

	delete data;
}

void ECManagementDelay::calcMFActivity()
{
	if(currentTrial<csStartTrialN)
	{
		apMF=mfs->calcActivity(mfFreqBG);

		return;
	}

	if(currentTime>=csOnTime && currentTime<csOffTime)
	{
		if(currentTime<csPOffTime)
		{
			apMF=mfs->calcActivity(mfFreqInCSPhasic);
		}
		else
		{
			apMF=mfs->calcActivity(mfFreqInCSTonic);
		}
	}
	else
	{
		apMF=mfs->calcActivity(mfFreqBG);
	}
}

void ECManagementDelay::calcSimActivity()
{
	float eyelidPos;

	if(currentTime==(csOffTime-1) && currentTrial>=csStartTrialN)
	{
		simulation->updateErrDrive(1.0);
	}
	else
	{
//		simulation->updateErrDrive(0, 0);
	}

	if(currentTime>=csOnTime && !gIncGRtoGOSet &&currentTrial>=csStartTrialN
			&& currentTime<csOffTime)
	{
		gIncGRtoGOSet=true;
		gIncGRtoGOReset=false;
		simulation->getInputNetList()[1]->setGIncGRtoGO(0.000026);
	}

	if(!gIncGRtoGOReset && currentTime>=csOffTime && currentTrial>=csStartTrialN)
	{
		gIncGRtoGOSet=false;
		gIncGRtoGOReset=true;
		simulation->getInputNetList()[1]->resetGIncGRtoGO();
	}

	if(currentTime>=csOnTime+200 && !grPCPlastSet
			&& currentTrial>=csStartTrialN && currentTime<csOffTime+200)
	{
		grPCPlastSet=true;
		grPCPlastReset=false;
		simulation->getMZoneList()[0]->setGRPCPlastSteps(-0.0005f*((float)(csOffTime-csOnTime)-200)/100.0f, 0.0005f);
		simulation->getMZoneList()[1]->setGRPCPlastSteps(-0.0005f*((float)(csOffTime-csOnTime)-200)/100.0f, 0.0005f);
	}

	if(!grPCPlastReset && currentTime>=csOffTime+200 && currentTrial>=csStartTrialN)
	{
		grPCPlastSet=false;
		grPCPlastReset=true;
		simulation->getMZoneList()[0]->resetGRPCPlastSteps();
		simulation->getMZoneList()[1]->resetGRPCPlastSteps();
	}

	simulation->updateMFInput(apMF);

	simulation->calcActivity();


	eyelidPos=eyelidFunc->calcStep(simulation->getMZoneList()[0]->exportAPNC());

	if(currentTime>=csOnTime-500 && currentTime<csOffTime+500
			&& currentTrial>=dataStartTrialN && currentTrial<dataStartTrialN+numDataTrials)
	{
		int ct=currentTime-(csOnTime-500);
		if(data->getTSPerRasterUpdate()>0)
		{
			if(ct%data->getTSPerRasterUpdate()==0 &&ct>0)
			{
				data->updateRaster("go0", simulation->getInputNetList()[0]->exportAPBufGO());
				data->updateRaster("sc0", simulation->getInputNetList()[0]->exportAPBufSC());
				data->updateRaster("bc0", simulation->getMZoneList()[0]->exportAPBufBC());
				data->updateRaster("pc0", simulation->getMZoneList()[0]->exportAPBufPC());
				data->updateRaster("nc0", simulation->getMZoneList()[0]->exportAPBufNC());
				data->updateRaster("io0", simulation->getMZoneList()[0]->exportAPBufIO());

				data->updateRaster("go1", simulation->getInputNetList()[1]->exportAPBufGO());
				data->updateRaster("sc1", simulation->getInputNetList()[1]->exportAPBufSC());
				data->updateRaster("bc1", simulation->getMZoneList()[1]->exportAPBufBC());
				data->updateRaster("pc1", simulation->getMZoneList()[1]->exportAPBufPC());
				data->updateRaster("nc1", simulation->getMZoneList()[1]->exportAPBufNC());
				data->updateRaster("io1", simulation->getMZoneList()[1]->exportAPBufIO());
			}
		}

		data->updatePSH("go0", simulation->getInputNetList()[0]->exportAPGO());
//		data->updatePSH("gr0", simulation->getInputNetList()[0]->exportAPGR());
		data->updatePSH("sc0", simulation->getInputNetList()[0]->exportAPSC());
		data->updatePSH("bc0", simulation->getMZoneList()[0]->exportAPBC());
		data->updatePSH("pc0", simulation->getMZoneList()[0]->exportAPPC());
		data->updatePSH("io0", simulation->getMZoneList()[0]->exportAPIO());

		data->updatePSH("go1", simulation->getInputNetList()[1]->exportAPGO());
//		data->updatePSH("gr1", simulation->getInputNetList()[1]->exportAPGR());
		data->updatePSH("sc1", simulation->getInputNetList()[1]->exportAPSC());
		data->updatePSH("bc1", simulation->getMZoneList()[1]->exportAPBC());
		data->updatePSH("pc1", simulation->getMZoneList()[1]->exportAPPC());
		data->updatePSH("io1", simulation->getMZoneList()[1]->exportAPIO());

		data->updateEyelid(eyelidPos);
	}

}

void ECManagementDelay::writeDataToFile()
{
	fstream dataOut;

	dataOut.open("dataOut", ios::out|ios::binary);
	data->writeData(dataOut);
	dataOut.close();

}
