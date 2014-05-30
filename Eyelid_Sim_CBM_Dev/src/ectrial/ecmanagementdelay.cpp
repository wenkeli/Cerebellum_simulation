/*
 * ecmanagementdelay.cpp
 *
 *  Created on: Sep 4, 2012
 *      Author: consciousness
 */

#include "../../includes/ectrial/ecmanagementdelay.h"
using namespace std;

ECManagementDelay::ECManagementDelay(string conParamFile, string actParamFile, int randSeed,
		int numT, int iti, int csOn, int csOff, int csPOff,
		int csStartTN, int dataStartTN, int nDataT,
		float fracCSTMF, float fracCSPMF, float fracCtxtMF,
		float bgFreqMin, float csBGFreqMin, float ctxtFreqMin, float csTFreqMin, float csPFreqMin,
		float bgFreqMax, float csBGFreqMax, float ctxtFreqMax, float csTFreqMax, float csPFreqMax,
		string dataFileName, int gpuIndStart, int numGPUP2)
		:ECManagementBase(conParamFile, actParamFile, numT, iti, randSeed, gpuIndStart, numGPUP2)
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

	simState->getActivityParams()->showParams(cout);
	simState->getConnectivityParams()->showParams(cout);

	eyelidFunc=new EyelidIntegrator(simState->getConnectivityParams()->getNumNC(),
			simState->getActivityParams()->getMSPerTimeStep(), 10.5, 0.012, -0.13, 0.13, 100);

	{
		EyelidOutParams eyelidParams;
		PSHParams pp;
		RasterParams rp;
		RawUIntParams ruip;
		map<string, PSHParams> pshParams;
		map<string, RasterParams> rasterParams;
		map<string, RawUIntParams> uintParams;

		eyelidParams.numTimeStepSmooth=4;

		pp.numCells=simState->getConnectivityParams()->getNumGO();
		pp.numTimeStepsPerBin=10;
		pshParams["go"]=pp;
		pshParams["grInputGO"]=pp;
		pshParams["goInputGO"]=pp;
		pshParams["goOutSynScale"]=pp;
		pp.numCells=simState->getConnectivityParams()->getNumMF();
		pshParams["mf"]=pp;
		pp.numCells=simState->getConnectivityParams()->getNumGR()/10;
		pshParams["gr"]=pp;
		pp.numCells=simState->getConnectivityParams()->getNumSC();
		pshParams["sc"]=pp;
		pp.numCells=simState->getConnectivityParams()->getNumBC();
		pshParams["bc"]=pp;
		pp.numCells=simState->getConnectivityParams()->getNumPC();
		pshParams["pc"]=pp;
		pp.numCells=simState->getConnectivityParams()->getNumIO();
		pshParams["io"]=pp;

		rasterParams.clear();
		rp.numCells=simState->getConnectivityParams()->getNumGO();
		rasterParams["go"]=rp;
		rp.numCells=simState->getConnectivityParams()->getNumSC();
		rasterParams["sc"]=rp;
		rp.numCells=simState->getConnectivityParams()->getNumBC();
		rasterParams["bc"]=rp;
		rp.numCells=simState->getConnectivityParams()->getNumPC();
		rasterParams["pc"]=rp;
		rp.numCells=simState->getConnectivityParams()->getNumNC();
		rasterParams["nc"]=rp;
		rp.numCells=simState->getConnectivityParams()->getNumIO();
		rasterParams["io"]=rp;

		uintParams.clear();
//		ruip.numRows=simState->getConnectivityParams()->getNumGO();
//		uintParams["grInputGO"]=ruip;


		data=new ECTrialsData(500, csOff-csOn, 500, simState->getActivityParams()->getMSPerTimeStep(),
				numDataTrials, pshParams, rasterParams, uintParams, eyelidParams);

		this->dataFileName=dataFileName;
	}

	grPCPlastSet=false;
	grPCPlastReset=true;

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
		simulation->updateErrDrive(0, 1.0);
	}
	else
	{
//		simulation->updateErrDrive(0, 0);
	}

	if(currentTime>=csOnTime+200 && !grPCPlastSet
			&& currentTrial>=csStartTrialN && currentTime<csOffTime+200)
	{
		grPCPlastSet=true;
		grPCPlastReset=false;
//		simulation->getMZoneList()[0]->setGRPCPlastSteps(-0.0001f*((float)(csOffTime-csOnTime)-200)/100.0f, 0.0001f);
	}

	if(!grPCPlastReset && currentTime>=csOffTime+200 && currentTrial>=csStartTrialN)
	{
		grPCPlastSet=false;
		grPCPlastReset=true;
		simulation->getMZoneList()[0]->resetGRPCPlastSteps();
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
				data->updateRaster("go", simulation->getInputNet()->exportAPBufGO());
				data->updateRaster("sc", simulation->getInputNet()->exportAPBufSC());
				data->updateRaster("bc", simulation->getMZoneList()[0]->exportAPBufBC());
				data->updateRaster("pc", simulation->getMZoneList()[0]->exportAPBufPC());
				data->updateRaster("nc", simulation->getMZoneList()[0]->exportAPBufNC());
				data->updateRaster("io", simulation->getMZoneList()[0]->exportAPBufIO());
			}
		}

		data->updatePSH("go", simulation->getInputNet()->exportAPGO());
		data->updatePSH("grInputGO", simulation->getInputNet()->exportSumGRInputGO());
		data->updatePSH("goInputGO", simulation->getInputNet()->exportSumGOInputGO());
		data->updatePSH("goOutSynScale", simulation->getInputNet()->exportGOOutSynScaleGOGO());
		data->updatePSH("mf", apMF);
		data->updatePSH("gr", simulation->getInputNet()->exportAPGR());
		data->updatePSH("sc", simulation->getInputNet()->exportAPSC());
		data->updatePSH("bc", simulation->getMZoneList()[0]->exportAPBC());
		data->updatePSH("pc", simulation->getMZoneList()[0]->exportAPPC());
		data->updatePSH("io", simulation->getMZoneList()[0]->exportAPIO());

//		data->updateRawUInt("grInputGO", simulation->getInputNet()->exportSumGRInputGO());

		data->updateEyelid(eyelidPos);
	}

}

void ECManagementDelay::writeDataToFile()
{
	fstream dataOut;

	dataOut.open(dataFileName.c_str(), ios::out|ios::binary);
	simulation->writeToState(dataOut);
	data->writeData(dataOut);
	dataOut.close();

}
