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
	mfFreqs=new ECMFPopulation(numMF, randSeed, fracCSTMF, fracCSPMF, fracCtxtMF,
			bgFreqMin, csBGFreqMin, ctxtFreqMin, csTFreqMin, csPFreqMin,
			bgFreqMax, csBGFreqMax, ctxtFreqMax, csTFreqMax, csPFreqMax);

	initialize(randSeed, csOn, csOff, csPOff, csStartTN, dataStartTN, nDataT, dataFileName);
}

ECManagementDelay::ECManagementDelay(string stateDataFile, int randSeed,
		int numT, int iti, int csOn, int csOff, int csPOff,
		int csStartTN, int dataStartTN, int nDataT,
		string dataFileName, int gpuIndStart, int numGPUP2)
		:ECManagementBase(stateDataFile, numT, iti, randSeed,
			gpuIndStart, numGPUP2)
{
	fstream sdFile;

	CBMState *dummyState;
	ECTrialsData *prevData;

	sdFile.open(stateDataFile.c_str(), ios::in|ios::binary);

	dummyState=new CBMState(sdFile);
	prevData=new ECTrialsData(sdFile);

	mfFreqs=new ECMFPopulation(prevData);
	initialize(randSeed, csOn, csOff, csPOff, csStartTN, dataStartTN, nDataT, dataFileName);

	delete dummyState;
	delete prevData;
	sdFile.close();
}

ECManagementDelay::ECManagementDelay(string stateDataFN, string mfFN, int randSeed,
		int numT, int iti, int csOn, int csOff, int csPOff,
		int csStartTN, int dataStartTN, int nDataT,
		string dataFileName, int gpuIndStart, int numGPUP2)
		:ECManagementBase(stateDataFN, numT, iti, randSeed,
			gpuIndStart, numGPUP2)
{
	fstream sdFile;
	fstream mfFile;

	sdFile.open(stateDataFN.c_str(), ios::in|ios::binary);
	mfFile.open(mfFN.c_str(), ios::in|ios::binary);

	mfFreqs=new ECMFPopulation(mfFile);
	initialize(randSeed, csOn, csOff, csPOff, csStartTN, dataStartTN, nDataT, dataFileName);

	sdFile.close();
	mfFile.close();
}

void ECManagementDelay::initialize(int randSeed, int csOn, int csOff, int csPOff,
		int csStartTN, int dataStartTN, int nDataT, string dataFileName)
{
	rSeed=randSeed;

	csOnTime=csOn;
	csOffTime=csOff;
	csPOffTime=csPOff;

	csStartTrialN=csStartTN;
	dataStartTrialN=dataStartTN;
	numDataTrials=nDataT;

	mfs=new PoissonRegenCells(numMF, randSeed, 4, 1);

	mfFreqBG=mfFreqs->getMFBG();
	mfFreqInCSTonic=mfFreqs->getMFInCSTonic();
	mfFreqInCSPhasic=mfFreqs->getMFFreqInCSPhasic();

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
}

ECManagementDelay::~ECManagementDelay()
{
	delete mfs;
	delete mfFreqs;

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

	//TODO: disable gr pc plast, make sure to unset this
//	simulation->getMZoneList()[0]->setGRPCPlastSteps(0, 0);
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
		simulation->getMZoneList()[0]->setGRPCPlastSteps(-0.0005f*((float)(csOffTime-csOnTime)-200)/100.0f, 0.0005f);
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
	string mfFN;
	fstream dataOut;

	mfFN=dataFileName+"_MFPop";

	dataOut.open(dataFileName.c_str(), ios::out|ios::binary);
	simulation->writeToState(dataOut);
	data->writeData(dataOut);
	dataOut.close();

	dataOut.open(mfFN.c_str(), ios::out);
	mfFreqs->writeToFile(dataOut);
	dataOut.close();

}
