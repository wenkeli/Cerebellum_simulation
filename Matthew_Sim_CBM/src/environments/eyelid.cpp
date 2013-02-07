#include <iostream>

#include "../../includes/environments/eyelid.hpp"
#include "../../includes/environments/environment.hpp"

using namespace std;

Eyelid::Eyelid(CRandomSFMT0 *randGen) : Environment(randGen) {

}

Eyelid::~Eyelid() {
    delete[] mfFreqBG;
    delete[] mfFreqInCSTonic;
    delete[] mfFreqInCSPhasic;

    delete data;
}

void Eyelid::setupMossyFibers(CBMState *simState) {
    int numMF = simState->getConnectivityParams()->getNumMF();
    int numCSTMF;
    int numCSPMF;
    int numCtxtMF;

    bool *isCSTonic;
    bool *isCSPhasic;
    bool *isContext;

    numTrials=20;
    interTrialI=5000;

    cerr<<"numTrials: "<<numTrials<<" iti:"<<interTrialI<<endl;
        
    currentTrial=0;
    currentTime=-1;

    csOnTime=2000;
    csOffTime=2750;
    csPOffTime=2040;

    csStartTrialN=5;
    dataStartTrialN=5;
    numDataTrials=10;

    fracCSTonicMF=0.025;
    fracCSPhasicMF=0.03;
    fracContextMF=0.03;

    backGFreqMin=1;
    csBackGFreqMin=1;
    contextFreqMin=30;
    csTonicFreqMin=40;
    csPhasicFreqMin=120;

    backGFreqMax=10;
    csBackGFreqMax=5;
    contextFreqMax=60;
    csTonicFreqMax=50;
    csPhasicFreqMax=130;

    mfFreqBG=new float[numMF];
    mfFreqInCSTonic=new float[numMF];
    mfFreqInCSPhasic=new float[numMF];

    isCSTonic=new bool[numMF];
    isCSPhasic=new bool[numMF];
    isContext=new bool[numMF];

    for(int i=0; i<numMF; i++)
    {
        mfFreqBG[i]=randGen->Random()*(backGFreqMax-backGFreqMin)+backGFreqMin;
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

            mfInd=randGen->IRandom(0, numMF-1);

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

            mfInd=randGen->IRandom(0, numMF-1);

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

            mfInd=randGen->IRandom(0, numMF-1);

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
            mfFreqBG[i]=randGen->Random()*(contextFreqMax-contextFreqMin)+contextFreqMin;
            mfFreqInCSTonic[i]=mfFreqBG[i];
            mfFreqInCSPhasic[i]=mfFreqBG[i];
        }

        if(isCSTonic[i])
        {
            mfFreqInCSTonic[i]=randGen->Random()*(csTonicFreqMax-csTonicFreqMin)+csTonicFreqMin;
            mfFreqInCSPhasic[i]=mfFreqInCSTonic[i];
        }

        if(isCSPhasic[i])
        {
            mfFreqInCSPhasic[i]=randGen->Random()*(csPhasicFreqMax-csPhasicFreqMin)+csPhasicFreqMin;
        }
    }

    simState->getActivityParams()->showParams(cout);
    simState->getConnectivityParams()->showParams(cout);

    eyelidFunc=new EyelidIntegrator(simState->getConnectivityParams()->getNumNC(),
                                    simState->getActivityParams()->getMSPerTimeStep(), 11, 0.012, 0.1, 100);

    {
        EyelidOutParams eyelidParams;
        PSHParams pp;
        RasterParams rp;
        map<string, PSHParams> pshParams;
        map<string, RasterParams> rasterParams;

        eyelidParams.numTimeStepSmooth=4;

        pp.numCells=simState->getConnectivityParams()->getNumGO();
        pp.numTimeStepsPerBin=10;
        pshParams["go"]=pp;
        pp.numCells=simState->getConnectivityParams()->getNumSC();
        pshParams["sc"]=pp;
        pp.numCells=simState->getConnectivityParams()->getNumBC();
        pshParams["bc"]=pp;
        pp.numCells=simState->getConnectivityParams()->getNumPC();
        pshParams["pc"]=pp;

        rp.numCells=simState->getConnectivityParams()->getNumGO();
        rasterParams["go"]=rp;
        rp.numCells=simState->getConnectivityParams()->getNumSC();
        rasterParams["sc"]=rp;
        rp.numCells=simState->getConnectivityParams()->getNumBC();
        rasterParams["bc"]=rp;
        rp.numCells=simState->getConnectivityParams()->getNumPC();
        rasterParams["pc"]=rp;


        data=new ECTrialsData(500, csOffTime-csOnTime, 500, simState->getActivityParams()->getMSPerTimeStep(),
                              numDataTrials, pshParams, rasterParams, eyelidParams);
    }

    delete[] isCSTonic;
    delete[] isCSPhasic;
    delete[] isContext;
}

float* Eyelid::getState() {
    if(currentTrial<csStartTrialN)
    {
        return mfFreqBG;
    }

    if(currentTime>=csOnTime && currentTime<csOffTime)
    {
        if(currentTime<csPOffTime)
        {
            return mfFreqInCSPhasic;
        }
        else
        {
            return mfFreqInCSTonic;
        }
    }
    else
    {
        return mfFreqBG;
    }

    return NULL;
}

void Eyelid::step(CBMSimCore *simCore) {
    currentTime++;

    if(currentTime>=interTrialI)
    {
        currentTime=0;
        currentTrial++;
    }

    float eyelidPos;
    
    if(currentTime==(csOffTime-1) && currentTrial>=csStartTrialN)
    {
        simCore->updateErrDrive(0, 1.0);
    }
    else
    {
//		simulation->updateErrDrive(0, 0);
    }

    eyelidPos=eyelidFunc->calcStep(simCore->getMZoneList()[0]->exportAPNC());

    if(currentTime>=csOnTime-500 && currentTime<csOffTime+500
       && currentTrial>=dataStartTrialN && currentTrial<dataStartTrialN+numDataTrials)
    {
        int ct=currentTime-(csOnTime-500);
        if(data->getTSPerRasterUpdate()>0)
        {
            if(ct%data->getTSPerRasterUpdate()==0 &&ct>0)
            {
                data->updateRaster("go", simCore->getInputNet()->exportAPBufGO());
                data->updateRaster("sc", simCore->getInputNet()->exportAPBufSC());
                data->updateRaster("bc", simCore->getMZoneList()[0]->exportAPBufBC());
                data->updateRaster("pc", simCore->getMZoneList()[0]->exportAPBufPC());
            }
        }

        data->updatePSH("go", simCore->getInputNet()->exportAPGO());
        data->updatePSH("sc", simCore->getInputNet()->exportAPSC());
        data->updatePSH("bc", simCore->getMZoneList()[0]->exportAPBC());
        data->updatePSH("pc", simCore->getMZoneList()[0]->exportAPPC());

        data->updateEyelid(eyelidPos);
    }
}

bool Eyelid::terminated() {
    return currentTrial>=numTrials;
}
