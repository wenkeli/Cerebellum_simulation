#include <iostream>
#include <fstream>

#include "../includes/simthread.hpp"

using namespace std;

SimThread::SimThread(QObject *parent, int numMZ, int randSeed, string conPF, string actPF, Environment *env)
    : QThread(parent), alive(true), paused(false), trialLength(5000), numMZ(numMZ), env(env)
{
    vector<int> mzoneCRSeeds; // Seed for each MZ's connectivity
    vector<int> mzoneARSeeds; // Seed for each MZ's activity

    // Create MZ seeds from the main random seed
    for (int i=0; i<numMZ; i++) {
        mzoneCRSeeds.push_back(rand());
        mzoneARSeeds.push_back(rand());
    }
    cout << "mzoneCRSeeds: ";
    for (int i=0; i<numMZ; i++)
        cout << mzoneCRSeeds[i] << " ";
    cout << endl;
    cout << "mzoneARSeeds: ";
    for (int i=0; i<numMZ; i++)
        cout << mzoneARSeeds[i] << " ";
    cout << endl;

    // Load the parameter files
    fstream conPStream(conPF.c_str(), fstream::in);
    fstream actPStream(actPF.c_str(), fstream::in);

    // Create the simulation
    simState = new CBMState(actPStream, conPStream, numMZ, randSeed,
                      &mzoneCRSeeds[0], &mzoneARSeeds[0]);
    simCore = new CBMSimCore(simState, &randSeed);

    conPStream.close();
    actPStream.close();

    setupMFs(randSeed);
}

SimThread::SimThread(QObject *parent, int numMZ, int randSeed, std::string savedSimFile, Environment *env)
    : QThread(parent), alive(true), paused(false), trialLength(5000), numMZ(numMZ), env(env)
{
    fstream stateStream(savedSimFile.c_str(), fstream::in);

    // Hack: Ignore the first line of the file because it contains boost serialization stuff
    string line;
    std::getline(stateStream, line);

    // Create the simulation
    simState = new CBMState(stateStream);
    simCore = new CBMSimCore(simState, &randSeed);
    stateStream.close();

    setupMFs(randSeed);
}

SimThread::~SimThread()
{
    delete simState;
    delete simCore;
    delete mfs;
}

void SimThread::setupMFs(int randSeed) {
    // Get the number of cells of each type
    numGR = simState->getConnectivityParams()->getNumGR();
    numGO = simState->getConnectivityParams()->getNumGO();
    numGL = simState->getConnectivityParams()->getNumGL();
    numMF = simState->getConnectivityParams()->getNumMF();
    numSC = simState->getConnectivityParams()->getNumSC();
    numBC = simState->getConnectivityParams()->getNumBC();
    numPC = simState->getConnectivityParams()->getNumPC();
    numNC = simState->getConnectivityParams()->getNumNC();
    numIO = simState->getConnectivityParams()->getNumIO();

    env->setupMossyFibers(simState);

    const float threshDecayTau = 4.0f; // Rate of decay = 1-exp(-msPerTS/threshDecayTau)
    const float msPerTimeStep = 1.0f;
    mfs = new PoissonRegenCells(numMF, randSeed, threshDecayTau, msPerTimeStep);

    inNet = simCore->getInputNet();
    MZoneInterface **mzList = simCore->getMZoneList();
    for (int i=0; i<numMZ; i++)
        mZones.push_back(mzList[i]);

    // Register the data types to be used
    qRegisterMetaType<std::vector<ct_uint8_t> >("std::vector<ct_uint8_t>");
    qRegisterMetaType<std::vector<float> >("std::vector<float>");
    qRegisterMetaType<QColor>("QColor");
}

void SimThread::run()
{
    int simStep = 0;
    while (alive && !env->terminated()) {
        if (paused) {
            usleep(1000);
            continue;
        }

        if (simStep % 10000 == 0) cout << endl;
        if (simStep % 1000 == 0) cout << "." << flush;

        float *mfFreq = env->getState();
        const ct_uint8_t *apMF = mfs->calcActivity(mfFreq);
        simCore->updateMFInput(apMF);
        simCore->calcActivity();
        env->step(simCore);

        // Update the visualizations of all the different views
        vector<ct_uint8_t> apMFVis(apMF, apMF + numGO * sizeof apMF[0]);
        emit(updateINTW(apMFVis, simStep));

        const ct_uint8_t *apSC = inNet->exportAPSC();
        vector<ct_uint8_t> apSCVis(apSC, apSC + numSC * sizeof apSC[0]);
        emit(updateSCTW(apSCVis, simStep));

        for (int mz=0; mz<numMZ; mz++) {
            const ct_uint8_t *apBC = mZones[mz]->exportAPBC();
            vector<ct_uint8_t> apBCVis(apBC, apBC + numBC * sizeof apBC[0]);
            emit(updateBCTW(apBCVis, simStep, mz));

            const ct_uint8_t *apPC = mZones[mz]->exportAPPC();
            const float *vmPC = mZones[mz]->exportVmPC();
            vector<ct_uint8_t> apPCVis(apPC, apPC + numPC * sizeof apPC[0]);
            vector<float> vmPCVis(vmPC, vmPC + numPC * sizeof vmPC[0]);
            for (int i=0; i<numPC; i++)
                vmPCVis[i] = (vmPC[i]+80)/80;
            emit(updatePCTW(apPCVis, vmPCVis, simStep, mz));

            const ct_uint8_t *apNC = mZones[mz]->exportAPNC();
            const float *vmNC = mZones[mz]->exportVmNC();
            vector<ct_uint8_t> apNCVis(apNC, apNC + numNC * sizeof apNC[0]);
            vector<float> vmNCVis(vmNC, vmNC + numNC * sizeof vmNC[0]);
            for (int i=0; i<numNC; i++)
                vmNCVis[i] = (vmNC[i]+80)/80;
            emit(updateNCTW(apNCVis, vmNCVis, simStep, mz));

            const ct_uint8_t *apIO = mZones[mz]->exportAPIO();
            const float *vmIO = mZones[mz]->exportVmIO();
            vector<ct_uint8_t> apIOVis(apIO, apIO + numIO * sizeof apIO[0]);
            vector<float> vmIOVis(vmIO, vmIO + numIO * sizeof vmIO[0]);
            for (int i=0; i<numIO; i++)
                vmIOVis[i] = (vmIO[i]+80)/80;
            emit(updateIOTW(apIOVis, vmIOVis, simStep, mz));
        }

        if (simStep % trialLength == 0)
            emit(blankTW(Qt::black));

        simStep++;
    }
}

void SimThread::disablePlasticity() {
    cout << "Freezing Plasticity." << endl;
    ActivityParams *actParams = simState->getActParamsInternal();
    actParams->synLTPStepSizeGRtoPC = 0;
    actParams->synLTDStepSizeGRtoPC = 0;
    actParams->synLTDStepSizeMFtoNC = 0;
    actParams->synLTPStepSizeMFtoNC = 0;
}

void SimThread::saveSimState(string saveFile) {
    ofstream ofs(saveFile.c_str());
    {
        boost::archive::text_oarchive oa(ofs);
        oa << (*env);
    }
    
    std::fstream filestr(saveFile.c_str(), fstream::out | fstream::app);
    simCore->writeToState(filestr);
    filestr.close();
}
