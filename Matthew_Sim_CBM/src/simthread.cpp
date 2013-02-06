#include <iostream>
#include <fstream>

#include "../includes/simthread.hpp"

using namespace std;

SimThread::SimThread(QObject *parent, int numMZ, int randSeed, string conPF, string actPF, Environment *env)
    : QThread(parent), alive(true), trialLength(5000), numMZ(numMZ), env(env)
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
    mZone = simCore->getMZoneList()[0];

    // Register the data types to be used
    qRegisterMetaType<std::vector<ct_uint8_t> >("std::vector<ct_uint8_t>");
    qRegisterMetaType<std::vector<float> >("std::vector<float>");
    qRegisterMetaType<QColor>("QColor");
}

SimThread::~SimThread()
{
    delete simState;
    delete simCore;
    delete mfs;
}

void SimThread::run()
{
    for (int simStep=0; alive && !env->terminated(); simStep++) {
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

        const ct_uint8_t *apBC = mZone->exportAPBC();
        vector<ct_uint8_t> apBCVis(apBC, apBC + numBC * sizeof apBC[0]);
        emit(updateBCTW(apBCVis, simStep));

        const ct_uint8_t *apPC = mZone->exportAPPC();
        const float *vmPC = mZone->exportVmPC();
        vector<ct_uint8_t> apPCVis(apPC, apPC + numPC * sizeof apPC[0]);
        vector<float> vmPCVis(vmPC, vmPC + numPC * sizeof vmPC[0]);
        for (int i=0; i<numPC; i++)
            vmPCVis[i] = (vmPC[i]+80)/80;
        emit(updatePCTW(apPCVis, vmPCVis, simStep));

        const ct_uint8_t *apNC = mZone->exportAPNC();
        const float *vmNC = mZone->exportVmNC();
        vector<ct_uint8_t> apNCVis(apNC, apNC + numNC * sizeof apNC[0]);
        vector<float> vmNCVis(vmNC, vmNC + numNC * sizeof vmNC[0]);
        for (int i=0; i<numNC; i++)
            vmNCVis[i] = (vmNC[i]+80)/80;
        emit(updateNCTW(apNCVis, vmNCVis, simStep));

        const ct_uint8_t *apIO = mZone->exportAPIO();
        const float *vmIO = mZone->exportVmIO();
        vector<ct_uint8_t> apIOVis(apIO, apIO + numIO * sizeof apIO[0]);
        vector<float> vmIOVis(vmIO, vmIO + numIO * sizeof vmIO[0]);
        for (int i=0; i<numIO; i++)
            vmIOVis[i] = (vmIO[i]+80)/80;
        emit(updateIOTW(apIOVis, vmIOVis, simStep));

        if (simStep % trialLength == 0)
            emit(blankTW(Qt::black));
    }
}
