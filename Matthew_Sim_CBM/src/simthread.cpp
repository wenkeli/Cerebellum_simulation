#include <iostream>
#include <fstream>
#include <algorithm>

#include "../includes/simthread.h"

using namespace std;

SimThread::SimThread(QObject *parent, int numMZ, int randSeed, string conPF, string actPF)
    : QThread(parent), alive(true), running(true), trialLength(5000), numMZ(numMZ)
{
    if (randSeed >= 0) {
        cout << "Using random seed: " << randSeed << endl;
        srand(randSeed);
    } else {
        randSeed = time(NULL);
        cout << "No seed specified. Seeding to time. Seed: " << randSeed << endl;
        srand(randSeed);
    }

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

    randGen = new CRandomSFMT0(randSeed);

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

    setupMossyFibers(randSeed);

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

void SimThread::setupMossyFibers(int randSeed)
{
    const float threshDecayTau = 4.0f; // Rate of decay = 1-exp(-msPerTS/threshDecayTau)
    const float msPerTimeStep = 1.0f;
    mfs = new PoissonRegenCells(numMF, randSeed, threshDecayTau, msPerTimeStep);
    mfFreq.resize(numMF);
    mfFreqRelaxed.resize(numMF);
    mfFreqExcited.resize(numMF);

    for(int i=0; i<numMF; i++) {
        const float backGFreqMin = 1;
        const float backGFreqMax = 10;
        mfFreqRelaxed[i]=randGen->Random()*(backGFreqMax-backGFreqMin)+backGFreqMin;
    }

    vector<int> mfInds(numMF);
    for (int i=0; i<numMF; i++)
        mfInds[i] = i;
    std::random_shuffle(mfInds.begin(), mfInds.end());

    const int numContextMF = numMF * .03;

    for (int i=0; i<numContextMF; i++) {
        const float contextFreqMin = 30;
        const float contextFreqMax = 60;
        mfFreqRelaxed[mfInds.back()]=randGen->Random()*(contextFreqMax-contextFreqMin)+contextFreqMin;
        mfInds.pop_back();
    }

    for (int i=0; i<numMF; i++) {
        const float excitedFreqMin = 30;
        const float excitedFreqMax = 60;
        mfFreqExcited[i]=randGen->Random()*(excitedFreqMax-excitedFreqMax)+excitedFreqMin;
        mfExcited.push_back(false);
    }
}

void SimThread::run()
{
    for (int simStep=0; alive; simStep++) {
        if (simStep % 10000 == 0) cout << endl;
        if (simStep % 1000 == 0) cout << "." << flush;

        for (int i=0; i<numMF; i++) {
            mfFreq[i] = mfExcited[i] ? mfFreqExcited[i] : mfFreqRelaxed[i];
        }

        const ct_uint8_t *apMF = mfs->calcActivity(&mfFreq[0]);
        simCore->updateMFInput(apMF);
        simCore->calcActivity();

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
