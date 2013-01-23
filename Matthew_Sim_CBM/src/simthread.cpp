#include <iostream>
#include <fstream>

#include "../includes/simthread.h"

using namespace std;

SimThread::SimThread(QObject *parent, int numMZ, int randSeed, string conPF, string actPF)
    : QThread(parent), running(true), numMZ(numMZ)
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
    ifstream conPStream(conPF.c_str());
    ifstream actPStream(actPF.c_str());

    // Create the simulation
    simState = new CBMState(actPStream, conPStream, numMZ, randSeed,
                      &mzoneCRSeeds[0], &mzoneARSeeds[0]);
    simCore = new CBMSimCore(simState, &randSeed);

    conPStream.close();
    actPStream.close();

    // Setup the Mossy Fibers
    numMF = simState->getConnectivityParams()->getNumMF();
    float threshDecayTau = 4.0f; // Rate of decay = 1-exp(-msPerTS/threshDecayTau)
    float msPerTimeStep = 1.0f;
    mfs = new PoissonRegenCells(numMF, randSeed, threshDecayTau, msPerTimeStep);
    for (int i=0; i<numMF; i++)
        mfFreq.push_back(0); // MF Firings Frequencies

    numGO = simState->getConnectivityParams()->getNumGO();
}

SimThread::~SimThread()
{
    delete simState;
    delete simCore;
    delete mfs;
}

void SimThread::handleCheck()
{
    cout << "Got a call from the main window." << endl;
}

void SimThread::run()
{
    // Create the input visualization
    // int windowWidth  = 800;
    // int windowHeight = numGO;
    // ActTemporalView inputNetTView(numGO, 1, windowWidth, windowWidth, windowHeight, Qt::white, "inputNet");

    // inputNetTView.show();
    // inputNetTView.update();    

    for (int simStep=0; ; simStep++) {
        if (simStep % 10000 == 0)
            cout << endl;
        if (simStep % 1000 == 0)
            cout << "." << flush;

        // Calculate MF Activity
        for (int i=0; i<numMF; i++) {
            float contextFreqMin = 30;
            float contextFreqMax = 60;
            mfFreq[i] = randGen->Random()*(contextFreqMax-contextFreqMin)+contextFreqMin;    
        }
        const ct_uint8_t *apMF = mfs->calcActivity(&mfFreq[0]);
        
        // Calculate Sim Activity
        simCore->updateMFInput(apMF);
        simCore->calcActivity();

        // Display the activity
        // const ct_uint8_t* ap = simCore->getInputNet()->exportHistMF();
        // vector<ct_uint8_t> tmp;
        // for (int i=0; i<numGO; i++) {
        //     tmp.push_back(ap[i]);
        // }
        // inputNetTView.drawRaster(tmp, simStep);
        // inputNetTView.show();
        // inputNetTView.update();

        // if (simStep % windowWidth == 0)
        //     inputNetTView.drawBlank(Qt::black);
    }
}
