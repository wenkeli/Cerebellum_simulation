#include <iostream>
#include <fstream>

#include "../includes/simthread.h"

using namespace std;

SimThread::SimThread(QObject *parent, int numMZ, int randSeed, string conPF, string actPF)
    : QThread(parent), alive(true), running(true), trialLength(5000), numMZ(numMZ),
      inputNetTView(NULL), scTView(NULL), bcTView(NULL), pcTView(NULL), ncTView(NULL), ioTView(NULL)
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

    // Setup the Mossy Fibers
    float threshDecayTau = 4.0f; // Rate of decay = 1-exp(-msPerTS/threshDecayTau)
    float msPerTimeStep = 1.0f;
    mfs = new PoissonRegenCells(numMF, randSeed, threshDecayTau, msPerTimeStep);
    mfFreq.resize(numMF);

    inNet = simCore->getInputNet();
    mZone = simCore->getMZoneList()[0];
}

SimThread::~SimThread()
{
    delete simState;
    delete simCore;
    delete mfs;
    if (inputNetTView) delete inputNetTView;
    if (scTView)       delete scTView;
}

ActTemporalView* SimThread::createTemporalView(int numCells, int windowWidth, int windowHeight,
                                                QColor col, string name, vector<ct_uint8_t>* visVec,
                                                vector<float> *vmVec)
{
    int pixelsPerCell = windowHeight / numCells;
    ActTemporalView *view = new ActTemporalView(numCells, pixelsPerCell, trialLength,
                                                windowWidth, windowHeight,
                                                col, name.c_str()); 
    view->setAttribute(Qt::WA_DeleteOnClose);
    view->show();
    view->update();
    visVec->resize(numCells);
    if (vmVec) vmVec->resize(numCells);
    return view;
}

/* --------------- Methods to view cell groups --------------------- */
void SimThread::displayInputNetTView() {
    if (inputNetTView) return;
    inputNetTView = createTemporalView(1024, trialLength/4, 1024, Qt::white, "inputNet", &apMFVis);
    connect(inputNetTView, SIGNAL(destroyed(QObject*)), this, SLOT(destroyInputNetTView()));
}
void SimThread::displayStellateTView() {
    if (scTView) return;
    scTView = createTemporalView(numSC, trialLength/4, numSC, Qt::white, "stellate", &apSCVis);
    connect(scTView, SIGNAL(destroyed(QObject*)), this, SLOT(destroyStellateTView()));
}
void SimThread::displayBasketTView() {
    if (bcTView) return;
    bcTView = createTemporalView(numBC, trialLength/4, numBC, Qt::green, "basket", &apBCVis);
    connect(bcTView, SIGNAL(destroyed(QObject*)), this, SLOT(destroyBasketTView()));
}
void SimThread::displayPurkinjeTView() {
    if (pcTView) return;
    pcTView = createTemporalView(numPC, trialLength/4, numPC*8, Qt::red, "purkinje", &apPCVis, &vmPCVis);
    connect(pcTView, SIGNAL(destroyed(QObject*)), this, SLOT(destroyPurkinjeTView()));
}
void SimThread::displayNucleusTView() {
    if (ncTView) return;
    ncTView = createTemporalView(numNC, trialLength/2, numNC*16, Qt::green, "nucleus", &apNCVis, &vmNCVis);
    connect(ncTView, SIGNAL(destroyed(QObject*)), this, SLOT(destroyNucleusTView()));
}
void SimThread::displayOliveTView() {
    if (ioTView) return;
    ioTView = createTemporalView(numIO, trialLength/4, numIO*32, Qt::white, "inferior olive", &apIOVis, &vmIOVis);
    connect(ioTView, SIGNAL(destroyed(QObject*)), this, SLOT(destroyOliveTView()));
}

void SimThread::displayFirings(ActTemporalView *view, const ct_uint8_t* ap, vector<ct_uint8_t>& vis, int simStep,
                               int numCells, const float *vm, vector<float> *vmVis) {
    if (!view) return;
    for (int i=0; i<numCells; i++) {
        vis[i] = ap[i];
        if (vmVis)
            (*vmVis)[i] = vm[i];
    }
    if (vmVis)
        view->drawVmRaster(vis, *vmVis, simStep);
    else
        view->drawRaster(vis, simStep);
    view->show();
    view->update();
    if (simStep % trialLength == 0)
        view->drawBlank(Qt::black);
}

void SimThread::run()
{
    for (int simStep=0; alive; simStep++) {
        if (simStep % 10000 == 0) cout << endl;
        if (simStep % 1000 == 0) cout << "." << flush;

        // Calculate MF Activity
        for (int i=0; i<numMF; i++) {
            float contextFreqMin = 30;
            float contextFreqMax = 60;
            mfFreq[i] = randGen->Random()*(contextFreqMax-contextFreqMin)+contextFreqMin;    
        }
        const ct_uint8_t *apMF = mfs->calcActivity(&mfFreq[0]);
        simCore->updateMFInput(apMF);
        simCore->calcActivity();

        // Display activity of the different cellular groups
        displayFirings(inputNetTView, inNet->exportHistMF(), apMFVis, simStep, numGO);
        displayFirings(scTView, inNet->exportAPSC(), apSCVis, simStep, numSC);
        displayFirings(bcTView, mZone->exportAPBC(), apBCVis, simStep, numBC);
        displayFirings(pcTView, mZone->exportAPPC(), apPCVis, simStep, numPC);
        displayFirings(ncTView, mZone->exportAPNC(), apNCVis, simStep, numNC);
        displayFirings(ioTView, mZone->exportAPIO(), apIOVis, simStep, numIO);
    }
}
