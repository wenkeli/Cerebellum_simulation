#ifndef SIMTHREAD_H_
#define SIMTHREAD_H_

#include <string>
#include <vector>
#include <assert.h>
#include <fstream>
#include <queue>
#include <QtCore/QThread>

#include <CBMStateInclude/interfaces/cbmstate.h>
#include <CBMCoreInclude/interface/cbmsimcore.h>
#include <CBMToolsInclude/poissonregencells.h>
#include <CBMVisualInclude/acttemporalview.h>

#include "cbmViz.h"
#include "environments/environment.hpp"

// Saves the simulation to a file given an environment and the core
static void saveSim(std::string saveFile, Environment &env, CBMSimCore &simCore) {
    std::ofstream ofs(saveFile.c_str());
    {
        boost::archive::text_oarchive oa(ofs);
        oa << env;
    }
    
    std::fstream filestr(saveFile.c_str(), std::fstream::out | std::fstream::app);
    simCore.writeToState(filestr);
    filestr.close();
}

// Loads the simulation from a file and an already constructed environment
static CBMState* loadSim(std::string saveFile, Environment &env) {
    long pos; // Position that boost serialization ends
    std::ifstream ifs(saveFile.c_str());
    {
        boost::archive::text_iarchive ia(ifs);
        ia >> env;
        pos = ifs.tellg();
    }

    std::fstream stateStream(saveFile.c_str(), std::fstream::in);
    stateStream.seekp(pos); // Go to end of boost serialization
    CBMState *simState = new CBMState(stateStream);
    stateStream.close();
    return simState;
}

class SimThread : public QThread
{
    Q_OBJECT

public:
    SimThread(QObject *parent, int numMZ, int randSeed, std::string conPF, std::string actPF, Environment *env);
    SimThread(QObject *parent, int numMZ, int randSeed, CBMState *simState, Environment *env);
    ~SimThread();

    bool alive, paused;

    int trialLength; // Typically 5000
    int numMZ; // Microzones
    int numGR; // Granule Cells
    int numGO; // Golgi Cells
    int numGL; // Glomeruli?
    int numMF; // Mossy Fibers
    int numSC; // Stellate Cells
    int numBC; // Basket Cells
    int numPC; // Purkinje Cells
    int numNC; // Nucleus Cells
    int numIO; // Inferior Olive Cells

    void setupMFs(int randSeed);
    void disablePlasticity();
    void activateCF(int zoneN=0) { assert(zoneN < numMZ); simCore->updateErrDrive(zoneN, 1.0); };
    void activateMF(int mfNum) { assert(mfNum < numMF); env->mfExcited[mfNum] = true; };
    void deactivateMF(int mfNum) { assert(mfNum < numMF); env->mfExcited[mfNum] = false; };
    void save(std::string saveFile="cbm.st") { saveSim(saveFile, *env, *simCore); }
    void load(std::string saveFile="cbm.st") {
        env = new Environment(randGen);
        loadSim(saveFile, *env);
    }

public slots:
    void createGLVisualization();

signals:
    void updateINTW(std::vector<ct_uint8_t>, int t);
    void updateSCTW(std::vector<ct_uint8_t>, int t);
    void updateBCTW(std::vector<ct_uint8_t>, int t, int mz);
    void updatePCTW(std::vector<ct_uint8_t>, std::vector<float>, int t, int mz);
    void updateNCTW(std::vector<ct_uint8_t>, std::vector<float>, int t, int mz, float window);
    void updateIOTW(std::vector<ct_uint8_t>, std::vector<float>, int t, int mz);
    void blankTW(QColor bc);

protected:
    Environment *env;
    CBMState *simState;
    CBMSimCore *simCore;
    PoissonRegenCells *mfs;
    InNetInterface *inNet;
    std::vector<MZoneInterface*> mZones;

    CerebellumViz *cbmViz;
    
    CRandomSFMT0 *randGen;

    void setupMossyFibers(int randSeed);

private:
    void run();

    // Variables used to keep a running avg of NC firings
    static const int ncWindowLength = 100;
    std::queue<float> ncActQueue;
    float sumNCAct;
};

#endif /* SIMTHREAD_H_ */
