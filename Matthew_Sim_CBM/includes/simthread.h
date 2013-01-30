#ifndef SIMTHREAD_H_
#define SIMTHREAD_H_

#include <string>
#include <vector>

#include <QtCore/QThread>

#include <CBMStateInclude/interfaces/cbmstate.h>
#include <CBMCoreInclude/interface/cbmsimcore.h>
#include <CBMToolsInclude/poissonregencells.h>
#include <CBMVisualInclude/acttemporalview.h>

class SimThread : public QThread
{
    Q_OBJECT

public:
    SimThread(QObject *parent, int numMZ, int randSeed, std::string conPF, std::string actPF);
    ~SimThread();

    bool alive, running;

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

signals:
    void updateINTW(std::vector<ct_uint8_t>, int t);
    void updateSCTW(std::vector<ct_uint8_t>, int t);
    void updateBCTW(std::vector<ct_uint8_t>, int t);
    void updatePCTW(std::vector<ct_uint8_t>, std::vector<float>, int t);
    void updateNCTW(std::vector<ct_uint8_t>, std::vector<float>, int t);
    void updateIOTW(std::vector<ct_uint8_t>, std::vector<float>, int t);
    void blankTW(QColor bc);

protected:
    CBMState *simState;
    CBMSimCore *simCore;
    PoissonRegenCells *mfs;
    InNetInterface *inNet;
    MZoneInterface *mZone;
    
    CRandomSFMT0 *randGen;

    std::vector<float> mfFreq;

private:
    void run();
};

#endif /* SIMTHREAD_H_ */
