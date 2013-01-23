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

    void handleCheck();

    bool running;

protected:
    CBMState *simState;
    CBMSimCore *simCore;
    PoissonRegenCells *mfs;
    
    CRandomSFMT0 *randGen;

    int numMZ, numMF, numGO;
    std::vector<float> mfFreq;

private:
    void run();
};

#endif /* SIMTHREAD_H_ */
