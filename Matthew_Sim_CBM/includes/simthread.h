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

    // Methods to display different cell groups
    void displayInputNetTView();
    void displayStellateTView();
    void displayBasketTView();
    void displayPurkinjeTView();
    void displayNucleusTView();
    void displayOliveTView();

    bool alive, running;

public slots:
    void destroyInputNetTView() { inputNetTView = NULL; };
    void destroyStellateTView() { scTView = NULL; };
    void destroyBasketTView()   { bcTView = NULL; };
    void destroyPurkinjeTView() { pcTView = NULL; };
    void destroyNucleusTView()  { ncTView = NULL; };
    void destroyOliveTView()    { ioTView = NULL; };

protected:
    ActTemporalView* createTemporalView(int numCells, int windowWidth, int windowHeight,
                                         QColor col, std::string name, std::vector<ct_uint8_t> *visVec,
                                         std::vector<float> *vmVec = 0);
    void displayFirings(ActTemporalView *view, const ct_uint8_t* ap, std::vector<ct_uint8_t>& vis,
                        int simStep, int numCells, const float* vm = 0, std::vector<float> *vmVis = 0);

    CBMState *simState;
    CBMSimCore *simCore;
    PoissonRegenCells *mfs;
    InNetInterface *inNet;
    MZoneInterface *mZone;
    
    CRandomSFMT0 *randGen;

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

    std::vector<float> mfFreq;

    // Used for visualizing the different cell types
    std::vector<ct_uint8_t> apGRVis;
    std::vector<ct_uint8_t> apGOVis;
    std::vector<ct_uint8_t> apGLVis;
    std::vector<ct_uint8_t> apMFVis;
    std::vector<ct_uint8_t> apSCVis;
    std::vector<ct_uint8_t> apBCVis;
    std::vector<ct_uint8_t> apPCVis;
    std::vector<float>      vmPCVis;
    std::vector<ct_uint8_t> apNCVis;
    std::vector<float>      vmNCVis;
    std::vector<ct_uint8_t> apIOVis;
    std::vector<float>      vmIOVis;

    // Visualizations
    ActTemporalView *inputNetTView;
    ActTemporalView *scTView;
    ActTemporalView *bcTView;
    ActTemporalView *pcTView;
    ActTemporalView *ncTView;
    ActTemporalView *ioTView;

private:
    void run();
};

#endif /* SIMTHREAD_H_ */
