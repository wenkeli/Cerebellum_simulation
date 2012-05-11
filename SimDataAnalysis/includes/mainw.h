#ifndef MAINW_H
#define MAINW_H

#include <vector>
#include <QtGui/QMainWindow>
#include <QtGui/QApplication>
#include <QtGui/QFileDialog>
#include <QtCore/QString>

#include "common.h"
#include "globalvars.h"
#include "datamodules/psh.h"
#include "datamodules/pshgpu.h"
#include "datamodules/simerrorec.h"
#include "datamodules/simexternalec.h"
#include "datamodules/siminnet.h"
#include "datamodules/simmfinputec.h"
#include "datamodules/simmzone.h"
#include "datamodules/simoutputec.h"
#include "analysismodules/grpshpopanalysis.h"
#include "analysismodules/grconpshanalysis.h"
#include "analysismodules/spikerateanalysis.h"
#include "analysismodules/pshtravclusterbase.h"
#include "analysismodules/pshtravclusterpos2st.h"
#include "analysismodules/pshtravclustereucdist.h"
#include "analysismodules/innetspatialvis.h"
#include "analysismodules/pshmultibinact.h"

#include "pshdispw.h"

#include "ui_mainw.h"

class MainW : public QMainWindow
{
    Q_OBJECT

public:
    MainW(QWidget *parent, QApplication *application);
    ~MainW();

private:
    Ui::MainWClass ui;
    QApplication *app;
    PSHDispw *curSingleWindow;
    PSHDispw *curMultiWindow;

    PSHDispw *curClusterWindow;
    PSHDispw *curClusterCellWindow;

    PSHDispw *curSpatialWindow;

    int calcTempMetricBinN;

    QString cellTypes[8];

    PSHData **pshs[8];
    PSHData **curPSH;


    GRConPSHAnalysis *grConAnalysis;
    BasePSHTravCluster *pshTravCluster;

    SpikeRateAnalysis **srAnalysis[8];
    SpikeRateAnalysis **curSRAnalysis;

    InNetSpatialVis *spatialVis;

    bool pshLoaded;
    bool simLoaded;

//    void calcGRTotalSpikes();
//    void calcGRTotalSpikesPC();
//
//    void calcGRTempSpecific();
//    void calcGRPopTempMetric();
//
//    void calcGRPlastTempMetric(ofstream &);
//    void calcGRPlastTempMetricPC(ofstream &);
//    void initGRPlastTempVars();
//
//    void calcGRLTDSynWeight(int, float);
//    void calcGRLTDSynWeightPC(int, float, int);
//
//    void calcGRLTPSynWeight(int, double);
//    void calcGRLTPSynWeightPC(int, double, int);
//
//    void calcGRPlastPopAct(int);
//    void calcGRPlastPopActPC(int, int);
//
//    double calcGRPlastPopActDiff(int);
//    double calcGRPlastPopActDiffPC(int, int);
//
//    void calcGRPlastPopActDiffSum(int);
//
//
//    void calcGRLTDPopSpec(int);
//    void calcGRLTDPopAmp(int);

public slots:
	void dispSingleCellNP();
	void dispMultiCellNP();
	void updateSingleCellDisp(int);
	void updateMultiCellDisp(int);
	void updateMultiCellBound(int);
	void updateCellType(int);
	void loadPSHFile();

	void calcPFPCPlasticity();
	void exportPFPCPlastAct();

	void showGRInMFGOPSHs();
	void showGROutGOPSHs();

	void calcSpikeRates();
	void exportSpikeRates();

	void updateClusterCellType(int);
	void makeClusters();
	void updateClusterDisp(int);
	void updateClusterCellDisp(int);
	void dispClusterNP();
	void dispClusterCellNP();

	void dispInNetSpatialNP();
	void updateInNetSpatial(int);
	void exportInNetBinData();

	void generate3DClusterData();

	void loadSimFile();

	void exportSim();
	void exportSinglePSH();
};

#endif // MAINW_H
