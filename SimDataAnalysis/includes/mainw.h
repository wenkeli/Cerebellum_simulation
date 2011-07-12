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
#include "analysismodules/grpshpopanalysis.h"
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

    int calcTempMetricBinN;

    QString cellTypes[8];

    PSHData **pshs[8];
    PSHData **curPSH;

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

	void loadSimFile();

	void exportSim();
	void exportSinglePSH();
};

#endif // MAINW_H
