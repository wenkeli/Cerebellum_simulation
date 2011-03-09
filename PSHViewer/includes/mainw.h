#ifndef MAINW_H
#define MAINW_H

#include <vector>
#include <QtGui/QMainWindow>
#include <QtGui/QApplication>
#include <QtGui/QFileDialog>
#include <QtCore/QString>

#include "common.h"
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
    PSHDispw *curAllWindow;

    bool grTotalCalced;

    void calcGRTotalSpikes();
    void calcGRTempSpecific();
    void calcGRPopTempMetric();

    void calcGRLTDTempMetric();
    void calcGRLTDSynWeight(int);
    void calcGRLTDPopAct(int);
    void calcGRLTDPopActDiff(int);
    void calcGRLTDPopActDiffSum(int);
    void calcGRLTDPopSpec(int);
    void calcGRLTDPopAmp(int);

public slots:
	void dispSingleCell();
	void dispAllCells();
	void loadPSHFile();
	void calcTempMetrics();
};

#endif // MAINW_H
