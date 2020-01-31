#ifndef SIMCTRLP_H
#define SIMCTRLP_H

#include <QtGui/QWidget>
#include "ui_simctrlp.h"
#include "simdispw.h"
#include "actdiagw.h"
#include "spikeratesdispw.h"
#include "simthread.h"
#include "writeout.h"

class SimCtrlP : public QWidget
{
    Q_OBJECT

public:
    SimCtrlP(QWidget *parent = 0);
    ~SimCtrlP();

private:
    Ui::SimCtrlPClass ui;
    bool paused;
    SimDispW *panel;
    ActDiagW *activityW;
    SimThread *simThread;

public slots:
	void startSim();
	void pauseSim();
	void stopSim();
	void dispSpikeRates();
	void exportPSH();
	void exportSim();

	void changeDispMode(int dispMode);
	void changeActMode(int actMode);
	void changePSHMode(int pshMode);
	void changeRasterMode(int rasterMode);
	void changeSRHistMode(int srhMode);
};

#endif // SIMCTRLP_H
