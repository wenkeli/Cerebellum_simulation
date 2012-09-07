#ifndef MAINW_H
#define MAINW_H

#include <iostream>
#include <vector>

#include <QtGui/QWidget>
#include <QtGui/QApplication>
#include <QtGui/QColor>

#include <CBMVisualInclude/actspatialview.h>
#include <CBMVisualInclude/acttemporalview.h>

#include "../ectrial/ecmanagementbase.h"
#include "../ectrial/ecmanagementdelay.h"
#include "simthread.h"

#include "uic/ui_mainw.h"

class MainW : public QWidget
{
    Q_OBJECT

public:
    MainW(QApplication *app, QWidget *parent = 0);
    ~MainW();

public slots:
	void run();

private:
    Ui::MainWClass ui;

    ECManagementBase *manager;
    SimThread *compThread;

    ActSpatialView *inputNetSpatialView;

    ActTemporalView *inputNetTView;
    ActTemporalView *scTView;
    ActTemporalView *bcTView;
    ActTemporalView *pcTView;
    ActTemporalView *ncTView;
    ActTemporalView *ioTView;

};

#endif // MAINW_H
