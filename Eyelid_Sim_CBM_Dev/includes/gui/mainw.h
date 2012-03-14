#ifndef MAINW_H
#define MAINW_H

#include <iostream>
#include <vector>

#include <QtGui/QWidget>
#include <QtGui/QApplication>
#include <QtGui/QColor>

#include <actspatialview.h>

#include "../ecmanagement.h"
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

    ECManagement *manager;
    SimThread *compThread;

    ActSpatialView *spatialView;
};

#endif // MAINW_H
