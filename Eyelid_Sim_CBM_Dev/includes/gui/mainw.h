#ifndef MAINW_H
#define MAINW_H

#include <iostream>
#include <vector>

#include <QtGui/QWidget>
#include <QtGui/QApplication>
#include <QtGui/QColor>
#include "uic/ui_mainw.h"

#include <interface/cbmsimcore.h>
#include <tools/mfpoissonregen.h>
#include <actspatialview.h>

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

    ActSpatialView *spatialView;
};

#endif // MAINW_H
