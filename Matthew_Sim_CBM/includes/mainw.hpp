#ifndef MAINW_H
#define MAINW_H

#include <iostream>
#include <vector>
#include <fstream>
#include <string>

#include <QtGui/QWidget>
#include <QtGui/QApplication>
#include <QtGui/QColor>
#include <QtCore/QString>
#include <QtCore/QStringList>

#include <QtGui/QPushButton>

#include <CBMVisualInclude/actspatialview.h>
#include <CBMVisualInclude/acttemporalview.h>

#include <CBMStateInclude/interfaces/iconnectivityparams.h>

#include "simthread.hpp"

class MainW : public QWidget
{
    Q_OBJECT

public:
    MainW(QWidget *parent, int numMZ, int randSeed, std::string conPF, std::string actPF,
          Environment *env);
    ~MainW();

protected:
    void keyPressEvent(QKeyEvent *);
    void keyReleaseEvent(QKeyEvent *);

    SimThread thread;

    ActTemporalView *inputNetTView;
    ActTemporalView *scTView;
    ActTemporalView *bcTView;
    ActTemporalView *pcTView;
    ActTemporalView *ncTView;
    ActTemporalView *ioTView;

public slots:
    void displayInputNetTView();
    void displayStellateTView();
    void displayBasketTView();
    void displayPurkinjeTView();
    void displayNucleusTView();
    void displayOliveTView();
};

#endif // MAINW_H
