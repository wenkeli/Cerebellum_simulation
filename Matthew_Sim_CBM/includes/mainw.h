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

#include <QPushButton>

#include <CBMVisualInclude/actspatialview.h>
#include <CBMVisualInclude/acttemporalview.h>

#include <CBMStateInclude/interfaces/iconnectivityparams.h>

#include "simthread.h"

class MainW : public QWidget
{
    Q_OBJECT

public:
    MainW(QApplication *app, QWidget *parent, int numMZ, int randSeed, std::string conPF, std::string actPF);
    ~MainW();

protected:
    void keyPressEvent(QKeyEvent *);
    void keyReleaseEvent(QKeyEvent *);

    SimThread thread;

public slots:
    void displayInputNetTView() { thread.displayInputNetTView(); };
    void displayStellateTView() { thread.displayStellateTView(); };
    void displayBasketTView()   { thread.displayBasketTView(); };
    void displayPurkinjeTView() { thread.displayPurkinjeTView(); };
    void displayNucleusTView()  { thread.displayNucleusTView(); };
    void displayOliveTView()    { thread.displayOliveTView(); };
};

#endif // MAINW_H
