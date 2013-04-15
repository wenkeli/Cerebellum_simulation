#ifndef MAINW_H
#define MAINW_H

#include <iostream>
#include <vector>
#include <fstream>
#include <string>

#include <QtGui/QWidget>
#include <QtGui/QApplication>
#include <QtGui/QColor>
#include <QtGui/QVBoxLayout>
#include <QtGui/QPushButton>
#include <QtCore/QString>
#include <QtCore/QStringList>

#include <CBMVisualInclude/actspatialview.h>
#include <CBMVisualInclude/acttemporalview.h>

#include <CBMStateInclude/interfaces/iconnectivityparams.h>

#include "simthread.hpp"

class MainW : public QWidget
{
    Q_OBJECT

public:
    MainW(QWidget *parent, SimThread *thread, Environment *env);
    ~MainW();

protected:
    void keyPressEvent(QKeyEvent *);
    void keyReleaseEvent(QKeyEvent *);

    SimThread *thread;

    ActTemporalView inputNetTView;
    ActTemporalView scTView;
    std::vector<ActTemporalView*> bcTViews;
    std::vector<ActTemporalView*> pcTViews;
    std::vector<ActTemporalView*> ncTViews;
    std::vector<ActTemporalView*> ioTViews;

    QVBoxLayout vbox;
    QPushButton inputNetTButton, stellateTButton, basketTButton,
        purkinjeTButton, nucleusTButton, oliveTButton;

public slots:
    void drawBCRaster(std::vector<ct_uint8_t> aps, int t, int mz);
    void drawPCVmRaster(std::vector<ct_uint8_t> aps, std::vector<float> vm, int t, int mz);
    void drawNCVmRaster(std::vector<ct_uint8_t> aps, std::vector<float> vm, int t, int mz);
    void drawIOVmRaster(std::vector<ct_uint8_t> aps, std::vector<float> vm, int t, int mz);    
};

#endif // MAINW_H
