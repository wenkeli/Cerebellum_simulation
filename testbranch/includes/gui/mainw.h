#ifndef MAINW_H
#define MAINW_H

#include <fstream>
#include <vector>
#include <iostream>
#include <sstream>
#include <string>

#include <QtGui/QMainWindow>
#include <QtGui/QApplication>
#include <QtCore/QStringList>
#include <QtCore/QString>
#include <QtGui/QColor>

#include <CXXToolsInclude/stdDefinitions/pstdint.h>

#include <CBMStateInclude/interfaces/cbmstate.h>
#include <CBMStateInclude/interfaces/iactivityparams.h>
#include <CBMStateInclude/interfaces/iconnectivityparams.h>
#include <CBMStateInclude/interfaces/iinnetconstate.h>

#include <CBMDataInclude/outdata/eyelidout.h>
#include <CBMDataInclude/spikeraster/spikerasterbitarray.h>
#include <CBMDataInclude/interfaces/ectrialsdata.h>
#include <CBMDataInclude/interfaces/ispikeraster.h>

#include <CBMVisualInclude/connectivityview.h>

#include "uic/ui_mainw.h"

class MainW : public QMainWindow
{
    Q_OBJECT

public:
    MainW(QApplication *application, QWidget *parent = 0);
    ~MainW();

public slots:
	void showConnection();
	void updateConCellN(int cellNum);
	void updateConCellT(int cellType);

private:
    Ui::MainWClass ui;

    MainW();

    QApplication *app;

    CBMState *simState;
    IInNetConState *innetCS;
    IConnectivityParams *conP;

    ConnectivityView *conView;

    int conDispCellN;
    int conDispCellT;
    std::vector<QString> conDispCTNames;
    std::vector<int> conDispCTMaxNs;
    std::vector<std::vector<unsigned int> > conDispCTs;

    ECTrialsData *data;
};

#endif // MAINW_H
