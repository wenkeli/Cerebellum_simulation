#ifndef GENESISMW_H
#define GENESISMW_H

#include <QtGui/QMainWindow>
#include <QtGui/QApplication>
#include <QtCore/QMutex>
#include <QtGui/QFileDialog>
#include <QtCore/QString>
#include "common.h"

#include "ui_mainw.h"
#include "synapsegenesis.h"
#include "initsim.h"
#include "globalvars.h"
#include "simthread.h"
#include "dispdialoguep.h"
#include "simdispw.h"
#include "simctrlp.h"
#include "readin.h"

//main window class that is responsible for initial display, and handles spawning
//DispDialogueP to display connections. Also handles initializing connections and
//spawning simulation thread and displaying realtime simulation data.
//see mainw.ui for graphical information on the widget layouts and signal slot setup.
class MainW : public QMainWindow
{
    Q_OBJECT //necessary for qt signal slot mechanism, ignore.

public:
	//QApplication *a: set the application object so the quit button works properly
    MainW(QWidget *parent, QApplication *a);
    ~MainW();
    //returns the main text status display for other classes to write status
    //messages to
    QTextBrowser *getStatusBox();
    //set the application object so the quit button works properly to quit the application
    //void setApp(QApplication *);

private:
    Ui::MainWClass ui; //auto generated code from ui_mainw.h, ignore
    QApplication *app; //main application this window is attached to

public slots:
	//custom qt slots to handle various events, such as display various types of connections
	void makeConns(); //makes the connections
	void loadSim(); //load connections and state data from a file
	void showMFGRMainP(); //show MFGR connection
	void showMFGOMainP(); //show MGRO connection
	void showGRGOMainP(); //show GRGO connection
	void showGOGRMainP(); //show GOGR connection
	void runSimulation(); //run the simulation, and display realtime data
};

#endif // GENESISMW_H
