/*
 * simthread.h
 *
 *  Created on: Feb 16, 2009
 *      Author: wen
 */

#ifndef SIMTHREAD_H_
#define SIMTHREAD_H_

#include <QtCore/QThread>
#include <QtCore/QMutex>
#include <QtCore/QMutexLocker>
#include <QtCore/QRect>
#include <QtGui/QTransform>
#include <QtCore/QWaitCondition>

#include "initsim.h"
#include "../includes/calcactivities.h"

#include "common.h"
#include "globalvars.h"
#include "simdispw.h"
#include "actdiagw.h"

//#ifdef ACTDEBUG
//#include "actdiagw.h"
//#endif


//thread class that allows spawning a separate thread for running the simulation
class SimThread : public QThread
{

	Q_OBJECT

public:
	//QObject *parent: pass the caller of this thread as parent, can't be NULL
	//SimDispW *panel: sets the associated real time display for this simulation, can't be NULL
	explicit SimThread(QObject *parent, SimDispW *panel, ActDiagW *actW); //pass the caller of this thread as parent
	//void setStates(SimDispW *);//sets the associated real time display for this simulation
	void run(); //implemented virtual function that gets called when starting the thread

signals:
	void updateRaster(vector<bool>, int);
	void updatePSH(vector<unsigned short>, int, bool);
	void updateSCBCPCActs(SCBCPCActs, int);
	void updateIONCPCActs(IONCPCActs, int);
	void updateTotalAct(int, int);
	void updateCSBackground(int);
	void updateBlankDisp();
	void updateActW(vector<unsigned char>, vector<bool>);
//	void updateSCBCPCActs(vector<bool>, vector<bool>, vector<bool>, vector<float>);
//	void updateIONCPCActs(vector<bool>, vector<bool>, vector<bool>, vector<float>, vector<float>, vector<float>);
private:
	//various display functions
//	void dispActivities();

	SimDispW *dispW; //the associated realtime display for this simulation
	ActDiagW *activityW;

	void simLoop(); //the main simulation function
	void simLoopNew();
};

#endif /* SIMTHREAD_H_ */
