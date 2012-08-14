/*
 * simthread.h
 *
 *  Created on: Mar 2, 2012
 *      Author: consciousness
 */

#ifndef SIMTHREAD_H_
#define SIMTHREAD_H_

#include <vector>
#include <iostream>

#include <QtCore/QThread>
#include <QtCore/QMutex>
#include <QtCore/QMutexLocker>
#include <QtCore/QRect>
#include <QtGui/QColor>
#include <QtCore/QTime>

#include <QtCore/QWaitCondition>

#include <interface/cbmsimcore.h>
#include <tools/mfpoissonregen.h>
#include <actspatialview.h>
#include <acttemporalview.h>

#include "../ecmanagement.h"

class SimThread : public QThread
{
	Q_OBJECT

public:
	explicit SimThread(QObject *parent, ECManagement *ecsim,
			ActSpatialView *sview, ActTemporalView *pcTV);
	~SimThread();

	void run();

	void lockAccessData();
	void unlockAccessData();
signals:
	void updateSpatialW(std::vector<bool>, int cellT, bool refresh);
	void updatePCTW(std::vector<bool>, std::vector<float>, int t);

private:
	ECManagement *management;

	ActSpatialView *spatialView;
	ActTemporalView *pcTView;

	void simLoop();

	SimThread();

	QMutex accessDataLock;
	QTime timer;
};

#endif /* SIMTHREAD_H_ */
