/*
 * simthread.h
 *
 *  Created on: Mar 2, 2012
 *      Author: consciousness
 */

#ifndef SIMTHREAD_H_
#define SIMTHREAD_H_

#include <vector>

#include <QtCore/QThread>
#include <QtCore/QMutex>
#include <QtCore/QMutexLocker>
#include <QtCore/QRect>
#include <QtGui/QColor>

#include <QtCore/QWaitCondition>

#include <interface/cbmsimcore.h>
#include <tools/mfpoissonregen.h>
#include <actspatialview.h>

#include "../ecmanagement.h"

class SimThread : public QThread
{
	Q_OBJECT

public:
	explicit SimThread(QObject *parent, ECManagement *ecsim, ActSpatialView *sview);
	~SimThread();

	void run();

	void lockAccessData();
	void unlockAccessData();
signals:
	void updateSpatialW(std::vector<bool>, int cellT, bool refresh);

private:
	ECManagement *management;

	ActSpatialView *spatialView;

	void simLoop();

	SimThread();

	QMutex accessDataLock;
};

#endif /* SIMTHREAD_H_ */
