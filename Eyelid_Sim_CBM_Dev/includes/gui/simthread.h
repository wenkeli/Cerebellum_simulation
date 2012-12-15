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

#include <CXXToolsInclude/stdDefinitions/pstdint.h>

#include <CBMVisualInclude/actspatialview.h>
#include <CBMVisualInclude/acttemporalview.h>

#include "../ectrial/ecmanagementbase.h"

class SimThread : public QThread
{
	Q_OBJECT

public:
	explicit SimThread(QObject *parent, ECManagementBase *ecsim,
			ActSpatialView *inputNetSV,
			ActTemporalView *inputNetTV,
			ActTemporalView *scTV,
			ActTemporalView *bcTV,
			ActTemporalView *pcTV,
			ActTemporalView *ncTV,
			ActTemporalView *ioTV);
	~SimThread();

	void run();

	void lockAccessData();
	void unlockAccessData();
signals:
	void updateSpatialW(std::vector<ct_uint8_t>, int cellT, bool refresh);
	void spatialFrameDump();

	void updateINTW(std::vector<ct_uint8_t>, int t);
	void updateSCTW(std::vector<ct_uint8_t>, int t);
	void updateBCTW(std::vector<ct_uint8_t>, int t);
	void updatePCTW(std::vector<ct_uint8_t>, std::vector<float>, int t);
	void updateNCTW(std::vector<ct_uint8_t>, std::vector<float>, int t);
	void updateIOTW(std::vector<ct_uint8_t>, std::vector<float>, int t);
	void blankTW(QColor bc);

private:
	ECManagementBase *management;

	ActSpatialView *inputNetSView;

	ActTemporalView *inputNetTView;
	ActTemporalView *scTView;
	ActTemporalView *bcTView;
	ActTemporalView *pcTView;
	ActTemporalView *ncTView;
	ActTemporalView *ioTView;

	void simLoop();

	SimThread();

	QMutex accessDataLock;
	QTime timer;
};

#endif /* SIMTHREAD_H_ */
