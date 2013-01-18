/*
 * interthreadcomm.h
 *
 *  Created on: Jan 18, 2013
 *      Author: consciousness
 */

#ifndef INTERTHREADCOMM_H_
#define INTERTHREADCOMM_H_

#include <QtCore/QMutex>

class InterThreadComm
{
public:
	InterThreadComm();

	QMutex accessDispParamLock;
	int inNetDispCellT;
};


#endif /* INTERTHREADCOMM_H_ */
