/*
 * interthreadcomm.cpp
 *
 *  Created on: Jan 18, 2013
 *      Author: consciousness
 */


#include "../includes/interthreadcomm.h"

InterThreadComm::InterThreadComm()
{
	accessDispParamLock.unlock();
	inNetDispCellT=0;
}

