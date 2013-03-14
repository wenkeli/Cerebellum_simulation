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

	for(int i=0; i<7; i++)
	{
		showActPanels[i]=false;
	}
}

