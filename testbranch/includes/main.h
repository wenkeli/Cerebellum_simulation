/*
 * main.h
 *
 *  Created on: Jan 9, 2013
 *      Author: consciousness
 */

#ifndef MAIN_H_
#define MAIN_H_

#include <fstream>
#include <iostream>
#include <time.h>

#include <QtGui/QApplication>
#include "gui/mainw.h"

#include <CBMStateInclude/params/activityparams.h>
#include <CBMStateInclude/params/connectivityparams.h>

#include <CBMStateInclude/state/innetactivitystate.h>
#include <CBMStateInclude/state/innetconnectivitystate.h>

#include <CBMStateInclude/state/mzoneactivitystate.h>
#include <CBMStateInclude/state/mzoneconnectivitystate.h>

#include <CBMStateInclude/interfaces/cbmstate.h>

int main(int argc, char *argv[]);


#endif /* MAIN_H_ */
