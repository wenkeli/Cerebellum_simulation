/*
 * main.h
 *
 *  Created on: Jan 5, 2010
 *      Author: wen
 */

#ifndef MAIN_H_
#define MAIN_H_

#include <QtGui/QApplication>

#include "common.h"
#include "mainw.h"

unsigned int numTrials=1;

unsigned short pshMF[NUMBINS][NUMMF];
unsigned short pshMFMax=1;

unsigned short pshGO[NUMBINS][NUMGO];
unsigned short pshGOMax=1;

unsigned short pshGR[NUMBINS][NUMGR];
unsigned short pshGRMax=1;
unsigned short pshGRTrans[NUMGR][NUMBINS];

unsigned int grTotalSpikes[NUMGR];
float grTempSpecificity[NUMGR][NUMBINS];
unsigned short grTempSpPeakBin[NUMGR];
float grTempSpPeakVal[NUMGR];

float grPopSpecMean[NUMBINS];
float grPopSpecSR[NUMBINS];
vector<vector<unsigned short> > pshValGR;

int main(int argc, char **argv);

#endif /* MAIN_H_ */
