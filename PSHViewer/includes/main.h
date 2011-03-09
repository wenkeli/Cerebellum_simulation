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
unsigned long grBinTotalSpikes[NUMBINS];
float grTempSpecificity[NUMGR][NUMBINS];
unsigned short grTempSpPeakBin[NUMGR];
float grTempSpPeakVal[NUMGR];

float specGRPopSpMean[NUMBINS];
float activeGRPopSpMean[NUMBINS];
float totalGRPopSpMean[NUMBINS];

int numGRSpecific[NUMBINS];
int numGRActive[NUMBINS];

float specGRPopActMean[NUMBINS];
float activeGRPopActMean[NUMBINS];
float totalGRPopActMean[NUMBINS];
float spTotGRPopActR[NUMBINS];
float spActGRPopActR[NUMBINS];
float actTotGRPopActR[NUMBINS];

float grWeightsLTD[NUMBINS][NUMGR];
double grPopActLTD[NUMBINS][NUMBINS];
double grPopActDiffLTD[NUMBINS][NUMBINS];
double grPopActDiffSumLTD[NUMBINS];
float grPopActSpecLTD[NUMBINS];
float grPopActAmpLTD[NUMBINS];


vector<vector<unsigned short> > pshActiveGR;

int main(int argc, char **argv);

#endif /* MAIN_H_ */
