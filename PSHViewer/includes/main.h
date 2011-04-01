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
float ratesGRTrans[NUMGR][NUMBINS];

unsigned int pshPC[NUMBINS][NUMPC];
unsigned int pshPCMax=1;

unsigned int pshBC[NUMBINS][NUMBC];
unsigned int pshBCMax=1;

unsigned int pshSC[NUMBINS][NUMSC];
unsigned int pshSCMax=1;

double grTotalSpikes[NUMGR];
double grBinTotalSpikes[NUMBINS];
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

float grWeightsPlast[NUMBINS][NUMGR];
double grPopActPlast[NUMBINS][NUMBINS];
double grPopActDiffPlast[NUMBINS][NUMBINS];
double grPopActDiffSumPlast[NUMBINS];
float grPopActSpecPlast[NUMBINS];
float grPopActAmpPlast[NUMBINS];


//simulation state variables
short conNumMFtoGR[NUMMF+1];
char conNumMFtoGO[NUMMF+1];
short conNumGOtoGR[NUMGO+1];
char conNumGRtoGO[NUMGR+1];

int conMFtoGR[NUMMF+1][NUMGRPERMF];
short conMFtoGO[NUMMF+1][MFGOSYNPERMF];
int conGOtoGR[NUMGO+1][NUMGROUTPERGO];
short conGRtoGO[NUMGR+1][GRGOSYNPERGR];
char conBCtoPC[NUMBC][BCPCSYNPERBC];
char conIOCouple[NUMIO][IOCOUPSYNPERIO];
char conPCtoNC[NUMPC][PCNCSYNPERPC];

char typeMFs[NUMMF+1];
float bgFreqContsMF[NUMCONTEXTS][NUMMF+1];
float incFreqMF[NUMMF+1];
short csStartMF[NUMMF+1];
short csEndMF[NUMMF+1];

float pfSynWeightPC[NUMGR];


vector<vector<unsigned short> > pshActiveGR;

int main(int argc, char **argv);

#endif /* MAIN_H_ */
