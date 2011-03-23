/*
 * common.h
 *
 *  Created on: May 4, 2009
 *      Author: wen
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>
#include <ctime>

#ifndef NULL
#define NULL 0
#endif

#define NUMBINS 240//200

#define CSSTARTBIN 20
#define CSSTARTT 100

#define CSENDBIN 220
#define CSENDT 1100

#define CSDURATIONBIN 200
#define CSDURATIONT 1000

#define NUMMF 1024
#define NUMGO 1024
#define NUMGR 1048576

#define BINWIDTH 5

#define ALLVIEWPH 1024
#define ALLVIEWPW 1200 //1000

#define SINGLEVIEWPH 500
#define SINGLEVIEWPW 1200 //1000

#define TEMPMETSLIDINGW 40 //200 ms

#define GRSYNWEIGHTINI 1.0f
#define GRSYNWEIGHTMAX 2.0f
#define LTDSTEP 0.05f//0.2f
#define LTPSTEP 0.05f//0.5f


using namespace std;

extern unsigned int numTrials;

extern unsigned short pshMF[NUMBINS][NUMMF];
extern unsigned short pshMFMax;

extern unsigned short pshGO[NUMBINS][NUMGO];
extern unsigned short pshGOMax;

extern unsigned short pshGR[NUMBINS][NUMGR];
extern unsigned short pshGRMax;
extern unsigned short pshGRTrans[NUMGR][NUMBINS];

extern unsigned int grTotalSpikes[NUMGR];
extern unsigned long grBinTotalSpikes[NUMBINS];
extern float grTempSpecificity[NUMGR][NUMBINS];
extern unsigned short grTempSpPeakBin[NUMGR];
extern float grTempSpPeakVal[NUMGR];

extern int numGRSpecific[NUMBINS];
extern int numGRActive[NUMBINS];

extern float specGRPopSpMean[NUMBINS];
extern float activeGRPopSpMean[NUMBINS];
extern float totalGRPopSpMean[NUMBINS];

extern float specGRPopActMean[NUMBINS];
extern float activeGRPopActMean[NUMBINS];
extern float totalGRPopActMean[NUMBINS];
extern float spTotGRPopActR[NUMBINS];
extern float spActGRPopActR[NUMBINS];
extern float actTotGRPopActR[NUMBINS];

extern float grWeightsPlast[NUMBINS][NUMGR];
extern double grPopActPlast[NUMBINS][NUMBINS];
extern double grPopActDiffPlast[NUMBINS][NUMBINS];
extern double grPopActDiffSumPlast[NUMBINS];
extern float grPopActSpecPlast[NUMBINS];
extern float grPopActAmpPlast[NUMBINS];

extern vector<vector<unsigned short> > pshActiveGR;

#endif /* COMMON_H_ */
