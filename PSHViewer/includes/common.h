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

//simulation state parameters
//connectivity parameters
#define NUMGRPERMF 5120
#define MFGOSYNPERMF 64
#define NUMGROUTPERGO 3840
#define GRGOSYNPERGR 2
#define BCPCSYNPERBC 4
#define IOCOUPSYNPERIO 1
#define PCNCSYNPERPC 3

#define NUMMF 1024
#define NUMGO 1024
#define NUMGR 1048576
#define NUMBC 128
#define NUMIO 4
#define NUMPC 32
#define NUMSC 512
//end connectivity parameters
//mossy fiber parameters
#define NUMCONTEXTS 2
//end mossy fiber parameters
//end simulation state parameters



using namespace std;

extern unsigned int numTrials;

extern unsigned short pshMF[NUMBINS][NUMMF];
extern unsigned short pshMFMax;

extern unsigned short pshGO[NUMBINS][NUMGO];
extern unsigned short pshGOMax;

extern unsigned short pshGR[NUMBINS][NUMGR];
extern unsigned short pshGRMax;
extern unsigned short pshGRTrans[NUMGR][NUMBINS];
extern float ratesGRTrans[NUMGR][NUMBINS];

extern unsigned int pshPC[NUMBINS][NUMPC];
extern unsigned int pshPCMax;

extern unsigned int pshBC[NUMBINS][NUMBC];
extern unsigned int pshBCMax;

extern unsigned int pshSC[NUMBINS][NUMSC];
extern unsigned int pshSCMax;

//temporal metric variables
extern double grTotalSpikes[NUMGR];
extern double grBinTotalSpikes[NUMBINS];
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

//simulation state variables
extern short conNumMFtoGR[NUMMF+1];
extern char conNumMFtoGO[NUMMF+1];
extern short conNumGOtoGR[NUMGO+1];
extern char conNumGRtoGO[NUMGR+1];

extern int conMFtoGR[NUMMF+1][NUMGRPERMF];
extern short conMFtoGO[NUMMF+1][MFGOSYNPERMF];
extern int conGOtoGR[NUMGO+1][NUMGROUTPERGO];
extern short conGRtoGO[NUMGR+1][GRGOSYNPERGR];
extern char conBCtoPC[NUMBC][BCPCSYNPERBC];
extern char conIOCouple[NUMIO][IOCOUPSYNPERIO];
extern char conPCtoNC[NUMPC][PCNCSYNPERPC];

extern char typeMFs[NUMMF+1];
extern float bgFreqContsMF[NUMCONTEXTS][NUMMF+1];
extern float incFreqMF[NUMMF+1];
extern short csStartMF[NUMMF+1];
extern short csEndMF[NUMMF+1];

extern float pfSynWeightPC[NUMGR];

extern vector<vector<unsigned short> > pshActiveGR;

#endif /* COMMON_H_ */
