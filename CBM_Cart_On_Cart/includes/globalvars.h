/*
 * globalvars.h
 *
 *  Created on: Oct 27, 2009
 *      Author: wen
 */
#ifndef GLOBALVARS_H_
#define GLOBALVARS_H_

#include <QtCore/QMutex>
#include <vector>
#include <iostream>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>

#include "parameters.h"
#include "mfinputmodules/mfinputbase.h"
#include "errorinputmodules/errorinputbase.h"
#include "outputmodules/outputbase.h"
#include "externalmodules/externalbase.h"
#include "mzonemodules/mzone.h"
#include "innetmodules/innet.h"

#include "analysismodules/psh.h"

using namespace std;

// Global Random Number Generator
extern CRandomSFMT0 *randGen;

//control and thread variables
extern cudaStream_t streams[CUDANUMSTREAMS];

extern bool initialized;
extern QMutex bufLock;
extern QMutex simPauseLock;
extern QMutex simDispTypeLock;
extern int simDispType;
extern QMutex simStopLock;
extern bool simStop;

extern QMutex simPSHCheckLock;
extern bool simPSHCheck;

extern QMutex simCalcSpikeHistLock;
extern bool simCalcSpikeHist;

extern QMutex simDispActsLock;
extern bool simDispActs;

extern QMutex simDispRasterLock;
extern bool simDispRaster;

extern QMutex simMZDispNumLock;
extern int simMZDispNum;

extern QMutex accessPSHLock;

extern QMutex accessConnLock;

extern QMutex accessSpikeSumLock;

extern QMutex pfSynWeightPCLock;
//end control and thread variables

extern QMutex testOutputLock;
extern float testOutput;

extern BaseExternal* externalMod;

//simulation analysis variables - not cell type specific
extern unsigned int numTrials;
extern unsigned int msCount;
//end non cell type specific analysis variables

//simulation file outputs
extern ofstream simOut;
extern ofstream pshOut;
extern ifstream simIn;
//end file outputs

extern bool grInputSwitch;

//simulation setup
extern MZone *zones[NUMMZONES];

extern InNet *inputNetwork;

//end simulation setup

//analysis setup
//PSHs
extern PSHAnalysis *mfPSH;
extern PSHAnalysis *goPSH;
extern PSHAnalysis *grPSH;
extern PSHAnalysis *scPSH;

extern PSHAnalysis *bcPSH[NUMMZONES];
extern PSHAnalysis *pcPSH[NUMMZONES];
extern PSHAnalysis *ioPSH[NUMMZONES];
extern PSHAnalysis *ncPSH[NUMMZONES];
//end analysis setup

//---------mossy fiber variables-----------------

extern BaseMFInput *mfMod;

//---------end mossy fiber variables-------------

//---------- inferior olivary cell variables-----
extern BaseErrorInput *errMod[NUMMZONES];


//---------end IO variables---------------------

//---------- Nucleus cell variables-------------
extern BaseOutput *outputMod[NUMMZONES];
//-----------end NC variables------------------

//-------------simulation variables--------------

//CS onset times and duration
extern short const csOnset[4];
extern short const csDuration[4];
//-----------end simulation variables------------

//-----------start command line arguments------------
extern string cp_logfile;
extern int num_trials;
extern long max_trial_length;
extern float lower_cart_difficulty;
//-----------end command line arguments------------


#endif /* GLOBALVARS_H_ */
