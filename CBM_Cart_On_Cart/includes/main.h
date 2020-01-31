/*
 * main.h
 *
 *  Created on: Oct 27, 2009
 *      Author: wen
 */

#ifndef MAIN_H_
#define MAIN_H_

#include <QtCore/QMutex>
#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "initsim.h"
#include "calcactivities.h"
#include "mfinputmodules/mfinputbase.h"
#include "errorinputmodules/errorinputbase.h"
#include "outputmodules/outputbase.h"
#include "externalmodules/externalbase.h"
#include "mzonemodules/mzone.h"
#include "innetmodules/innet.h"

#include "analysismodules/psh.h"

//============variables used globally===================
//see globalvars.h for explanation of the variables
//control and thread variables
CRandomSFMT0 *randGen=NULL;

cudaStream_t streams[CUDANUMSTREAMS];

bool initialized=false;
QMutex bufLock;
QMutex simPauseLock;
QMutex simDispTypeLock;
int simDispType;
QMutex simStopLock;
bool simStop=false;

QMutex simPSHCheckLock;
bool simPSHCheck=false;

QMutex simCalcSpikeHistLock;
bool simCalcSpikeHist=false;

QMutex simDispActsLock;
bool simDispActs=false;

QMutex simDispRasterLock;
bool simDispRaster=false;

QMutex simMZDispNumLock;
int simMZDispNum;

QMutex accessPSHLock; //PSH variables

QMutex accessConnLock; //connectivity variables

QMutex accessSpikeSumLock;

QMutex pfSynWeightPCLock;
//end control and thread variables

QMutex testOutputLock;
float testOutput;

BaseExternal* externalMod;

//simulation analysis variables - not cell type specific
unsigned int numTrials;
unsigned int msCount;
//end non cell type specific analysis variables

//simulation file outputs
ofstream simOut;
ofstream pshOut;
ifstream simIn;
//end file outputs

bool grInputSwitch=false;

//simulation setup
MZone *zones[NUMMZONES];

InNet *inputNetwork;

//end simulation setup

//analysis setup
//PSHs
PSHAnalysis *mfPSH;
PSHAnalysis *goPSH;
PSHAnalysis *grPSH;
PSHAnalysis *scPSH;

PSHAnalysis *bcPSH[NUMMZONES];
PSHAnalysis *pcPSH[NUMMZONES];
PSHAnalysis *ioPSH[NUMMZONES];
PSHAnalysis *ncPSH[NUMMZONES];
//end analysis setup


//---------mossy fiber variables-----------------
BaseMFInput *mfMod;
//---------end mossy fiber variables-------------


//---------- inferior olivary cell variables-----
BaseErrorInput *errMod[NUMMZONES];
//---------end IO variables---------------------

//---------- Nucleus cell variables-------------
BaseOutput *outputMod[NUMMZONES];
//-----------end NC variables------------------

//-------------simulation variables--------------
short const csOnset[4]={1500, 1500, 1500, 1500};
short const csDuration[4]={1000, 1000, 1000, 1000};
//-----------end simulation variables------------

//=============end globally used variables===============

string cp_logfile("log.txt");
int num_trials = 10;
long max_trial_length = 1000000;
float lower_cart_difficulty = 50;

int main(int, char **);

#endif /* MAIN_H_ */
