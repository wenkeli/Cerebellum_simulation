/*
 * main.h
 *
 *  Created on: Oct 27, 2009
 *      Author: wen
 */

#ifndef MAIN_H_
#define MAIN_H_

#include <QtCore/QMutex>

#include "common.h"

#include "synapsegenesis.h"
#include "initsim.h"
#include "calcactivities.h"
#include "actdiagw.h"
#include "writeout.h"

//============variables used globally===================
//see globalvars.h for explanation of the variables
//control and thread variables
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

QMutex accessPSHLock; //PSH variables

QMutex accessConnLock; //connectivity variables

QMutex accessSpikeSumLock;
//end control and thread variables

//simulation analysis variables - not cell type specific
unsigned int numTrials;
unsigned int msCount;
//end non cell type specific analysis variables

//simulation file outputs
ofstream simOut;
ofstream pshOut;
ifstream simIn;
//end file outputs

//debug variables
#ifdef DEBUG
std::vector<int> incompGRsGOGR;
std::vector<int> incompGOsGOGR;
#endif
//end debug variables

//TODO: memory alignment for all data variables

//----------connectivity data-------------------
//keep track of how many connections were made
short numSynMFtoGR[NUMMF+1];
//new
short numMFtoGRN[NUMMF+1];

char numSynMFtoGO[NUMMF+1];
//new
char numMFtoGON[NUMMF+1];

short numSynGOtoGR[NUMGO+1];
//new
short numGOtoGRN[NUMGO+1];

char numSynGRtoGO[NUMGR+1];

//Connectivity matrix of MF to GR connections
//encode GR # and dendrite # in one number
int conMFtoGR[NUMMF+1][MFGRSYNPERMF];
//new
int conMFtoGRN[NUMMF+1][NUMGRPERMFN];

//Connectivity matrix of MF to GO Connections
short conMFtoGO[NUMMF+1][MFGOSYNPERMF];
//new
short conMFtoGON[NUMMF+1][MFGOSYNPERMF];

//connectivity matrix of GO to GR connections
//encode GR # and dendrite # in one number
int conGOtoGR[NUMGO+1][GOGRSYNPERGO];
//new
int conGOtoGRN[NUMGO+1][NUMGROUTPERGON];

//connectivity matrix of GR to GO connections
short conGRtoGO[NUMGR+1][GRGOSYNPERGR];

//connectivity matrix of BC to PC connections
char conBCtoPC[NUMBC][BCPCSYNPERBC];

//connectivity matrix of IO coupling connections
char conIOCouple[NUMIO][IOCOUPSYNPERIO];

//connectivity matrix of PC to NC connections
char conPCtoNC[NUMPC][PCNCSYNPERPC];

//-------------end connectivity------------------

//-------------GR input compression variable-----
unsigned char compMask[8];
//-------------end input compression-------------

//---------mossy fiber variables-----------------
const float threshDecayTMF=4;
const float threshDecayMF=1-exp(-TIMESTEP/threshDecayTMF);

char typeMFs[NUMMF+1];

//background frequencies for different contexts
float bgFreqContsMF[NUMCONTEXTS][NUMMF+1];

//increase in frequency during the presence of CS
float incFreqMF[NUMMF+1];
//the start time of that increase in frequency
//correspond to start of CS
short csStartMF[NUMMF+1];
//the end time for that increase in frequency
//not necessarily correspond to the end of CS (phasic fibers)
short csEndMF[NUMMF+1];

//MF thresholds
float threshMF[NUMMF+1];

//MF activity
bool apMF[NUMMF+1];

bool csOnMF[NUMMF+1];

//mossy fiber history for plasticity
bool histMF[NUMMF];

//analysis variables
unsigned short pshMF[PSHNUMBINS][NUMMF];
unsigned short pshMFMax;

//---------end mossy fiber variables-------------


//-----------Granule cell variables--------------
const float gEIncInitGR=0.02*0.22*1;//*0.9;//1;//2;//1;//.1;//*2;//0.02*0.22;//0.19;//0.22;//0.15;//0.11;//0.22;//*2;//6;
const float gIIncConstGR=0.2*0.22*2;//*1;//*2;//*1;//128;//24;//16;//4;//256;//*2;//0.08;//0.1;//0.2;//0.1;//0.027;//0.022;//*2;
const float threshBaseInitGR=-42;
const float threshMaxGR=-20;

const float gEDecayTGR=55;
const float gEDecayGR=exp(-TIMESTEP/gEDecayTGR);
const float gIDecayTGR=50;
const float gIDecayGR=exp(-TIMESTEP/gIDecayTGR);
const float threshDecayTGR=3;
const float threshDecayGR=1-exp(-TIMESTEP/threshDecayTGR);

const float gLeakGR=0.1/(6-TIMESTEP);

unsigned char inputsGRH[NUMGR+NUMGRPAD]; //host input to device

unsigned char apOutGR[NUMGR+NUMGRPAD]; //device output to host

//conduction delay variables
unsigned char delayGOMasksGR[NUMGR][GRGOSYNPERGR];
unsigned char delayBCPCSCMaskGR[NUMGR];
//end conduction delay

//GPU variables
float *vGRGPU;
float *gKCaGRGPU;

float *gEGR1GPU;
float *gEGR2GPU;
float *gEGR3GPU;
float *gEGR4GPU;
float *gEGRGPUSum;

float *gEIncGR1GPU;
float *gEIncGR2GPU;
float *gEIncGR3GPU;
float *gEIncGR4GPU;

float *gIGR1GPU;
float *gIGR2GPU;
float *gIGR3GPU;
float *gIGR4GPU;
float *gIGRGPUSum;

unsigned char *inputsGRGPU;
unsigned char *apOutGRGPU;
unsigned char *apBufGRGPU;
float *threshGRGPU;
float *threshBaseGRGPU;

//conduction delay
unsigned char *delayGOMask1GRGPU;
unsigned char *delayGOMask2GRGPU;
unsigned char *delayBCPCSCMaskGRGPU;
//end conduction delay

//GPU plasticity variables
unsigned char *historyGRGPU;
int histGRGPUPitch;
//end GPU variables
char histBinNGR=-1;

//analysis variables
unsigned short pshGR[PSHNUMBINS][NUMGR];
unsigned short pshGRMax;
unsigned int spikeSumGR[NUMGR];

//QMutex msCountLock;
//---------end Granule cell Variables------------


//------------Golgi cell variables---------------
const float threshMaxGO=-10;
const float threshBaseInitGO=-33;
const float gMFIncInitGO=0.07*0.28/64;//32;//2.6;//192;//64;//48;//*0;///1024;//256;//4096;//8192;//128;//8192;//1024;//48;//64;//32;//16;//128;//1000;//64;//64;//48;//32;//16;//7;//6;//16;//32//16;
const float gGRIncInitGO=0.02*0.28/12;///1.3;//2;//1.7;//2;//2.5;//2.7;//2.6;//2.1;//2.5;//2.5;//2.35;//15;//2.35;//2;//1.6;//1.5;//2.5;//5.5;//8;//32;//5.5;//4.5;//2.5;//2;//4;//8;//14;//18;//22;//16;//64;//14.5;//13.5;//14;//16;//20;//16;//20;//28;//32;//24;//16;//32;//24;//32;//40;//48//64;//24;//16//32;
const float gMGluRScaleGO=0;//0.000015;
const float gMGluRIncScaleGO=0.7;
const float mGluRScaleGO=0.1;
const float gluScaleGO=0.1;

const float gLeakGO=0.02/(6-TIMESTEP);
const float gMFDecayTGO=4.5;
const float gMFDecayGO=exp(-TIMESTEP/gMFDecayTGO);
const float gGRDecayTGO=4.5;
const float gGRDecayGO=exp(-TIMESTEP/gGRDecayTGO);
const float mGluRDecayGO=0.98;
const float gMGluRIncDecayGO=0.98;
const float gMGluRDecayGO=0.98;
const float gluDecayGO=0.98;

const float threshDecayTGO=20;
const float threshDecayGO=1-exp(-TIMESTEP/threshDecayTGO);

float vGO[NUMGO+1];
float threshGO[NUMGO+1];
float threshBaseGO[NUMGO+1];
bool apGO[NUMGO+1];
unsigned short inputMFGO[NUMGO+1];
unsigned short inputGRGO[NUMGO+1];
float gMFGO[NUMGO+1];
float gGRGO[NUMGO+1];
float gMGluRGO[NUMGO+1];

float gMFIncGO[NUMGO+1];
float gGRIncGO[NUMGO+1];
float gMGluRIncGO[NUMGO+1];
float mGluRGO[NUMGO+1];
float gluGO[NUMGO+1];

//analysis variables
unsigned short pshGO[PSHNUMBINS][NUMGO];
unsigned short pshGOMax;
unsigned int spikeSumGO[NUMGO];
//QMutex spikeSumGOLock;
//----------end Golgi cell variables-------------

//----------purkinje cell variables--------------
const float threshDecayTPC=5;
const float threshDecayPC=1-exp(-TIMESTEP/threshDecayTPC);
const float gPFDecayTPC=4.15;
const float gPFDecayPC=exp(-TIMESTEP/gPFDecayTPC);
const float gBCDecayTPC=5;
const float gBCDecayPC=exp(-TIMESTEP/gBCDecayTPC);
const float gSCDecayTPC=4.15;
const float GSCDecayPC=exp(-TIMESTEP/gSCDecayTPC);
const float gLeakPC=0.2/(6-TIMESTEP);

float pfSynWeightPC[NUMGR];
QMutex pfSynWeightPCLock;

float inputSumPFPC[NUMPC]; //copy from device
unsigned char inputBCPC[NUMPC];
float gPFPC[NUMPC];
float gBCPC[NUMPC];
float gSCPC[NUMPC][SCPCSYNPERPC];
float vPC[NUMPC];
float threshPC[NUMPC];
float threshBasePC[NUMPC];
bool apPC[NUMPC];

//GPU variables
float *pfSynWeightPCGPU;
float *inputPFPCGPU; //multiplication of weights with GR inputs, 32*NUMGR/NUMPC elements
unsigned int iPFPCGPUPitch;
float *tempSumPFPCGPU; //variable for holding temporary data for reduction
unsigned int tempSumPFPCGPUPitch;
float *inputSumPFPCGPU; //send to host, NUMPC elements
//end GPU variables

//plasticity variables
//for MF-NC plasticity
short histAllAPPC[NUMHISTBINSPC];
short histSumAllAPPC=0;
char histBinNPC=0;
short allAPPC=0;

//analysis variables
unsigned int pshPC[PSHNUMBINS][NUMPC];
unsigned int pshPCMax;
unsigned int spikeSumPC[NUMPC];
//QMutex spikeSumPCLock;
//----------end purkinje cell vars---------------

//----------basket cell variables---------------
const float gLeakBC=0.2/(6-TIMESTEP);
const float gPFDecayTBC=4.15;
const float gPFDecayBC=exp(-TIMESTEP/gPFDecayTBC);
const float gPCDecayTBC=5;
const float gPCDecayBC=exp(-TIMESTEP/gPCDecayTBC);
const float threshDecayTBC=10;
const float threshDecayBC=1-exp(-TIMESTEP/threshDecayTBC);

unsigned short inputSumPFBC[NUMBC]; //copy from GPU
unsigned char inputPCBC[NUMBC];
float gPFBC[NUMBC];
float gPCBC[NUMBC];
float threshBC[NUMBC];
float vBC[NUMBC];
bool apBC[NUMBC];

//GPU variables
unsigned short *inputPFBCGPU;
unsigned int iPFBCGPUPitch;
unsigned short *tempSumPFBCGPU;
unsigned int tempSumPFBCGPUPitch;
unsigned short *inputSumPFBCGPU; //send to host
//end GPU variables

//analysis variables
unsigned int pshBC[PSHNUMBINS][NUMBC];
unsigned int pshBCMax;
unsigned int spikeSumBC[NUMBC];
//QMutex spikeSumBCLock;
//----------end basket cell variables-----------

//----------stellate cell variables-------------
const float gLeakSC=0.2/(6-TIMESTEP);
const float gPFDecayTSC=4.15;
const float gPFDecaySC=exp(-TIMESTEP/gPFDecayTSC);
const float threshDecayTSC=22;
const float threshDecaySC=1-exp(-TIMESTEP/threshDecayTSC);

unsigned short inputSumPFSC[NUMSC]; //copy from GPU
float gPFSC[NUMSC];
float threshSC[NUMSC];
float vSC[NUMSC];
bool apSC[NUMSC];

//GPU variables
unsigned short *inputPFSCGPU;
unsigned int iPFSCGPUPitch;
unsigned short *tempSumPFSCGPU;
unsigned int tempSumPFSCGPUPitch;
unsigned short *inputSumPFSCGPU; //send to host
//end GPU variables

//analysis variables
unsigned int pshSC[PSHNUMBINS][NUMSC];
unsigned int pshSCMax;
unsigned int spikeSumSC[NUMSC];
//QMutex spikeSumSCLock;

//----------end stellate cell variables---------

//---------- inferior olivary cell variables-----
const float caDecayIO=0.96;
const float gLtCaHTIO=51;
const float gLtCaHMaxVIO=87;
const float gLtCAMMaxVIO=-56;
const float gHMaxVIO=78;
const float gHTauVIO=69;
const float threshDecayIO=1-exp(1-TIMESTEP/THRESHTAUIO);

bool inputNCIO[NUMIO][NCIOSYNPERIO];
float gNCIO[NUMIO][NCIOSYNPERIO];
float gHIO[NUMIO];
float gLtCaIO[NUMIO];
float caIO[NUMIO];
float gKCaIO[NUMIO];

float gLtCaHIO[NUMIO];

float threshIO[NUMIO];
float vIO[NUMIO];
float vCoupIO[NUMIO];
bool apIO[NUMIO];

int plasticityPFPCTimerIO[NUMIO];
//---------end IO variables---------------------

//---------- Nucleus cell variables-------------
const float mfNMDADecayTNC=50;
const float mfNMDADecayNC=exp(-TIMESTEP/mfNMDADecayTNC);
const float mfAMPADecayTNC=6;
const float mfAMPADecayNC=exp(-TIMESTEP/mfAMPADecayTNC);
const float gMFNMDAINCNC=1-exp(-TIMESTEP/3.0);
const float gMFAMPAINCNC=gMFNMDAINCNC;
const float gPCDecayTNC=4.15;
const float gPCDecayNC=exp(-TIMESTEP/gPCDecayTNC);
const float threshDecayTNC=5;
const float threshDecayNC=1-exp(-TIMESTEP/threshDecayTNC);

bool inputPCNC[NUMNC][PCNCSYNPERNC];
float gPCNC[NUMNC][PCNCSYNPERNC];
float gPCScaleNC[NUMNC][PCNCSYNPERNC];

bool inputMFNC[NUMNC][MFNCSYNPERNC];
float mfSynWeightNC[NUMNC][MFNCSYNPERNC];
float mfNMDANC[NUMNC][MFNCSYNPERNC];
float mfAMPANC[NUMNC][MFNCSYNPERNC];
float gMFNMDANC[NUMNC][MFNCSYNPERNC];
float gMFAMPANC[NUMNC][MFNCSYNPERNC];

float threshNC[NUMNC];
float vNC[NUMNC];
bool apNC[NUMNC];
float synIOReleasePNC[NUMNC];

//plasticity variables for MFNC synapses
bool noLTPMFNC=false;
bool noLTDMFNC=false;
float mfSynWChangeNC[NUMNC][MFNCSYNPERNC];

//analysis variable
unsigned int spikeSumNC[NUMNC];
//QMutex spikeSumNCLock;
//-----------end NC variables------------------

//-------------simulation variables--------------
char activeContext;
char activeCS;

short const csOnset[4]={1500, 1500, 1500, 1500};
short const csDuration[4]={1000, 1000, 1000, 1000};
//-----------end simulation variables------------

//=============end globally used variables===============

int main(int, char **);

#endif /* MAIN_H_ */
