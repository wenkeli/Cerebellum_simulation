/*
 * globalvars.h
 *
 *  Created on: Oct 27, 2009
 *      Author: wen
 */
#ifndef GLOBALVARS_H_
#define GLOBALVARS_H_

#include <QtCore/QMutex>
#include "parameters.h"
#include <vector>
#include <iostream>
#include <fstream>

using namespace std;

//control and thread variables
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

extern QMutex accessPSHLock;

extern QMutex accessConnLock;

extern QMutex accessSpikeSumLock;
//end control and thread variables

//simulation analysis variables - not cell type specific
extern unsigned int numTrials;
extern unsigned int msCount;
//end non cell type specific analysis variables

//simulation file outputs
extern ofstream simOut;
extern ofstream pshOut;
extern ifstream simIn;
//end file outputs

//debug variables
#ifdef DEBUG
extern std::vector<int> incompGRsGOGR;
extern std::vector<int> incompGOsGOGR;
#endif
//end debug variables

//----------connectivity data-------------------
//keep track of how many connections were made
extern short numSynMFtoGR[NUMMF+1];
//new
extern short numMFtoGRN[NUMMF+1];

extern char numSynMFtoGO[NUMMF+1];
//new
extern char numMFtoGON[NUMMF+1];

extern short numSynGOtoGR[NUMGO+1];
//new
extern short numGOtoGRN[NUMGO+1];

extern char numSynGRtoGO[NUMGR+1];

//Connectivity matrix of MF to GR connections
//encode GR # and dendrite # in one number
extern int conMFtoGR[NUMMF+1][MFGRSYNPERMF];
//new
extern int conMFtoGRN[NUMMF+1][NUMGRPERMFN];

//Connectivity matrix of MF to GO Connections
extern short conMFtoGO[NUMMF+1][MFGOSYNPERMF];
//new
extern short conMFtoGON[NUMMF+1][MFGOSYNPERMF];

//connectivity matrix of GO to GR connections
//encode GR # and dendrite # in one number
extern int conGOtoGR[NUMGO+1][GOGRSYNPERGO];
//new
extern int conGOtoGRN[NUMGO+1][NUMGROUTPERGON];

//connectivity matrix of GR to GO connections
extern short conGRtoGO[NUMGR+1][GRGOSYNPERGR];

//connectivity matrix of BC to PC connections
extern char conBCtoPC[NUMBC][BCPCSYNPERBC];

//connectivity matrix of IO coupling connections
extern char conIOCouple[NUMIO][IOCOUPSYNPERIO];

//connectivity matrix of PC to NC connections
extern char conPCtoNC[NUMPC][PCNCSYNPERPC];

//-------------end connectivity------------------

//-------------GR input compression variable-----
extern unsigned char compMask[8];
//-------------end input compression-------------

//---------mossy fiber variables-----------------
extern const float threshDecayTMF;
extern const float threshDecayMF;

extern char typeMFs[NUMMF+1];

//background frequencies for different contexts
extern float bgFreqContsMF[NUMCONTEXTS][NUMMF+1];

//increase in frequency during the presence of CS
extern float incFreqMF[NUMMF+1];
//the start time of that increase in frequency
//correspond to start of CS
extern short csStartMF[NUMMF+1];
//the end time for that increase in frequency
//not necessarily correspond to the end of CS (phasic fibers)
extern short csEndMF[NUMMF+1];

//MF thresholds
extern float threshMF[NUMMF+1];

//MF activity
extern bool apMF[NUMMF+1];

extern bool csOnMF[NUMMF+1];

//mossy fiber history for plasticity
extern bool histMF[NUMMF];

//analysis variables
extern unsigned short pshMF[PSHNUMBINS][NUMMF];
extern unsigned short pshMFMax;
//---------end mossy fiber variables-------------


//-----------Granule cell variables--------------
extern const float gEIncInitGR;
extern const float gIIncConstGR;
extern const float threshBaseInitGR;
extern const float threshMaxGR;

extern const float gEDecayTGR;
extern const float gEDecayGR;
extern const float gIDecayTGR;
extern const float gIDecayGR;
extern const float threshDecayTGR;
extern const float threshDecayGR;

extern const float gLeakGR;

extern unsigned char inputsGRH[NUMGR+NUMGRPAD]; //host input to device

extern unsigned char apOutGR[NUMGR+NUMGRPAD]; //device output to host

//conduction delay variables
extern unsigned char delayGOMasksGR[NUMGR][GRGOSYNPERGR];
extern unsigned char delayBCPCSCMaskGR[NUMGR];
//end conduction

//GPU variables
extern float *vGRGPU;
extern float *gKCaGRGPU;

extern float *gEGR1GPU;
extern float *gEGR2GPU;
extern float *gEGR3GPU;
extern float *gEGR4GPU;
extern float *gEGRGPUSum;

extern float *gEIncGR1GPU;
extern float *gEIncGR2GPU;
extern float *gEIncGR3GPU;
extern float *gEIncGR4GPU;

extern float *gIGR1GPU;
extern float *gIGR2GPU;
extern float *gIGR3GPU;
extern float *gIGR4GPU;
extern float *gIGRGPUSum;

extern unsigned char *inputsGRGPU;
extern unsigned char *apOutGRGPU;
extern unsigned char *apBufGRGPU;
extern float *threshGRGPU;
extern float *threshBaseGRGPU;

//conduction delay
extern unsigned char *delayGOMask1GRGPU;
extern unsigned char *delayGOMask2GRGPU;
extern unsigned char *delayBCPCSCMaskGRGPU;
//end conduction delay

//GPU plasticity variables
extern unsigned char *historyGRGPU;
extern int histGRGPUPitch;
//end GPU variables

//host plasticity variables
extern char histBinNGR;

//analysis variables
extern unsigned short pshGR[PSHNUMBINS][NUMGR];
extern unsigned short pshGRMax;
extern unsigned int spikeSumGR[NUMGR];

//extern QMutex msCountLock;
//---------end Granule cell Variables------------

//------------Golgi cell variables---------------
extern const float threshMaxGO;
extern const float threshBaseInitGO;
extern const float gMFIncInitGO;
extern const float gGRIncInitGO;
extern const float gMGluRScaleGO;
extern const float gMGluRIncScaleGO;
extern const float mGluRScaleGO;
extern const float gluScaleGO;

extern const float gLeakGO;
extern const float gMFDecayTGO;
extern const float gMFDecayGO;
extern const float gGRDecayTGO;
extern const float gGRDecayGO;
extern const float mGluRDecayGO;
extern const float gMGluRIncDecayGO;
extern const float gMGluRDecayGO;
extern const float gluDecayGO;

extern const float threshDecayTGO;
extern const float threshDecayGO;

extern float vGO[NUMGO+1];
extern float threshGO[NUMGO+1];
extern float threshBaseGO[NUMGO+1];
extern bool apGO[NUMGO+1];
extern unsigned short inputMFGO[NUMGO+1];
extern unsigned short inputGRGO[NUMGO+1];
extern float gMFGO[NUMGO+1];
extern float gGRGO[NUMGO+1];
extern float gMGluRGO[NUMGO+1];

extern float gMFIncGO[NUMGO+1];
extern float gGRIncGO[NUMGO+1];
extern float gMGluRIncGO[NUMGO+1];
extern float mGluRGO[NUMGO+1];
extern float gluGO[NUMGO+1];

//analysis variables
extern unsigned short pshGO[PSHNUMBINS][NUMGO];
extern unsigned short pshGOMax;
extern unsigned int spikeSumGO[NUMGO];
//extern QMutex spikeSumGOLock;
//----------end Golgi cell variables-------------

//----------purkinje cell variables--------------
extern const float threshDecayTPC;
extern const float threshDecayPC;
extern const float gPFDecayTPC;
extern const float gPFDecayPC;
extern const float gBCDecayTPC;
extern const float gBCDecayPC;
extern const float gSCDecayTPC;
extern const float GSCDecayPC;
extern const float gLeakPC;

extern float pfSynWeightPC[NUMGR];
extern QMutex pfSynWeightPCLock;

extern float inputSumPFPC[NUMPC]; //copy from device
extern unsigned char inputBCPC[NUMPC];
extern float gPFPC[NUMPC];
extern float gBCPC[NUMPC];
extern float gSCPC[NUMPC][SCPCSYNPERPC];
extern float vPC[NUMPC];
extern float threshPC[NUMPC];
extern float threshBasePC[NUMPC];
extern bool apPC[NUMPC];

//extern float tempSumArrDebug[32];

//GPU variables
extern float *pfSynWeightPCGPU;
extern float *inputPFPCGPU; //multiplication of weights with GR inputs, NUMPC*PFPCSYNPERPC elements
extern unsigned int iPFPCGPUPitch;
extern float *tempSumPFPCGPU; //variable for holding temporary data for reduction
extern unsigned int tempSumPFPCGPUPitch;
extern float *inputSumPFPCGPU; //send to host, NUMPC elements
//end GPU variables

//plasticity variables
//for MF-NC plasticity
extern short histAllAPPC[NUMHISTBINSPC];
extern short histSumAllAPPC;
extern char histBinNPC;
extern short allAPPC;

//analysis variables
extern unsigned int pshPC[PSHNUMBINS][NUMPC];
extern unsigned int pshPCMax;
extern unsigned int spikeSumPC[NUMPC];
//extern QMutex spikeSumPCLock;
//----------end purkinje cell vars---------------

//----------basket cell variables---------------
extern const float gLeakBC;
extern const float gPFDecayTBC;
extern const float gPFDecayBC;
extern const float gPCDecayTBC;
extern const float gPCDecayBC;
extern const float threshDecayTBC;
extern const float threshDecayBC;

extern unsigned short inputSumPFBC[NUMBC]; //copy from GPU
extern unsigned char inputPCBC[NUMBC];
extern float gPFBC[NUMBC];
extern float gPCBC[NUMBC];
extern float threshBC[NUMBC];
extern float vBC[NUMBC];
extern bool apBC[NUMBC];

//GPU variables
extern unsigned short *inputPFBCGPU;
extern unsigned int iPFBCGPUPitch;
extern unsigned short *tempSumPFBCGPU;
extern unsigned int tempSumPFBCGPUPitch;
extern unsigned short *inputSumPFBCGPU; //send to host
//end GPU variables

//analysis variables
extern unsigned int pshBC[PSHNUMBINS][NUMBC];
extern unsigned int pshBCMax;
extern unsigned int spikeSumBC[NUMBC];
//extern QMutex spikeSumBCLock;
//----------end basket cell variables-----------

//----------stellate cell variables-------------
extern const float gLeakSC;
extern const float gPFDecayTSC;
extern const float gPFDecaySC;
extern const float threshDecayTSC;
extern const float threshDecaySC;

extern unsigned short inputSumPFSC[NUMSC]; //copy from GPU
extern float gPFSC[NUMSC];
extern float threshSC[NUMSC];
extern float vSC[NUMSC];
extern bool apSC[NUMSC];

//GPU variables
extern unsigned short *inputPFSCGPU;
extern unsigned int iPFSCGPUPitch;
extern unsigned short *tempSumPFSCGPU;
extern unsigned int tempSumPFSCGPUPitch;
extern unsigned short *inputSumPFSCGPU; //send to host
//end GPU variables

//analysis variables
extern unsigned int pshSC[PSHNUMBINS][NUMSC];
extern unsigned int pshSCMax;
extern unsigned int spikeSumSC[NUMSC];
//extern QMutex spikeSumSCLock;
//----------end stellate cell variables---------

//---------- inferior olivary cell variables-----
extern const float caDecayIO;
extern const float gLtCaHTIO; //hTauCa
extern const float gLtCaHMaxVIO; //hMaxVP
extern const float gLtCAMMaxVIO; //mCaVP
extern const float gHMaxVIO; //iHMaxVP
extern const float gHTauVIO; //IHTauVP
extern const float threshDecayIO; //for non-oscillatory model

extern bool inputNCIO[NUMIO][NCIOSYNPERIO];
extern float gNCIO[NUMIO][NCIOSYNPERIO];
extern float gHIO[NUMIO];
extern float gLtCaIO[NUMIO];
extern float caIO[NUMIO];
extern float gKCaIO[NUMIO];

extern float gLtCaHIO[NUMIO]; //hCa

extern float threshIO[NUMIO];
extern float vIO[NUMIO];
extern float vCoupIO[NUMIO];
extern bool apIO[NUMIO];

extern int plasticityPFPCTimerIO[NUMIO];
//---------end IO variables---------------------

//---------- Nucleus cell variables-------------
extern const float mfNMDADecayTNC;
extern const float mfNMDADecayNC;
extern const float mfAMPADecayTNC;
extern const float mfAMPADecayNC;
extern const float gMFNMDAINCNC;
extern const float gMFAMPAINCNC;
extern const float gPCDecayTNC;
extern const float gPCDecayNC;
extern const float threshDecayTNC;
extern const float threshDecayNC;

extern bool inputPCNC[NUMNC][PCNCSYNPERNC];
extern float gPCNC[NUMNC][PCNCSYNPERNC];
extern float gPCScaleNC[NUMNC][PCNCSYNPERNC];

extern bool inputMFNC[NUMNC][MFNCSYNPERNC];
extern float mfSynWeightNC[NUMNC][MFNCSYNPERNC];
extern float mfNMDANC[NUMNC][MFNCSYNPERNC];
extern float mfAMPANC[NUMNC][MFNCSYNPERNC];
extern float gMFNMDANC[NUMNC][MFNCSYNPERNC];
extern float gMFAMPANC[NUMNC][MFNCSYNPERNC];

extern float threshNC[NUMNC];
extern float vNC[NUMNC];
extern bool apNC[NUMNC];
extern float synIOReleasePNC[NUMNC];

//plasticity variables for MFNC synapses
extern bool noLTPMFNC;
extern bool noLTDMFNC;
extern float mfSynWChangeNC[NUMNC][MFNCSYNPERNC];

//analysis variable
extern unsigned int spikeSumNC[NUMNC];
//extern QMutex spikeSumNCLock;
//-----------end NC variables------------------

//-------------simulation variables--------------
extern char activeContext;
extern char activeCS;

//CS onset times and duration
extern short const csOnset[4];
extern short const csDuration[4];
//-----------end simulation variables------------


#endif /* GLOBALVARS_H_ */
