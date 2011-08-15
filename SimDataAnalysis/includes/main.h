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
#include "datamodules/psh.h"
#include "datamodules/simerrorec.h"
#include "datamodules/simexternalec.h"
#include "datamodules/siminnet.h"
#include "datamodules/simmfinputec.h"
#include "datamodules/simmzone.h"
#include "datamodules/simoutputec.h"
#include "analysismodules/grpshpopanalysis.h"

PSHData *mfPSH;
PSHData *goPSH;
PSHData *grPSH;
PSHData *scPSH;

PSHData *bcPSH[NUMMZONES];
PSHData *pcPSH[NUMMZONES];
PSHData *ioPSH[NUMMZONES];
PSHData *ncPSH[NUMMZONES];

SimErrorEC *simErrMod[NUMMZONES];
SimOutputEC *simOutMod[NUMMZONES];
SimExternalEC *simExternalMod;
SimMFInputEC *simMFInputMod;
SimInNet *simInNetMod;
SimMZone *simMZoneMod[NUMMZONES];

GRPSHPopAnalysis *grPopTimingAnalysis;

SpikeRateAnalysis *mfSR;
SpikeRateAnalysis *goSR;
SpikeRateAnalysis *grSR;
SpikeRateAnalysis *scSR;

SpikeRateAnalysis *bcSR[NUMMZONES];
SpikeRateAnalysis *pcSR[NUMMZONES];
SpikeRateAnalysis *ioSR[NUMMZONES];
SpikeRateAnalysis *ncSR[NUMMZONES];

int main(int argc, char **argv);

#endif /* MAIN_H_ */
