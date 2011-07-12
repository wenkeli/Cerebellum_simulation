/*
 * globalvars.h
 *
 *  Created on: Jul 7, 2011
 *      Author: consciousness
 */

#ifndef GLOBALVARS_H_
#define GLOBALVARS_H_
#include "datamodules/psh.h"
#include "analysismodules/grpshpopanalysis.h"
#include "common.h"

extern PSHData *mfPSH;
extern PSHData *goPSH;
extern PSHData *grPSH;
extern PSHData *scPSH;

extern PSHData *bcPSH[NUMMZONES];
extern PSHData *pcPSH[NUMMZONES];
extern PSHData *ioPSH[NUMMZONES];
extern PSHData *ncPSH[NUMMZONES];

extern GRPSHPopAnalysis *grPopTimingAnalysis;

#endif /* GLOBALVARS_H_ */
