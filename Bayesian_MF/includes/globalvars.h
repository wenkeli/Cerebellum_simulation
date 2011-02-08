/*
 * globalvars.h
 *
 *  Created on: Feb 7, 2011
 *      Author: consciousness
 */

#ifndef GLOBALVARS_H_
#define GLOBALVARS_H_

#include "parameters.h"

extern float ratesMFInputA[NUMMF];
extern float ratesMFInputB[NUMMF];
extern unsigned int spikeCountsMF[NUMMF];
extern float sVs[NUMTRIALS];

extern const float threshDecayMF;

#endif /* GLOBALVARS_H_ */
