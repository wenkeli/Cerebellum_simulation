/*
 * main.h
 *
 *  Created on: Feb 7, 2011
 *      Author: consciousness
 */

#ifndef MAIN_H_
#define MAIN_H_


#ifdef INTELCC
#include <mathimf.h>
#else
#include <math.h>
#endif

#include "randomc.h"
#include "sfmt.h"
#include "parameters.h"
#include "bayesianeval.h"
#include "mfactivities.h"
#include "readinputs.h"
#include "writeoutputs.h"

float ratesMFInputA[NUMMF];
float ratesMFInputB[NUMMF];
unsigned int spikeCountsMF[NUMMF];
float sVs[NUMTRIALS];

const float threshDecayTMF=4;
const float threshDecayMF=1-exp(-TIMESTEPMS/threshDecayTMF);

int main(int, char **);

#endif /* MAIN_H_ */
