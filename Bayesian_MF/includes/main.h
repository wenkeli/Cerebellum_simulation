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

#include "parameters.h"
#include "bayesianeval.h"
#include "mfactivities.h"
#include "readinputs.h"
#include "writeoutputs.h"

int ratesMFInputA[NUMMF];
int ratesMFInputB[NUMMF];
int spikeCountsMF[NUMMF];
float sVs[NUMTRIALS];


int main(int, char **);

#endif /* MAIN_H_ */
