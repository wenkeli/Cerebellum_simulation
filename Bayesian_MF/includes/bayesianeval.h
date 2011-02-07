/*
 * bayesianeval.h
 *
 *  Created on: Feb 7, 2011
 *      Author: consciousness
 */

#ifndef BAYESIANEVAL_H_
#define BAYESIANEVAL_H_
#include "parameters.h"
#include "globalvars.h"

#ifdef INTELCC
#include <mathimf.h>
#else
#include <math.h>
#endif

void bayesianCalcSV(int);

#endif /* BAYESIANEVAL_H_ */
