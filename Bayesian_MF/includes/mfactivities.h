/*
 * mfactivities.h
 *
 *  Created on: Feb 7, 2011
 *      Author: consciousness
 */

#ifndef MFACTIVITIES_H_
#define MFACTIVITIES_H_

#ifdef INTELCC
#include <mathimf.h>
#else
#include <math.h>
#endif

#include "parameters.h"
#include "globalvars.h"

void calcMFActsPoisson();
void calcMFActsRegenPoisson();


#endif /* MFACTIVITIES_H_ */
