/*
 * mfactivities.h
 *
 *  Created on: Feb 7, 2011
 *      Author: consciousness
 */

#ifndef MFACTIVITIES_H_
#define MFACTIVITIES_H_

#include <cstdlib>
#include <string.h>
#ifdef INTELCC
#include <mathimf.h>
#else
#include <math.h>
#endif

#include "randomc.h"
#include "sfmt.h"
#include "parameters.h"
#include "globalvars.h"

void calcMFActsPoisson(int, CRandomSFMT0 &);
void calcMFActsRegenPoisson(int, CRandomSFMT0 &);


#endif /* MFACTIVITIES_H_ */
