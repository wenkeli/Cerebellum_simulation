/*
 * calcactivities.h
 *
 *  Created on: Feb 12, 2010
 *      Author: wen
 */

#ifndef CALCACTIVITIES_H_
#define CALCACTIVITIES_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "grKernels.h"
#include "pcKernels.h"
#include "bcKernels.h"
#include "scKernels.h"
#include "ioKernels.h"

#include "common.h"
#include "globalvars.h"

void calcCellActivities(short time, CRandomSFMT0 &randGen);

#endif /* CALCACTIVITIES_H_ */
