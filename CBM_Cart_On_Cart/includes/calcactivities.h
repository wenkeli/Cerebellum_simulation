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

#include "common.h"
#include "globalvars.h"

#include "mfinputmodules/mfinputbase.h"

#include "errorinputmodules/errorinputbase.h"

#include "outputmodules/outputbase.h"

#include "externalmodules/externalbase.h"

#include "mzonemodules/mzone.h"

#include "innetmodules/innet.h"

void calcCellActivities(short time, CRandomSFMT0 &randGen);

#endif /* CALCACTIVITIES_H_ */
