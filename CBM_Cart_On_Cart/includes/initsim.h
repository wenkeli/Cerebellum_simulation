/*
 * initsim.h
 *
 *  Created on: Feb 25, 2009
 *      Author: wen
 */

#ifndef INITSIM_H_
#define INITSIM_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "globalvars.h"

#include "mfinputmodules/mfinputbase.h"
#include "mfinputmodules/mfinputec.h"
#include "mfinputmodules/mfinputcp.h"
#include "errorinputmodules/errorinputbase.h"
#include "errorinputmodules/errorinputec.h"
#include "errorinputmodules/errorinputcp.h"
#include "outputmodules/outputbase.h"
#include "outputmodules/outputec.h"
#include "outputmodules/outputcp.h"
#include "externalmodules/externalbase.h"
#include "externalmodules/cartpole.h"
#include "externalmodules/externaldummy.h"
#include "mzonemodules/mzone.h"
#include "innetmodules/innet.h"
#include "innetmodules/innetnogo.h"
#include "innetmodules/innetnogrgo.h"
#include "innetmodules/innetnomfgo.h"
#include "innetmodules/innetsparsegrgo.h"
#include "analysismodules/psh.h"
#include "analysismodules/pshgpu.h"

void newSim();

void readSimIn(ifstream &infile);

void writeSimOut(ofstream &outfile);

void writePSHOut(ofstream &outfile);

void cleanSim();

#endif /* INITSIM_H_ */
