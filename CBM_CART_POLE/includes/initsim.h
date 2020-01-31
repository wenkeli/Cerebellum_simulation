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

void resetVars();
void resetActiveCS(char aCS);
void initSim();

void initCUDA();
void initGRCUDA();
void initPCCUDA();
void initBCCUDA();
void initSCCUDA();

void assignMF();
void initMF(CRandomSFMT0 &randGen);
void updateMFCSOn();

#endif /* INITSIM_H_ */
