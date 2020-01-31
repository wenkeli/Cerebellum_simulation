/*
 * cbmsimcore.h
 *
 *  Created on: Dec 14, 2011
 *      Author: consciousness
 */

#ifndef CBMSIMCORE_H_
#define CBMSIMCORE_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <limits.h>
#include <time.h>

#include <CXXToolsInclude/stdDefinitions/pstdint.h>
#include <CBMStateInclude/interfaces/cbmstate.h>
#include <CXXToolsInclude/randGenerators/sfmt.h>

#include "innetinterface.h"
#include "mzoneinterface.h"

#include "../mzonemodules/mzone.h"

#include "../innetmodules/innet.h"
#include "../innetmodules/innetallgrmfgo.h"

class CBMSimCore
{
public:
	CBMSimCore(CBMState *state, int gpuIndStart=-1, int numGPUP2=-1);
	CBMSimCore(CBMState *state, int *mzoneRSeed, int gpuIndStart=-1, int numGPUP2=-1);

	~CBMSimCore();

	void calcActivity();

	void updateMFInput(const ct_uint8_t *mfIn);
	void updateErrDrive(unsigned int zoneN, float errDriveRelative);

	void writeToState();
	void writeToState(std::fstream& outfile);

	InNetInterface* getInputNet();
	MZoneInterface** getMZoneList();

protected:
	void initCUDA();
	void initAuxVars();

	void syncCUDA(std::string title);

	ct_uint32_t numZones;

	CBMState *simState;

	InNet *inputNet;
	MZone **zones;

//	cudaStream_t streams[8];
	cudaStream_t **streams;
	int gpuIndStart;
	int numGPUs;

private:
	CBMSimCore();

	void construct(CBMState *state, int *mzoneRSeed, int gpuIndStart, int numGPUP2);

	unsigned long curTime;
};

#endif /* CBMSIMCORE_H_ */
