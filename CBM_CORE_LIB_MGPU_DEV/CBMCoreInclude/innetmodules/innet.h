/*
 * InNet.h
 *
 *  Created on: Jun 21, 2011
 *      Author: consciousness
 */

#ifndef INNET_H_
#define INNET_H_

#ifdef INTELCC
#include <mathimf.h>
#else //otherwise use standard math library
#include <math.h>
#endif

#include <vector>
#include <string.h>
#include <sstream>
#include <iostream>
#include <fstream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>

#include <CXXToolsInclude/stdDefinitions/pstdint.h>
#include <CXXToolsInclude/randGenerators/sfmt.h>
#include <CXXToolsInclude/memoryMgmt/dynamic2darray.h>

#include <CBMStateInclude/params/connectivityparams.h>
#include <CBMStateInclude/params/activityparams.h>
#include <CBMStateInclude/state/innetconnectivitystate.h>
#include <CBMStateInclude/state/innetactivitystate.h>

#include "../cuda/kernels.h"

#include "../interface/innetinterface.h"

class InNet : virtual public InNetInterface
{
public:
	InNet(ConnectivityParams *conParams, ActivityParams *actParams,
			InNetConnectivityState *conState, InNetActivityState *actState,
			int gpuIndStart, int numGPUs);
	virtual ~InNet();

	void writeToState();

	virtual void setGIncGRtoGO(float inc);
	virtual void resetGIncGRtoGO();

	virtual const ct_uint8_t* exportAPSC();
	virtual const ct_uint8_t* exportAPGO();
	virtual const ct_uint8_t* exportAPGR();

	virtual const ct_uint8_t* exportHistMF();
	virtual const ct_uint32_t* exportAPBufMF();
	virtual const ct_uint32_t* exportAPBufGR();
	virtual const ct_uint32_t* exportAPBufGO();
	virtual const ct_uint32_t* exportAPBufSC();

	virtual const float* exportVmGR();
	virtual const float* exportVmGO();
	virtual const float* exportVmSC();
	virtual const float* exportGESumGR();
	virtual const float* exportGISumGR();

	virtual const ct_uint32_t* exportSumGRInputGO();
	virtual const float* exportSumGOInputGO();
	virtual const float* exportGOOutSynScaleGOGO();

	virtual const ct_uint32_t* exportPFBCSum();

	virtual ct_uint32_t** getApBufGRGPUPointer();
	virtual ct_uint32_t** getDelayBCPCSCMaskGPUPointer();
	virtual ct_uint64_t** getHistGRGPUPointer();

	virtual ct_uint32_t** getGRInputGOSumHPointer();

	virtual void updateMFActivties(const ct_uint8_t *actInMF);
	virtual void calcGOActivities();
	virtual void calcSCActivities();
	virtual void updateMFtoGOOut();
	virtual void updateGOtoGOOut();
	virtual void resetMFHist(unsigned long t);

	virtual void runGRActivitiesCUDA(cudaStream_t **sts, int streamN);
	virtual void runSumPFBCCUDA(cudaStream_t **sts, int streamN);
	virtual void runSumPFSCCUDA(cudaStream_t **sts, int streamN);
	virtual void runSumGRGOOutCUDA(cudaStream_t **sts, int streamN);
	virtual void cpyAPMFHosttoGPUCUDA(cudaStream_t **sts, int streamN);
	virtual void cpyAPGOHosttoGPUCUDA(cudaStream_t **sts, int streamN);
	virtual void runUpdateMFInGRCUDA(cudaStream_t **sts, int streamN);
	virtual void runUpdateGOInGRCUDA(cudaStream_t **sts, int streamN);
	virtual void runUpdatePFBCSCOutCUDA(cudaStream_t **sts, int streamN);
	virtual void runUpdateGROutGOCUDA(cudaStream_t **sts, int streamN);
	virtual void cpyPFBCSumGPUtoHostCUDA(cudaStream_t **sts, int streamN);
	virtual void cpyPFSCSumGPUtoHostCUDA(cudaStream_t **sts, int streamN);
	virtual void cpyGRGOSumGPUtoHostCUDA(cudaStream_t **sts, int streamN);
	virtual void cpyGRGOSumGPUtoHostCUDA(cudaStream_t **sts, int streamN, ct_uint32_t **grInputGOSumHost);
	virtual void runUpdateGRHistoryCUDA(cudaStream_t **sts, int streamN, unsigned long t);

protected:
	virtual void initCUDA();
	virtual void initMFCUDA();
	virtual void initGRCUDA();
	virtual void initGOCUDA();
	virtual void initBCCUDA();
	virtual void initSCCUDA();

//	virtual void connectNetwork();

	ConnectivityParams *cp;
	ActivityParams *ap;

	InNetConnectivityState *cs;
	InNetActivityState *as;

	CRandomSFMT0 *randGen;

	int gpuIndStart;
	int numGPUs;
	int numGRPerGPU;

	unsigned int calcGRActNumGRPerB;
	unsigned int calcGRActNumBlocks;

	unsigned int updateGRGOOutNumGRPerR;
	unsigned int updateGRGOOutNumGRRows;

	unsigned int sumGRGOOutNumGOPerB;
	unsigned int sumGRGOOutNumBlocks;

	unsigned int updateMFInGRNumGRPerB;
	unsigned int updateMFInGRNumBlocks;

	unsigned int updateGOInGRNumGRPerB;
	unsigned int updateGOInGRNumBlocks;

	unsigned int updatePFBCSCNumGRPerB;
	unsigned int updatePFBCSCNumBlocks;

	unsigned int updateGRHistNumGRPerB;
	unsigned int updateGRHistNumBlocks;

//	float *testA;
//	float *testB;
//	float *testC;

	//mossy fibers
	//gpu related variables
	ct_uint32_t **apMFH;
	ct_uint32_t **apMFGPU;
	//end gpu related variables

	//---------golgi cell variables
	//gpu related variables
	//GPU parameters

	ct_uint32_t **apGOH;
	ct_uint32_t **grInputGOSumH;

	ct_uint32_t **apGOGPU;
	ct_uint32_t **grInputGOGPU;
	size_t *grInputGOGPUP;
	ct_uint32_t **grInputGOSumGPU;
	//end gpu variables

	ct_uint32_t *sumGRInputGO;
	float *sumInputGOGABASynDepGO;
	float tempGIncGRtoGO;
	//---------end golgi cell variables

	//---------granule cell variables
	float **gMFGRT;
	float **gGOGRT;
	ct_uint32_t **pGRDelayfromGRtoGOT;
	ct_uint32_t **pGRfromMFtoGRT;
	ct_uint32_t **pGRfromGOtoGRT;
	ct_uint32_t **pGRfromGRtoGOT;

	ct_uint32_t apBufGRHistMask;
	//gpu related variables
	//host variables
	ct_uint8_t *outputGRH;
	//end host variables

	float **gEGRGPU;
	size_t *gEGRGPUP;
	float **gEGRSumGPU;

	float **gIGRGPU;
	size_t *gIGRGPUP;
	float **gIGRSumGPU;

	ct_uint32_t **apBufGRGPU;
	ct_uint8_t **outputGRGPU;
	ct_uint32_t **apGRGPU;

	float **threshGRGPU;
	float **vGRGPU;
	float **gKCaGRGPU;
	ct_uint64_t **historyGRGPU;

	//conduction delays
	ct_uint32_t **delayGOMasksGRGPU;
	size_t *delayGOMasksGRGPUP;
	ct_uint32_t **delayBCPCSCMaskGRGPU;

	//connectivity
	ct_int32_t **numGOOutPerGRGPU;
	ct_uint32_t **grConGROutGOGPU;
	size_t *grConGROutGOGPUP;

	ct_int32_t **numGOInPerGRGPU;
	ct_uint32_t **grConGOOutGRGPU;
	size_t *grConGOOutGRGPUP;

	ct_int32_t **numMFInPerGRGPU;
	ct_uint32_t **grConMFOutGRGPU;
	size_t *grConMFOutGRGPUP;
	//end gpu variables

	//---------end granule cell variables

	//--------stellate cell variables

	//gpu related variables
	//host variables
	ct_uint32_t *inputSumPFSCH;
	//end host variables

	ct_uint32_t **inputPFSCGPU;
	size_t *inputPFSCGPUP;
	ct_uint32_t **inputSumPFSCGPU;
	//end gpu related variables

	//------------ end stellate cell variables

	//-----------basket cell variables
	//gpu related variables
	//host variables
	ct_uint32_t *inputSumPFBCH;

	//device variables
	ct_uint32_t **inputPFBCGPU;
	size_t *inputPFBCGPUP;
	ct_uint32_t **inputSumPFBCGPU;
	//end gpu related variables
	//-----------end basket cell variables

private:
	InNet();

	template<typename Type> cudaError_t getGRGPUData(Type **gpuData, Type *hostData);
//	template<typename Type> cudaError_t sendGRGPUData(Type **gpuData, Type *hostData);

};


#endif /* INNET_H_ */
