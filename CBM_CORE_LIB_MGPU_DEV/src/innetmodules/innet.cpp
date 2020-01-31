/*
 * innet.cpp
 *
 *  Created on: Jun 21, 2011
 *      Author: consciousness
 */

#include "../../CBMCoreInclude/innetmodules/innet.h"

using namespace std;
InNet::InNet(ConnectivityParams *conParams, ActivityParams *actParams,
		InNetConnectivityState *conState, InNetActivityState *actState,
		int gpuIndStart, int numGPUs)
{
	randGen=new CRandomSFMT0(time(NULL));

	cp=conParams;
	ap=actParams;
	cs=conState;
	as=actState;

	this->gpuIndStart=gpuIndStart;
	this->numGPUs=numGPUs;

	gGOGRT=allocate2DArray<float>(cp->maxnumpGRfromGOtoGR, cp->numGR);
	gMFGRT=allocate2DArray<float>(cp->maxnumpGRfromMFtoGR, cp->numGR);
	pGRDelayfromGRtoGOT=allocate2DArray<ct_uint32_t>(cp->maxnumpGRfromGRtoGO, cp->numGR);
	pGRfromMFtoGRT=allocate2DArray<ct_uint32_t>(cp->maxnumpGRfromMFtoGR, cp->numGR);
	pGRfromGOtoGRT=allocate2DArray<ct_uint32_t>(cp->maxnumpGRfromGOtoGR, cp->numGR);
	pGRfromGRtoGOT=allocate2DArray<ct_uint32_t>(cp->maxnumpGRfromGRtoGO, cp->numGR);

	apBufGRHistMask=(1<<(ap->tsPerHistBinGR))-1;
	cout<<"apBufGRHistMask "<<hex<<apBufGRHistMask<<dec<<endl;

	sumGRInputGO=new ct_uint32_t[cp->numGO];
	sumInputGOGABASynDepGO=new float[cp->numGO];

	tempGIncGRtoGO=actParams->gIncGRtoGO;

	initCUDA();
}

InNet::~InNet()
{
	delete randGen;

	//gpu related host variables
	cudaFreeHost(outputGRH);
	cudaFreeHost(inputSumPFBCH);
	cudaFreeHost(inputSumPFSCH);

	//gpu variables
	for(int i=0; i<numGPUs; i++)
	{
		cudaSetDevice(i+gpuIndStart);
		//mf variables
		cudaFreeHost(apMFH[i]);
		cudaFree(apMFGPU[i]);

		//GR variables
		cudaFree(outputGRGPU[i]);
		cudaFree(vGRGPU[i]);
		cudaFree(gKCaGRGPU[i]);
		cudaFree(gEGRGPU[i]);
		cudaFree(gEGRSumGPU[i]);
		cudaFree(gIGRGPU[i]);
		cudaFree(gIGRSumGPU[i]);
		cudaFree(apBufGRGPU[i]);
		cudaFree(apGRGPU[i]);
		cudaFree(threshGRGPU[i]);
		cudaFree(delayGOMasksGRGPU[i]);
		cudaFree(delayBCPCSCMaskGRGPU[i]);
		cudaFree(numGOOutPerGRGPU[i]);
		cudaFree(grConGROutGOGPU[i]);
		cudaFree(numGOInPerGRGPU[i]);
		cudaFree(grConGOOutGRGPU[i]);
		cudaFree(numMFInPerGRGPU[i]);
		cudaFree(grConMFOutGRGPU[i]);
		cudaFree(historyGRGPU[i]);

		//GO variables
		cudaFreeHost(apGOH[i]);
		cudaFree(apGOGPU[i]);
		cudaFree(grInputGOGPU[i]);
		cudaFree(grInputGOSumGPU[i]);
		cudaFreeHost(grInputGOSumH[i]);

		//BC variables
		cudaFree(inputPFBCGPU[i]);
		cudaFree(inputSumPFBCGPU[i]);

		//SC variables
		cudaFree(inputPFSCGPU[i]);
		cudaFree(inputSumPFSCGPU[i]);
		//end gpu variables

		cudaDeviceSynchronize();
	}

	//mf
	delete[] apMFH;
	delete[] apMFGPU;

	//gr
	delete2DArray<float>(gMFGRT);
	delete2DArray<float>(gGOGRT);
	delete2DArray<ct_uint32_t>(pGRDelayfromGRtoGOT);
	delete2DArray<ct_uint32_t>(pGRfromMFtoGRT);
	delete2DArray<ct_uint32_t>(pGRfromGOtoGRT);
	delete2DArray<ct_uint32_t>(pGRfromGRtoGOT);

	delete[] gEGRGPU;
	delete[] gEGRGPUP;
	delete[] gEGRSumGPU;

	delete[] gIGRGPU;
	delete[] gIGRGPUP;
	delete[] gIGRSumGPU;

	delete[] apBufGRGPU;
	delete[] outputGRGPU;
	delete[] apGRGPU;

	delete[] threshGRGPU;
	delete[] vGRGPU;
	delete[] gKCaGRGPU;
	delete[] historyGRGPU;

	delete[] delayGOMasksGRGPU;
	delete[] delayGOMasksGRGPUP;
	delete[] delayBCPCSCMaskGRGPU;

	delete[] numGOOutPerGRGPU;
	delete[] grConGROutGOGPU;
	delete[] grConGROutGOGPUP;

	delete[] numGOInPerGRGPU;
	delete[] grConGOOutGRGPU;
	delete[] grConGOOutGRGPUP;

	delete[] numMFInPerGRGPU;
	delete[] grConMFOutGRGPU;
	delete[] grConMFOutGRGPUP;


	delete[] grInputGOSumH;

	//go
	delete[] apGOH;
	delete[] apGOGPU;
	delete[] grInputGOGPU;
	delete[] grInputGOGPUP;
	delete[] grInputGOSumGPU;
	delete[] sumGRInputGO;
	delete[] sumInputGOGABASynDepGO;

	//bc
	delete[] inputPFBCGPU;
	delete[] inputPFBCGPUP;
	delete[] inputSumPFBCGPU;

	//sc
	delete[] inputPFSCGPU;
	delete[] inputPFSCGPUP;
	delete[] inputSumPFSCGPU;
}

void InNet::initCUDA()
{
	cudaError_t error;
	int maxNumGPUs;

	error = cudaGetDeviceCount(&maxNumGPUs);
	cerr<<"CUDA number of devices: "<<maxNumGPUs<<", "<<cudaGetErrorString(error)<<endl;
	cerr<<"number of devices used: "<<numGPUs<<" starting at GPU# "<<gpuIndStart<<endl;

	numGRPerGPU=cp->numGR/numGPUs;
	calcGRActNumGRPerB=512;
	calcGRActNumBlocks=numGRPerGPU/calcGRActNumGRPerB;

	updateGRGOOutNumGRPerR=512*(cp->numGO>512)+cp->numGO*(cp->numGO<=512);
	updateGRGOOutNumGRRows=numGRPerGPU/updateGRGOOutNumGRPerR;

	sumGRGOOutNumGOPerB=1024*(cp->numGO>1024)+cp->numGO*(cp->numGO<=1024);
	sumGRGOOutNumBlocks=cp->numGO/sumGRGOOutNumGOPerB;

	updateMFInGRNumGRPerB=1024*(cp->numMF>1024)+(cp->numMF<=1024)*cp->numMF;;
	updateMFInGRNumBlocks=numGRPerGPU/updateMFInGRNumGRPerB;

	updateGOInGRNumGRPerB=1024*(cp->numGO>=1024)+(cp->numGO<1024)*cp->numGO;
	updateGOInGRNumBlocks=numGRPerGPU/updateGOInGRNumGRPerB;

	updatePFBCSCNumGRPerB=512;
	updatePFBCSCNumBlocks=numGRPerGPU/updatePFBCSCNumGRPerB;

	updateGRHistNumGRPerB=1024;
	updateGRHistNumBlocks=numGRPerGPU/updateGRHistNumGRPerB;


	cerr<<"numGRPerGPU: "<<numGRPerGPU<<endl;
	cerr<<"calcGRActNumBlocks "<<calcGRActNumBlocks<<endl;

	cerr<<"updateGRGOOutNumGRPerR "<<updateGRGOOutNumGRPerR<<endl;
	cerr<<"updateGRGOOutNumGRRows "<<updateGRGOOutNumGRRows<<endl;

	cerr<<"sumGRGOOutNumGOPerB "<<sumGRGOOutNumGOPerB<<endl;
	cerr<<"numGRGOOutNumBlocks "<<sumGRGOOutNumBlocks<<endl;

	cerr<<"updateMFInGRNumGRPerB "<<updateMFInGRNumGRPerB<<endl;
	cerr<<"updateMFInGRNumBlocks "<<updateMFInGRNumBlocks<<endl;

	cerr<<"updateGOInGRNumGRPerB "<<updateGOInGRNumGRPerB<<endl;
	cerr<<"updateGOInGRNumBlocks "<<updateGOInGRNumBlocks<<endl;

	cerr<<"updateGRHistNumBlocks "<<updateGRHistNumBlocks<<endl;



//	cudaSetDevice(gpuIndStart);
//	error=cudaMalloc<float>(&testA, 1024*sizeof(float));
//	error=cudaMalloc<float>(&testB, 1024*sizeof(float));
//	error=cudaMalloc<float>(&testC, 1024*sizeof(float));

	cerr<<"input network cuda init..."<<endl;
	initMFCUDA();
	error=cudaGetLastError();
	cerr<<"CUDA MF init: "<<cudaGetErrorString(error)<<endl;
	initGRCUDA();
	error=cudaGetLastError();
	cerr<<"CUDA gr init: "<<cudaGetErrorString(error)<<endl;
	cerr<<"here"<<endl;
	initGOCUDA();
	error=cudaGetLastError();
	cerr<<"CUDA go init: "<<cudaGetErrorString(error)<<endl;
	initBCCUDA();
	error=cudaGetLastError();
	cerr<<"CUDA bc init: "<<cudaGetErrorString(error)<<endl;
	initSCCUDA();
	error=cudaGetLastError();
	cerr<<"CUDA sc init: "<<cudaGetErrorString(error)<<endl;

}
void InNet::initMFCUDA()
{
	cudaError_t error;

	apMFGPU=new ct_uint32_t*[numGPUs];
	apMFH=new ct_uint32_t*[numGPUs];
//	cudaSetDevice(gpuIndStart);
//	error=cudaHostAlloc((void **)&apMFH, cp->numMF*sizeof(unsigned int), cudaHostAllocPortable);
//	cerr<<"apMFH alloc: "<<cudaGetErrorString(error)<<endl;
//	cudaMallocHost((void **)&apMFH, cp->numMF*sizeof(unsigned int));

	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		cerr<<"setting device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaMalloc((void **)&apMFGPU[i], cp->numMF*sizeof(ct_uint32_t));
		cerr<<"Allocating apMFGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaMallocHost((void **)&apMFH[i], cp->numMF*sizeof(ct_uint32_t));
		cerr<<"Allocating apMFH for device "<<i<<": "<<cudaGetErrorString(error)<<endl;

		cudaDeviceSynchronize();

		cudaMemset(apMFGPU[i], 0, cp->numMF*sizeof(ct_uint32_t));

		for(int j=0; j<cp->numMF; j++)
		{
			apMFH[i][j]=0;
		}
	}
}
void InNet::initGRCUDA()
{
	cudaError_t error;

	gEGRGPU=new float*[numGPUs];
	gEGRGPUP=new size_t[numGPUs];
	gEGRSumGPU=new float*[numGPUs];

	gIGRGPU=new float*[numGPUs];
	gIGRGPUP=new size_t[numGPUs];
	gIGRSumGPU=new float*[numGPUs];

	apBufGRGPU=new ct_uint32_t*[numGPUs];
	outputGRGPU=new ct_uint8_t*[numGPUs];
	apGRGPU=new ct_uint32_t*[numGPUs];

	threshGRGPU=new float*[numGPUs];
	vGRGPU=new float*[numGPUs];
	gKCaGRGPU=new float*[numGPUs];
	historyGRGPU=new ct_uint64_t*[numGPUs];

	delayGOMasksGRGPU=new ct_uint32_t*[numGPUs];
	delayGOMasksGRGPUP=new size_t[numGPUs];
	delayBCPCSCMaskGRGPU=new ct_uint32_t*[numGPUs];

	numGOOutPerGRGPU=new ct_int32_t*[numGPUs];
	grConGROutGOGPU=new ct_uint32_t*[numGPUs];
	grConGROutGOGPUP=new size_t[numGPUs];

	numGOInPerGRGPU=new ct_int32_t*[numGPUs];
	grConGOOutGRGPU=new ct_uint32_t*[numGPUs];
	grConGOOutGRGPUP=new size_t[numGPUs];

	numMFInPerGRGPU=new ct_int32_t*[numGPUs];
	grConMFOutGRGPU=new ct_uint32_t*[numGPUs];
	grConMFOutGRGPUP=new size_t[numGPUs];


//	error=cudaSetDevice(gpuIndStart);
//	cerr<<"setting device 0: "<<cudaGetErrorString(error)<<endl;
//	error=cudaHostAlloc((void **)&outputGRH, cp->numGR*sizeof(ct_uint8_t), cudaHostAllocPortable);
//	cerr<<"allocating outputGRH: "<<cudaGetErrorString(error)<<endl;
	outputGRH=new ct_uint8_t[cp->numGR];

//	cudaMallocHost((void **)&outputGRH, numGR*sizeof(unsigned char))

	for(int i=0; i<cp->numGR; i++)
	{
		outputGRH[i]=0;
	}

	//allocate memory for GPU
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		cerr<<"setting device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaMalloc((void **)&outputGRGPU[i], numGRPerGPU*sizeof(ct_uint8_t));
		cerr<<"allocating outputGRGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaMalloc((void **)&apGRGPU[i], numGRPerGPU*sizeof(ct_uint32_t));
		cerr<<"allocating apGRGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
//		error=cudaMalloc((void **)&vGRGPU[i], numGRPerGPU*sizeof(float));
		error=cudaMalloc<float>(&(vGRGPU[i]), numGRPerGPU*sizeof(float));
		cerr<<"numGRPerGPU "<<numGRPerGPU<<endl;
		cerr<<"allocating vGRGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaMalloc((void **)&gKCaGRGPU[i], numGRPerGPU*sizeof(float));
		cerr<<"allocating gKCaGRGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;

		error=cudaMallocPitch((void **)&gEGRGPU[i], (size_t *)&gEGRGPUP[i],
				numGRPerGPU*sizeof(float), cp->maxnumpGRfromMFtoGR);
		cerr<<"gEGRGPUP: "<<gEGRGPUP[i]<<endl;
		error=cudaMalloc((void **)&gEGRSumGPU[i], numGRPerGPU*sizeof(float));
		cerr<<"allocating gEGRSumGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaMallocPitch((void **)&gIGRGPU[i], (size_t *)&gIGRGPUP[i],
				numGRPerGPU*sizeof(float), cp->maxnumpGRfromGOtoGR);
		cerr<<"gEGRGPUP: "<<gIGRGPUP[i]<<endl;
		error=cudaMalloc((void **)&gIGRSumGPU[i], numGRPerGPU*sizeof(float));
		cerr<<"allocating gIGRSumGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaMalloc((void **)&apBufGRGPU[i], numGRPerGPU*sizeof(ct_uint32_t));
		error=cudaMalloc((void **)&threshGRGPU[i], numGRPerGPU*sizeof(float));
		cerr<<"allocating threshGRGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		//variables for conduction delays
		error=cudaMalloc((void **)&delayBCPCSCMaskGRGPU[i], numGRPerGPU*sizeof(ct_uint32_t));
		error=cudaMallocPitch((void **)&delayGOMasksGRGPU[i], (size_t *)&delayGOMasksGRGPUP[i],
				numGRPerGPU*sizeof(ct_uint32_t), cp->maxnumpGRfromGRtoGO);
		//end conduction delay

		//connectivity
		error=cudaMallocPitch((void **)&grConGROutGOGPU[i], (size_t *)&grConGROutGOGPUP[i],
					numGRPerGPU*sizeof(ct_uint32_t), cp->maxnumpGRfromGRtoGO);
		error=cudaMalloc((void **)&numGOOutPerGRGPU[i], numGRPerGPU*sizeof(ct_int32_t));

		error=cudaMallocPitch((void **)&grConGOOutGRGPU[i], (size_t *)&grConGOOutGRGPUP[i],
					numGRPerGPU*sizeof(ct_uint32_t), cp->maxnumpGRfromGOtoGR);
		error=cudaMalloc((void **)&numGOInPerGRGPU[i], numGRPerGPU*sizeof(ct_int32_t));

		error=cudaMallocPitch((void **)&grConMFOutGRGPU[i], (size_t *)&grConMFOutGRGPUP[i],
					numGRPerGPU*sizeof(ct_uint32_t), cp->maxnumpGRfromMFtoGR);
		error=cudaMalloc((void **)&numMFInPerGRGPU[i], numGRPerGPU*sizeof(ct_int32_t));
		//end connectivity

		error=cudaMalloc((void **)&historyGRGPU[i], numGRPerGPU*sizeof(ct_uint64_t));
		//end GPU memory allocation
		cudaDeviceSynchronize();

		cerr<<"GR GPU memory allocation: "<<cudaGetErrorString(error)<<endl;
	}

//	create a transposed copy of the matrices from activity state and connectivity
	for(int i=0; i<cp->maxnumpGRfromGOtoGR; i++)
	{
		for(int j=0; j<cp->numGR; j++)
		{
			gGOGRT[i][j]=as->gGOGR[j][i];
			pGRfromGOtoGRT[i][j]=cs->pGRfromGOtoGR[j][i];
		}
	}
	for(int i=0; i<cp->maxnumpGRfromMFtoGR; i++)
	{
		for(int j=0; j<cp->numGR; j++)
		{
			gMFGRT[i][j]=as->gMFGR[j][i];
			pGRfromMFtoGRT[i][j]=cs->pGRfromMFtoGR[j][i];
		}
	}
	for(int i=0; i<cp->maxnumpGRfromGRtoGO; i++)
	{
		for(int j=0; j<cp->numGR; j++)
		{
			pGRDelayfromGRtoGOT[i][j]=cs->pGRDelayMaskfromGRtoGO[j][i];
			pGRfromGRtoGOT[i][j]=cs->pGRfromGRtoGO[j][i];
		}
	}

	//initialize GR GPU variables
	cerr<<"start GPU memory initialization"<<endl;
	for(int i=0; i<numGPUs; i++)
	{
		int cpyStartInd;
		int cpySize;

		cpyStartInd=numGRPerGPU*i;//cp->numGR*i/numGPUs;
		cpySize=numGRPerGPU;
		cudaSetDevice(i+gpuIndStart);

		error=cudaMemcpy(vGRGPU[i], &(as->vGR[cpyStartInd]),
				cpySize*sizeof(float), cudaMemcpyHostToDevice);
		error=cudaMemcpy(gKCaGRGPU[i], &(as->gKCaGR[cpyStartInd]),
				cpySize*sizeof(float), cudaMemcpyHostToDevice);

//		error=cudaMemcpy(outputGRGPU[i], &outputGRH[cpyStartInd],
//				cpySize*sizeof(ct_uint8_t), cudaMemcpyHostToDevice);

		cerr<<"cuda memory copy vGRGPU, outputGRGPU, and gKCAGRGPU: "<<cudaGetErrorString(error)<<endl;

		for(int j=0; j<cp->maxnumpGRfromMFtoGR; j++)
		{
			error=cudaMemcpy((void *)((char *)gEGRGPU[i]+j*gEGRGPUP[i]),
					&gMFGRT[j][cpyStartInd], cpySize*sizeof(float), cudaMemcpyHostToDevice);
			error=cudaMemcpy((void *)((char *)grConMFOutGRGPU[i]+j*grConMFOutGRGPUP[i]),
					&pGRfromMFtoGRT[j][cpyStartInd], cpySize*sizeof(ct_uint32_t), cudaMemcpyHostToDevice);
		}
		cerr<<"cuda memory copy gEGRGPU and grConMFOutGRGPU: "<<cudaGetErrorString(error)<<endl;

		error=cudaMemcpy(gEGRSumGPU[i], &(as->gMFSumGR[cpyStartInd]), cpySize*sizeof(float), cudaMemcpyHostToDevice);

		for(int j=0; j<cp->maxnumpGRfromGOtoGR; j++)
		{
			error=cudaMemcpy((void *)((char *)gIGRGPU[i]+j*gIGRGPUP[i]),
					&gGOGRT[j][cpyStartInd], cpySize*sizeof(float), cudaMemcpyHostToDevice);
			error=cudaMemcpy((void *)((char *)grConGOOutGRGPU[i]+j*grConGOOutGRGPUP[i]),
					&pGRfromGOtoGRT[j][cpyStartInd], cpySize*sizeof(ct_uint32_t), cudaMemcpyHostToDevice);
		}
		error=cudaMemcpy(gIGRSumGPU[i], &(as->gGOSumGR[cpyStartInd]),
				cpySize*sizeof(float), cudaMemcpyHostToDevice);

		error=cudaMemcpy(apBufGRGPU[i], &(as->apBufGR[cpyStartInd]),
				cpySize*sizeof(ct_uint32_t), cudaMemcpyHostToDevice);

		error=cudaMemcpy(threshGRGPU[i], &(as->threshGR[cpyStartInd]),
				cpySize*sizeof(float), cudaMemcpyHostToDevice);

		for(int j=0; j<cp->maxnumpGRfromGRtoGO; j++)
		{
			error=cudaMemcpy((void *)((char *)delayGOMasksGRGPU[i]+j*delayGOMasksGRGPUP[i]),
					&pGRDelayfromGRtoGOT[j][cpyStartInd], cpySize*sizeof(float), cudaMemcpyHostToDevice );
			error=cudaMemcpy((void *)((char *)grConGROutGOGPU[i]+j*grConGROutGOGPUP[i]),
					&pGRfromGRtoGOT[j][cpyStartInd], cpySize*sizeof(unsigned int), cudaMemcpyHostToDevice);
		}

		error=cudaMemcpy(delayBCPCSCMaskGRGPU[i], &(cs->pGRDelayMaskfromGRtoBSP[cpyStartInd]),
				cpySize*sizeof(ct_uint32_t), cudaMemcpyHostToDevice);

		error=cudaMemcpy(numGOOutPerGRGPU[i], &(cs->numpGRfromGRtoGO[cpyStartInd]),
				cpySize*sizeof(ct_int32_t), cudaMemcpyHostToDevice);

		error=cudaMemcpy(numGOInPerGRGPU[i], &(cs->numpGRfromGOtoGR[cpyStartInd]),
				cpySize*sizeof(ct_int32_t), cudaMemcpyHostToDevice);

		error=cudaMemcpy(numMFInPerGRGPU[i], &(cs->numpGRfromMFtoGR[cpyStartInd]),
				cpySize*sizeof(ct_int32_t), cudaMemcpyHostToDevice);

		error=cudaMemcpy(historyGRGPU[i], &(as->historyGR[cpyStartInd]),
				cpySize*sizeof(ct_uint64_t), cudaMemcpyHostToDevice);

		cudaMemset(outputGRGPU[i], 0, cpySize*sizeof(ct_uint8_t));
		cudaMemset(apGRGPU[i], 0, cpySize*sizeof(ct_uint32_t));

		cudaDeviceSynchronize();
	}
	//end copying to GPU
	cerr<<"numGRPerGPU "<<numGRPerGPU<<endl;
}
void InNet::initGOCUDA()
{
	grInputGOSumH=new ct_uint32_t*[numGPUs];
	apGOH=new ct_uint32_t*[numGPUs];
	apGOGPU=new ct_uint32_t*[numGPUs];
	grInputGOGPU=new ct_uint32_t*[numGPUs];
	grInputGOGPUP=new size_t[numGPUs];
	grInputGOSumGPU=new ct_uint32_t*[numGPUs];

//	cudaSetDevice(gpuIndStart);
//	cudaHostAlloc((void **)&apGOH, cp->numGO*sizeof(unsigned int), cudaHostAllocPortable);
//	cudaMallocHost((void **)&apGOH, numGO*sizeof(unsigned int));

	//initialize host memory
	for(int i=0; i<numGPUs; i++)
	{
		cudaSetDevice(i+gpuIndStart);
		cudaMallocHost((void **)&grInputGOSumH[i], cp->numGO*sizeof(ct_uint32_t));
		cudaMallocHost((void **)&apGOH[i], cp->numGO*sizeof(ct_uint32_t));
		for(int j=0; j<cp->numGO; j++)
		{
			grInputGOSumH[i][j]=0;
			apGOH[i][j]=0;
		}
		//allocate gpu memory
		cudaMalloc((void **)&apGOGPU[i], cp->numGO*sizeof(ct_uint32_t));

		cudaMallocPitch((void **)&grInputGOGPU[i], (size_t *)&grInputGOGPUP[i],
				cp->numGO*sizeof(ct_uint32_t), updateGRGOOutNumGRRows);
		cudaMalloc((void **)&grInputGOSumGPU[i], cp->numGO*sizeof(ct_uint32_t));

		cudaDeviceSynchronize();

		for(int j=0; j<updateGRGOOutNumGRRows; j++)
		{
			cudaMemset(((char *)grInputGOGPU[i]+j*grInputGOGPUP[i]),
					0, cp->numGO*sizeof(ct_uint32_t));
		}

		cudaMemset(apGOGPU[i], 0, cp->numGO*sizeof(ct_uint32_t));
		cudaMemset(grInputGOSumGPU[i], 0, cp->numGO*sizeof(ct_uint32_t));
		cudaDeviceSynchronize();
	}
}
void InNet::initBCCUDA()
{
	inputPFBCGPU=new ct_uint32_t*[numGPUs];
	inputPFBCGPUP=new size_t[numGPUs];
	inputSumPFBCGPU=new ct_uint32_t*[numGPUs];

	//allocate host memory
	cudaSetDevice(gpuIndStart);
	cudaHostAlloc((void **)&inputSumPFBCH, cp->numBC*sizeof(ct_uint32_t), cudaHostAllocPortable);

	cudaDeviceSynchronize();
//	cudaMallocHost((void **)&inputSumPFBCH, numBC*sizeof(unsigned int));

	//initialize host variables
	for(int i=0; i<cp->numBC; i++)
	{
		inputSumPFBCH[i]=0;
	}

	for(int i=0; i<numGPUs; i++)
	{
		cudaSetDevice(i+gpuIndStart);
		//allocate GPU memory
		cudaMallocPitch((void **)&inputPFBCGPU[i], (size_t *)&inputPFBCGPUP[i],
				cp->numpBCfromGRtoBC*sizeof(ct_uint32_t), cp->numBC/numGPUs);
		cudaMalloc((void **)&inputSumPFBCGPU[i], cp->numBC/numGPUs*sizeof(ct_uint32_t));
		//end GPU allocation

		for(int j=0; j<cp->numBC/numGPUs; j++)
		{
			cudaMemset(((char *)inputPFBCGPU[i]+j*inputPFBCGPUP[i]), 0,
					cp->numpBCfromGRtoBC*sizeof(ct_uint32_t));
		}

		cudaMemset(inputSumPFBCGPU[i], 0, cp->numBC/numGPUs*sizeof(ct_uint32_t));

		cudaDeviceSynchronize();
	}
}
void InNet::initSCCUDA()
{
	inputPFSCGPU=new ct_uint32_t*[numGPUs];
	inputPFSCGPUP=new size_t[numGPUs];
	inputSumPFSCGPU=new ct_uint32_t*[numGPUs];

	//allocate host memory
	cudaSetDevice(gpuIndStart);
	cudaHostAlloc((void **)&inputSumPFSCH, cp->numSC*sizeof(ct_uint32_t), cudaHostAllocPortable);

	cudaDeviceSynchronize();
//	cudaMallocHost((void **)&inputSumPFSCH, numSC*sizeof(ct_uint32_t));
	//initialize host variables
	for(int i=0; i<cp->numSC; i++)
	{
		inputSumPFSCH[i]=0;
	}

	for(int i=0; i<numGPUs; i++)
	{
		//allocate GPU memory
		cudaSetDevice(i+gpuIndStart);
		cudaMallocPitch((void **)&inputPFSCGPU[i], (size_t *)&inputPFSCGPUP[i],
				cp->numpSCfromGRtoSC*sizeof(ct_uint32_t), cp->numSC/numGPUs);

		cudaMalloc((void **)&inputSumPFSCGPU[i], cp->numSC/numGPUs*sizeof(ct_uint32_t));
		//end GPU allocation

		for(int j=0; j<cp->numSC/numGPUs; j++)
		{
			cudaMemset(((char *)inputPFSCGPU[i]+j*inputPFSCGPUP[i]), 0,
					cp->numpSCfromGRtoSC*sizeof(ct_uint32_t));
		}

		cudaMemset(inputSumPFSCGPU[i], 0, cp->numSC/numGPUs*sizeof(ct_uint32_t));

		cudaDeviceSynchronize();
	}
}

void InNet::writeToState()
{
	cudaError_t error;
	//GR variables
	getGRGPUData<ct_uint8_t>(outputGRGPU, as->apGR);
	getGRGPUData<ct_uint32_t>(apBufGRGPU, as->apBufGR);
	getGRGPUData<float>(gEGRSumGPU, as->gMFSumGR);
	getGRGPUData<float>(gIGRSumGPU, as->gGOSumGR);

	getGRGPUData<float>(threshGRGPU, as->threshGR);
	getGRGPUData<float>(vGRGPU, as->vGR);
	getGRGPUData<float>(gKCaGRGPU, as->gKCaGR);
	getGRGPUData<ct_uint64_t>(historyGRGPU, as->historyGR);
	for(int i=0; i<numGPUs; i++)
	{
		int cpyStartInd;
		int cpySize;

		cpyStartInd=numGRPerGPU*i;
		cpySize=numGRPerGPU;

		cudaSetDevice(i+gpuIndStart);



		for(int j=0; j<cp->maxnumpGRfromMFtoGR; j++)
		{
			error=cudaMemcpy(&gMFGRT[j][cpyStartInd], (void *)((char *)gEGRGPU[i]+j*gEGRGPUP[i]),
					cpySize*sizeof(float), cudaMemcpyDeviceToHost);
		}

		for(int j=0; j<cp->maxnumpGRfromGOtoGR; j++)
		{
			error=cudaMemcpy(&gGOGRT[j][cpyStartInd], (void *)((char *)gIGRGPU[i]+j*gIGRGPUP[i]),
					cpySize*sizeof(float), cudaMemcpyDeviceToHost);
		}
	}

	for(int i=0; i<cp->maxnumpGRfromMFtoGR; i++)
	{
		for(int j=0; j<cp->numGR; j++)
		{
			as->gMFGR[j][i]=gMFGRT[i][j];
		}
	}

	for(int i=0; i<cp->maxnumpGRfromGOtoGR; i++)
	{
		for(int j=0; j<cp->numGR; j++)
		{
			as->gGOGR[j][i]=gGOGRT[i][j];
		}
	}

	//stellate variables
	for(int i=0; i<cp->numSC; i++)
	{
		as->inputSumPFSC[i]=inputSumPFSCH[i];
	}
}

void InNet::setGIncGRtoGO(float inc)
{
	tempGIncGRtoGO=inc;
}

void InNet::resetGIncGRtoGO()
{
	tempGIncGRtoGO=ap->gIncGRtoGO;
}

const ct_uint8_t* InNet::exportAPSC()
{
	return (const ct_uint8_t *)as->apSC;
}
const ct_uint8_t* InNet::exportAPGO()
{
	return (const ct_uint8_t *)as->apGO;
}
const ct_uint8_t* InNet::exportAPGR()
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		cudaSetDevice(i+gpuIndStart);
		error=cudaMemcpy((void *)&outputGRH[i*numGRPerGPU], outputGRGPU[i],
				numGRPerGPU*sizeof(ct_uint8_t), cudaMemcpyDeviceToHost);
#ifdef DEBUGOUT
		cerr<<"exportAPGR cuda memcpy: "<<cudaGetErrorString(error)<<endl;
#endif
	}

	return (const ct_uint8_t *)outputGRH;
}

template<typename Type>cudaError_t InNet::getGRGPUData(Type **gpuData, Type *hostData)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		cudaSetDevice(i+gpuIndStart);
		cudaMemcpy((void *)&hostData[i*numGRPerGPU], gpuData[i],
				numGRPerGPU*sizeof(Type), cudaMemcpyDeviceToHost);
	}
	return cudaGetLastError();
}

const float* InNet::exportVmGR()
{
	getGRGPUData<float>(vGRGPU, as->vGR);
	return (const float *)as->vGR;
}

const float* InNet::exportVmGO()
{
	return (const float *)as->vGO;
}
const float* InNet::exportVmSC()
{
	return (const float *)as->vSC;
}

const float* InNet::exportGESumGR()
{
	getGRGPUData<float>(gEGRSumGPU, as->gMFSumGR);
	return (const float *)as->gMFSumGR;
}
const float* InNet::exportGISumGR()
{
	getGRGPUData<float>(gIGRSumGPU, as->gGOSumGR);
	return (const float *)as->gGOSumGR;
}

const ct_uint32_t* InNet::exportSumGRInputGO()
{
	return (const ct_uint32_t *)sumGRInputGO;
}

const float* InNet::exportSumGOInputGO()
{
	return (const float *)sumInputGOGABASynDepGO;
}

const float* InNet::exportGOOutSynScaleGOGO()
{
	return (const float *)as->goGABAOutSynScaleGOGO;
}

const ct_uint8_t* InNet::exportHistMF()
{
	return (const ct_uint8_t *)as->histMF;
}
const ct_uint32_t* InNet::exportAPBufMF()
{
	return (const ct_uint32_t *)as->apBufMF;
}
const ct_uint32_t* InNet::exportAPBufGO()
{
	return (const ct_uint32_t *)as->apBufGO;
}
const ct_uint32_t* InNet::exportAPBufGR()
{
	getGRGPUData<ct_uint32_t>(apBufGRGPU, as->apBufGR);
	return (const ct_uint32_t *)as->apBufGR;
}
const unsigned int* InNet::exportAPBufSC()
{
	return (const unsigned int *)as->apBufSC;
}

const ct_uint32_t* InNet::exportPFBCSum()
{
	return (const ct_uint32_t *) inputSumPFBCH;
}
ct_uint32_t** InNet::getApBufGRGPUPointer()
{
	return apBufGRGPU;
}
ct_uint32_t** InNet::getDelayBCPCSCMaskGPUPointer()
{
	return delayBCPCSCMaskGRGPU;
}
ct_uint64_t** InNet::getHistGRGPUPointer()
{
	return historyGRGPU;
}

ct_uint32_t** InNet::getGRInputGOSumHPointer()
{
	return grInputGOSumH;
}

void InNet::updateMFActivties(const ct_uint8_t *actInMF)
{
#pragma omp parallel
	{
#pragma omp for
		for(int i=0; i<cp->numMF; i++)
		{
			as->histMF[i]=as->histMF[i] || (actInMF[i]>0);
			for(int j=0; j<numGPUs; j++)
			{
				apMFH[j][i]=(actInMF[i]>0);
			}
			as->apBufMF[i]=(as->apBufMF[i]<<1)|((actInMF[i]>0)*0x00000001);
		}
	}
}
void InNet::calcGOActivities()
{
#ifdef DEBUGOUT
	cout<<"inputMFGO"<<endl;
	for(int i=0; i<10; i++)
	{
		cout<<as->inputMFGO[i]<<" ";
	}
	cout<<endl;
#endif

#pragma omp parallel
	{
#pragma omp for
		for(int i=0; i<cp->numGO; i++)
		{
			unsigned int totalGRIn;

			totalGRIn=0;
			for(int j=0; j<numGPUs; j++)
			{
				totalGRIn+=grInputGOSumH[j][i];
			}

			sumGRInputGO[i]=totalGRIn;
			sumInputGOGABASynDepGO[i]=as->inputGOGABASynDepGO[i];

			as->gMFGO[i]=as->inputMFGO[i]*ap->gIncMFtoGO+as->gMFGO[i]*ap->gDecMFtoGO;

			as->gGRGO[i]=totalGRIn*tempGIncGRtoGO+as->gGRGO[i]*ap->gDecGRtoGO;
//			as->gGOGO[i]=as->inputGOGO[i]*ap->gGABAIncGOtoGO+as->gGOGO[i]*ap->gGABADecGOtoGO;
			//todo: test synaptic depression
			as->gGOGO[i]=as->inputGOGABASynDepGO[i]*ap->gGABAIncGOtoGO+as->gGOGO[i]*ap->gGABADecGOtoGO;

			as->gluGO[i]=as->gluGO[i]*ap->gluDecayGO+totalGRIn*ap->gluScaleGO*exp(-1.5*as->gluGO[i]);
			as->mGluRGO[i]=as->mGluRGO[i]*ap->mGluRDecayGO+as->gluGO[i]*ap->mGluRScaleGO*exp(-as->mGluRGO[i]);
			as->gMGluRIncGO[i]=as->gMGluRIncGO[i]*ap->gMGluRIncDecayGO+as->mGluRGO[i]*ap->gMGluRIncScaleGO*exp(-as->gMGluRIncGO[i]);
			as->gMGluRGO[i]=as->gMGluRGO[i]*ap->gMGluRDecGRtoGO+as->gMGluRIncGO[i]*ap->gMGluRScaleGRtoGO;
			as->threshCurGO[i]=as->threshCurGO[i]+(ap->threshRestGO-as->threshCurGO[i])*ap->threshDecGO;
			as->vGO[i]=as->vGO[i]+(ap->gLeakGO*(ap->eLeakGO-as->vGO[i]))
					+(as->gGOGO[i]*(ap->eGABAGO-as->vGO[i]))
					-(as->gMFGO[i]+as->gGRGO[i])*as->vGO[i]
					+as->vCoupleGO[i];
//			+(as->gMGluRGO[i]*(ap->eMGluRGO-as->vGO[i]))


			as->apGO[i]=as->vGO[i]>as->threshCurGO[i];
			for(int j=0; j<numGPUs; j++)
			{
				apGOH[j][i]=as->apGO[i];
			}
			as->apBufGO[i]=(as->apBufGO[i]<<1)|(as->apGO[i]*0x00000001);

			as->threshCurGO[i]=as->apGO[i]*ap->threshMaxGO+(!as->apGO[i])*as->threshCurGO[i];

			as->inputMFGO[i]=0;
			as->inputGOGO[i]=0;

			as->inputGOGABASynDepGO[i]=0;
		}
	}
#ifdef DEBUGOUT
	cout<<"inputMFGO after"<<endl;
	for(int i=0; i<10; i++)
	{
		cout<<as->inputMFGO[i]<<" ";
	}
	cout<<endl;
	cout<<"gGRGO: ";
	for(int i=0; i<5; i++)
	{
		cout<<as->gGRGO[i]<<" ";
	}
	cout<<endl;
#endif
}
void InNet::calcSCActivities()
{
#pragma omp parallel
	{
#pragma omp for
		for(int i=0; i<cp->numSC; i++)
		{
			as->gPFSC[i]=as->gPFSC[i]+(inputSumPFSCH[i]*ap->gIncGRtoSC);
			as->gPFSC[i]=as->gPFSC[i]*ap->gDecGRtoSC;

			as->vSC[i]=as->vSC[i]+(ap->gLeakSC*(ap->eLeakSC-as->vSC[i]))-as->gPFSC[i]*as->vSC[i];

			as->apSC[i]=as->vSC[i]>as->threshSC[i];
			as->apBufSC[i]=(as->apBufSC[i]<<1)|(as->apSC[i]*0x00000001);

			as->threshSC[i]=as->threshSC[i]+ap->threshDecSC*(ap->threshRestSC-as->threshSC[i]);
			as->threshSC[i]=as->apSC[i]*ap->threshMaxSC+(!as->apSC[i])*(as->threshSC[i]);
		}
	}
}
void InNet::updateMFtoGOOut()
{
#pragma omp parallel
	{
#pragma omp for
		for(int i=0; i<cp->numMF; i++)
		{
			if(apMFH[0][i])
			{
#pragma omp critical
				{
					for(int j=0; j<cs->numpMFfromMFtoGO[i]; j++)
					{
						as->inputMFGO[cs->pMFfromMFtoGO[i][j]]++;
					}
				}
			}
		}
	}
}
void InNet::updateGOtoGOOut()
{
#pragma omp parallel
	{
#pragma omp for
		for(int i=0; i<cp->numGO; i++)
		{
			if(as->apGO[i])
			{
#pragma omp critical
				{
					for(int j=0; j<cs->numpGOGABAOutGOGO[i]; j++)
					{
						as->inputGOGO[cs->pGOGABAOutGOGO[i][j]]++;

						as->inputGOGABASynDepGO[cs->pGOGABAOutGOGO[i][j]]+=as->goGABAOutSynScaleGOGO[i];
					}
				}
			}

			as->goGABAOutSynScaleGOGO[i]=as->goGABAOutSynScaleGOGO[i]+
					(!as->apGO[i])*(1-as->goGABAOutSynScaleGOGO[i])*ap->goGABAGOGOSynRec-
					as->apGO[i]*(as->goGABAOutSynScaleGOGO[i]*(1-ap->goGABAGOGOSynDepF));
		}

		//cout<<as->goGABAOutSynScaleGOGO[0]<<" "
		//		<<as->goGABAOutSynScaleGOGO[511]<<" "
		//		<<as->goGABAOutSynScaleGOGO[1023]<<" "
		//		<<ap->goGABAGOGOSynRec<<endl;
#pragma omp for
		for(int i=0; i<cp->numGO; i++)
		{
			float threshCoupleGO;

			as->vCoupleGO[i]=0;

			threshCoupleGO=0;
			for(int j=0; j<cs->numpGOCoupInGOGO[i]; j++)
			{
				as->vCoupleGO[i]=as->vCoupleGO[i]+ap->coupleRiRjRatioGO*
						(as->vGO[cs->pGOCoupInGOGO[i][j]]-as->vGO[i]);

				threshCoupleGO=threshCoupleGO+ap->coupleRiRjRatioGO*
						(as->threshCurGO[cs->pGOCoupInGOGO[i][j]]-as->threshCurGO[i]);
			}
			as->threshCurGO[i]=as->threshCurGO[i]+threshCoupleGO;
		}
	}
}

void InNet::resetMFHist(unsigned long t)
{
	if(t%ap->numTSinMFHist==0)
	{
#pragma omp parallel
		{
#pragma omp for
			for(int i=0; i<cp->numMF; i++)
			{
				as->histMF[i]=false;
			}
		}
	}
}
void InNet::runGRActivitiesCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
//		cerr<<"grActivityCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		callGRActKernel(sts[i][streamN], calcGRActNumBlocks, calcGRActNumGRPerB,
				vGRGPU[i], gKCaGRGPU[i], threshGRGPU[i],
				apBufGRGPU[i], outputGRGPU[i], apGRGPU[i],
				gEGRSumGPU[i], gIGRSumGPU[i],
				ap->gLeakGR, ap->eLeakGR, ap->eGOGR,
				ap->threshRestGR, ap->threshMaxGR, ap->threshDecGR);
#ifdef DEBUGOUT
		error=cudaGetLastError();
		cerr<<"grActivityCUDA: kernel launch for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}
void InNet::runSumPFBCCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
//		cerr<<"runSumPFBCCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		callSumKernel<ct_uint32_t, true, false>
		(sts[i][streamN], inputPFBCGPU[i], inputPFBCGPUP[i],
				inputSumPFBCGPU[i], 1, cp->numBC/numGPUs, 1, cp->numpBCfromGRtoBC);
#ifdef DEBUGOUT
		error=cudaGetLastError();
		cerr<<"runSumPFBCCUDA: kernel launch for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}
void InNet::runSumPFSCCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
//		cerr<<"runSumPFSCCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		callSumKernel<ct_uint32_t, true, false>
		(sts[i][streamN], inputPFSCGPU[i], inputPFSCGPUP[i],
				inputSumPFSCGPU[i], 1, cp->numSC/numGPUs, 1, cp->numpSCfromGRtoSC);
#ifdef DEBUGOUT
		error=cudaGetLastError();
		cerr<<"runSumPFBCCUDA: kernel launch for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}
void InNet::runSumGRGOOutCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
//		cerr<<"runSumGRGOOutCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		callSumGRGOOutKernel(sts[i][streamN], sumGRGOOutNumBlocks, sumGRGOOutNumGOPerB,
				updateGRGOOutNumGRRows, grInputGOGPU[i], grInputGOGPUP[i], grInputGOSumGPU[i]);
#ifdef DEBUGOUT
//		error=cudaGetLastError();
//		cerr<<"runSumGRGOOutCUDA: kernel launch for gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}
void InNet::cpyAPMFHosttoGPUCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
//		cerr<<"cpyAPMFHosttoGPUCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		error=cudaMemcpyAsync(apMFGPU[i], apMFH[i], cp->numMF*sizeof(ct_uint32_t), cudaMemcpyHostToDevice, sts[i][streamN]);
#ifdef DEBUGOUT
		cerr<<"cpyAPMFHosttoGPUCUDA: async copy for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}
void InNet::cpyAPGOHosttoGPUCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
//		cerr<<"cpyAPGOHosttoGPUCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		error=cudaMemcpyAsync(apGOGPU[i], apGOH[i], cp->numGO*sizeof(ct_uint32_t), cudaMemcpyHostToDevice, sts[i][streamN]);
#ifdef DEBUGOUT
		cerr<<"cpyAPGOHosttoGPUCUDA: async copy for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}
void InNet::runUpdateMFInGRCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
//		cerr<<"runUpdateMFInGRCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		callUpdateInGROPKernel(sts[i][streamN], updateMFInGRNumBlocks, updateMFInGRNumGRPerB,
				cp->numMF, apMFGPU[i], gEGRGPU[i], gEGRGPUP[i],
				grConMFOutGRGPU[i], grConMFOutGRGPUP[i],
				numMFInPerGRGPU[i], gEGRSumGPU[i], ap->gDecMFtoGR, ap->gIncMFtoGR);
#ifdef DEBUGOUT
		error=cudaGetLastError();
		cerr<<"runUpdateMFInGRCUDA: kernel launch for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}
void InNet::runUpdateGOInGRCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
//		cerr<<"runUpdateGOInGRCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		callUpdateInGROPKernel(sts[i][streamN], updateGOInGRNumBlocks, updateGOInGRNumGRPerB,
				cp->numGO, apGOGPU[i], gIGRGPU[i], gIGRGPUP[i],
				grConGOOutGRGPU[i], grConGOOutGRGPUP[i],
				numGOInPerGRGPU[i], gIGRSumGPU[i], ap->gDecGOtoGR, ap->gIncGOtoGR);
#ifdef DEBUGOUT
		error=cudaGetLastError();
		cerr<<"runUpdateGOInGRCUDA: kernel launch for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}
void InNet::runUpdatePFBCSCOutCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
//		cerr<<"runUpdatePFBCSCOutCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		callUpdatePFBCSCOutKernel(sts[i][streamN], updatePFBCSCNumBlocks, updatePFBCSCNumGRPerB,
				apBufGRGPU[i], delayBCPCSCMaskGRGPU[i],
				inputPFBCGPU[i], inputPFBCGPUP[i], cp->numpBCfromGRtoBCP2,
				inputPFSCGPU[i], inputPFSCGPUP[i], cp->numpSCfromGRtoSCP2);
#ifdef DEBUGOUT
		error=cudaGetLastError();
		cerr<<"runUpdatePFBCSCOutCUDA: kernel launch for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}
void InNet::runUpdateGROutGOCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
//		cerr<<"runUpdateGROutGOCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		callUpdateGROutGOKernel(sts[i][streamN], updateGRGOOutNumGRRows, updateGRGOOutNumGRPerR,
				cp->numGO, apBufGRGPU[i], grInputGOGPU[i], grInputGOGPUP[i],
				delayGOMasksGRGPU[i], delayGOMasksGRGPUP[i],
				grConGROutGOGPU[i], grConGROutGOGPUP[i], numGOOutPerGRGPU[i]);
#ifdef DEBUGOUT
		error=cudaGetLastError();
		cerr<<"runUpdateGROutGOCUDA: kernel launch for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}
void InNet::cpyPFBCSumGPUtoHostCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
//		cerr<<"cpyPFBCSumGPUtoHostCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		error=cudaMemcpyAsync(&inputSumPFBCH[cp->numBC*i/numGPUs], inputSumPFBCGPU[i],
				cp->numBC/numGPUs*sizeof(ct_uint32_t),
				cudaMemcpyDeviceToHost, sts[i][streamN]);
#ifdef DEBUGOUT
		cerr<<"cpyPFBCSumGPUtoHostCUDA: async copy for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}
void InNet::cpyPFSCSumGPUtoHostCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
//		cerr<<"cpyPFSCSumGPUtoHostCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		error=cudaMemcpyAsync(&inputSumPFSCH[cp->numSC*i/numGPUs], inputSumPFSCGPU[i],
				cp->numSC/numGPUs*sizeof(ct_uint32_t),
				cudaMemcpyDeviceToHost, sts[i][streamN]);
#ifdef DEBUGOUT
		cerr<<"cpyPFSCSumGPUtoHostCUDA: async copy for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}
void InNet::cpyGRGOSumGPUtoHostCUDA(cudaStream_t **sts, int streamN)
{
//	cudaError_t error;
//	for(int i=0; i<numGPUs; i++)
//	{
//		error=cudaSetDevice(i+gpuIndStart);
////		cerr<<"cpyGRGOSumGPUtoHostCUDA: switching to gpu #"<<i<<
////				": "<<cudaGetErrorString(error)<<endl;
//		error=cudaMemcpyAsync(grInputGOSumH[i], grInputGOSumGPU[i], cp->numGO*sizeof(ct_uint32_t),
//				cudaMemcpyDeviceToHost, sts[i][streamN]);
//#ifdef DEBUGOUT
////		cerr<<"cpyGRGOSumGPUtoHostCUDA: async copy for gpu #"<<i<<
////				": "<<cudaGetErrorString(error)<<endl;
//#endif
//	}

	cpyGRGOSumGPUtoHostCUDA(sts, streamN, grInputGOSumH);
}

void InNet::cpyGRGOSumGPUtoHostCUDA(cudaStream_t **sts, int streamN, ct_uint32_t **grInputGOSumHost)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);

		error=cudaMemcpyAsync(grInputGOSumHost[i], grInputGOSumGPU[i], cp->numGO*sizeof(ct_uint32_t),
				cudaMemcpyDeviceToHost, sts[i][streamN]);
	}
}

void InNet::runUpdateGRHistoryCUDA(cudaStream_t **sts, int streamN, unsigned long t)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
//		cerr<<"runUpdateGRHistoryCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		if(t%ap->tsPerHistBinGR==0)
		{
			callUpdateGRHistKernel(sts[i][streamN], updateGRHistNumBlocks, updateGRHistNumGRPerB,
					apBufGRGPU[i], historyGRGPU[i], apBufGRHistMask);
#ifdef DEBUGOUT
//			error=cudaGetLastError();
//			cerr<<"runUpdateGRHistoryCUDA: kernel launch for gpu #"<<i<<
//					": "<<cudaGetErrorString(error)<<endl;
#endif
		}
	}
}
