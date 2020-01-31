/*
 * kernels.cu
 *
 *  Created on: Jun 6, 2011
 *      Author: consciousness
 */

#include "../../CBMCoreInclude/cuda/kernels.h"
#include <iostream>

 extern __shared__ ct_uint32_t sharedIOBufGR[];

 extern __shared__ float sharedAbbott[];


__global__ void testKernel(float *a, float *b, float *c)
 {
 	int i=blockIdx.x*blockDim.x+threadIdx.x;

 	c[i]=a[i]+b[i];
 }


__global__ void updatePFPCSynAbbottIO(float *synWPFPC, float *testRand, ct_uint64_t *historyGR,
		ct_uint32_t *stateInd, ct_uint32_t *levelInd,
		float *qProb, float *pProb, float *synWTable,
		unsigned int numStates, unsigned int stateTableSize,
		ct_uint64_t plastCheckMask, int doLTP, unsigned int offset)
{
	int tid=threadIdx.x;
	int i=blockIdx.x*blockDim.x+tid+offset;

	int doP;
	int doQ;

	int currTableInd;

	int doPlast;

	float *abbottQ;
	float *abbottP;
	float *abbottW;

	abbottQ=&sharedAbbott[0];
	abbottP=&sharedAbbott[stateTableSize];
	abbottW=&sharedAbbott[2*stateTableSize];

	if(tid<stateTableSize)
	{
		abbottQ[tid]=qProb[tid];
		abbottP[tid]=pProb[tid];
		abbottW[tid]=synWTable[tid];
	}

	__syncthreads();

	currTableInd=numStates*levelInd[i]+stateInd[i];

	doPlast=historyGR[i]&plastCheckMask>0;

	doQ=doPlast*(testRand[i]<abbottQ[currTableInd])*
			((!levelInd[i] && doLTP)+(levelInd[i] && !doLTP));

	doP=doPlast*(testRand[i]<abbottP[currTableInd])*
			((!levelInd[i] && !doLTP)+(levelInd[i] && doLTP));

	stateInd[i]*=(!doQ);
	stateInd[i]+=doP;

	levelInd[i]+=doQ*((-1*((int)levelInd[i]))+(!levelInd[i]));

	currTableInd=numStates*levelInd[i]+stateInd[i];

	synWPFPC[i]=abbottW[currTableInd];
}


//**-----------------GR Kernels------------------**
__global__ void calcActivityGRGPU(float *vm, float *gKCa,
		float *thresh, ct_uint32_t *apBuf, ct_uint8_t *apOutGR, ct_uint32_t *apGR,
		float *gESum, float *gISum,
		float gLeak, float eLeak, float eGOIn,
		float threshBase, float threshMax, float threshDecay)
{
	float tempV;
	float tempGKCa;
	float tempThresh;
	unsigned int tempAP;

	int i=blockIdx.x*blockDim.x+threadIdx.x;

	tempV=vm[i];
	tempGKCa=gKCa[i];

	tempV=tempV+(gLeak+tempGKCa*tempGKCa*tempGKCa*tempGKCa)*(eLeak-tempV)
			-(gESum[i])*tempV+(gISum[i])*(eGOIn-tempV);

	tempThresh=thresh[i];
	tempThresh=tempThresh+(threshBase-tempThresh)*threshDecay;
	tempAP=tempV>tempThresh;
	thresh[i]=tempAP*threshMax+(!tempAP)*tempThresh;

	tempGKCa=tempGKCa*0.9999f;
	gKCa[i]=tempAP*(tempGKCa+0.003f)+(!tempAP)*tempGKCa;

	apBuf[i]=(apBuf[i]<<1)|(tempAP);
	apOutGR[i]=tempAP;
	apGR[i]=tempAP;
	vm[i]=tempV;
}

__global__ void updateGRGOOutGPU(ct_uint32_t *apBuf,
		ct_uint32_t *goOut, size_t goOutPitch,
		ct_uint32_t *delay, size_t delayPitch,
		ct_uint32_t *con, size_t conPitch,
		ct_int32_t *numSyn, int nWrites)
{
	int tid=threadIdx.x;
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int *conRow;
	unsigned int *delayRow;
	unsigned int *goRow=(unsigned int *)((char *)goOut+blockIdx.x*goOutPitch);

	int tempNS=numSyn[index];
	unsigned int tempOut;

	for(int i=0; i<nWrites; i++)
	{
		sharedIOBufGR[tid+i*blockDim.x]=0;
	}

	__syncthreads();
	for(int i=0; i<tempNS; i++)
	{
		conRow=(ct_uint32_t *)((char *)con+i*conPitch);
		delayRow=(ct_uint32_t *)((char *)delay+i*delayPitch);

		tempOut=(apBuf[index]&delayRow[index])>0;

		if(tempOut>0)
		{
			atomicAdd(&sharedIOBufGR[conRow[index]], 1);
		}
	}
	__syncthreads();
	for(int i=0; i<nWrites; i++)
	{
		goRow[tid+i*blockDim.x]=sharedIOBufGR[tid+i*blockDim.x];
	}
}

__global__ void sumGRGOOutGPU(unsigned int nRows, ct_uint32_t *goOut, size_t goOutPitch, ct_uint32_t *goOutSum)
{
	unsigned int *goOutRow;
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int tempSum;

	tempSum=0;
	for(int i=0; i<nRows; i++)
	{
		goOutRow=(unsigned int *)((char *)goOut+i*goOutPitch);

		tempSum+=goOutRow[index];
	}

	goOutSum[index]=tempSum;
}

__global__ void updateGRInOPGPU(unsigned int inNLoads, ct_uint32_t *apIn,
		float *g, size_t gPitch,
		ct_uint32_t *conFromIn, size_t conFromInPitch,
		ct_int32_t *numInPerGR, float *gSum, float gDecay, float gIncConst)
{
//	__shared__ unsigned int sAPIn[nCellIn+1];

	int tid=threadIdx.x;
	int index=blockIdx.x*blockDim.x+threadIdx.x;

	unsigned int *conRow;

	int tempNSyn=numInPerGR[index];

	int tempApInSum=0;

	for(int i=0; i<inNLoads; i++)
	{
		sharedIOBufGR[tid+i*blockDim.x]=apIn[tid+i*blockDim.x];
	}
	__syncthreads();

	for(int i=0; i<tempNSyn; i++)
	{
		conRow=(unsigned int *)((char *)conFromIn+i*conFromInPitch);
		tempApInSum+=sharedIOBufGR[conRow[index]];

	}

	gSum[index]=gSum[index]*gDecay+gIncConst*(tempApInSum);
}


//__global__ void updateGRInGPU(unsigned int *apIn,
//		float *g, unsigned int gPitch,
//		unsigned int *conFromIn, unsigned int conFromInPitch,
//		int *numInPerGR, float *gSum, float gDecay, float gIncConst)
//{
//	__shared__ unsigned int sAPIn[nCellIn+1];
//
//	int tid=threadIdx.x;
//	int index=blockIdx.x*blockDim.x+threadIdx.x;
//
//	unsigned int *conRow;
//	float *gRow;
//	int tempNSyn=numInPerGR[index];
//	float tempGSum=0;
//
//	sAPIn[tid]=apIn[tid];
//	__syncthreads();
//
//	for(int i=0; i<tempNSyn; i++)
//	{
//		float tempG;
//
//		gRow=(float *)((char *)g+i*gPitch);
//		conRow=(unsigned int *)((char *)conFromIn+i*conFromInPitch);
//		tempG=gRow[index]*gDecay+gIncConst*(sAPIn[conRow[index]]);
//
//		tempGSum+=tempG;
//		gRow[index]=tempG;
//	}
//	gSum[index]=tempGSum;
//}

//__global__ void updateGRSumInGPU(float *g, float gDecay, float gIncConst, float inSum)
//{
//	int index=blockIdx.x*blockDim.x+threadIdx.x;
//
//	g[index]=g[index]*gDecay+gIncConst*inSum;
//}

__global__ void updateGRHistory(ct_uint32_t *apBuf, ct_uint64_t *apHist, ct_uint32_t bufTestMask)
{
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	ct_uint64_t tempHist=apHist[i]<<1;
	apHist[i]=tempHist|((apBuf[i]&0x0000001F)>0)*0x00000001; //bufTestMask
}
__global__ void updatePFBCSCOutGPU(ct_uint32_t *apBuf, ct_uint32_t *delay,
		ct_uint32_t *pfBC, size_t pfBCPitch, unsigned int numPFInPerBC, unsigned int numPFInPerBCP2,
		ct_uint32_t *pfSC, size_t pfSCPitch, unsigned int numPFInPerSC, unsigned int numPFInPerSCP2)
{
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	ct_uint32_t tempOut;
	unsigned int *pfBCRow=(ct_uint32_t *)((char *)pfBC+(index>>numPFInPerBCP2)*pfBCPitch);
	unsigned int *pfSCRow=(ct_uint32_t *)((char *)pfSC+(index>>numPFInPerSCP2)*pfSCPitch);

	tempOut=(apBuf[index]&delay[index])>0;

	pfBCRow[index&(numPFInPerBC-1)]=tempOut;
	pfSCRow[index&(numPFInPerSC-1)]=tempOut;
}

__global__ void updatePFPCOutGPU(ct_uint32_t *apBuf, ct_uint32_t *delay,
		float *synWeight, float *pfPC, size_t pfPCPitch, unsigned int numPFInPerPC, unsigned int numPFInPerPCP2)
{
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int tempOut;
	float *pfPCRow=(float *)((char *)pfPC+(index>>numPFInPerPCP2)*pfPCPitch);

	tempOut=(apBuf[index]&delay[index])>0;

	pfPCRow[index&(numPFInPerPC-1)]=synWeight[index]*tempOut;
}

//__global__ void updateGROutForHGPU(unsigned char *output, unsigned int *apBuf)
//{
//	int index=blockIdx.x*blockDim.x+threadIdx.x;
//	output[index]=apBuf[index]&0x00000001;
//}
//**---------------end GR Kernels-------------------**

//**---------------IO kernels-----------------**
__global__ void updatePFPCSynIO(float *synWPFPC, ct_uint64_t *historyGR, ct_uint64_t plastCheckMask,
		unsigned int offset, float plastStep)
{
	int i=blockIdx.x*blockDim.x+threadIdx.x+offset;
//	synWPFPC[i]=synWPFPC[i]+checkBinGR[i]*(doLTD*PFPCLTDDECPF+(!doLTD)*PFPCLTPINCPF);
	synWPFPC[i]=synWPFPC[i]+((historyGR[i]&0x0000008000000000)>0)*plastStep;//plastCheckMask

	synWPFPC[i]=(synWPFPC[i]>0)*synWPFPC[i];
	synWPFPC[i]=(synWPFPC[i]>1)+(synWPFPC[i]<=1)*synWPFPC[i];
}

//**---------------end IO kernels-------------**

//**---------------common kernels-------------**
template <typename Type, unsigned int blockSize, unsigned int sDataSize, bool inMultiPitch, bool outMultiPitch>
__global__ void sumInputsNew(Type *input, unsigned int inputPitch,
		Type *output, unsigned int outputPitch, unsigned int rowLength)
{
		__shared__ Type sData[sDataSize];

	int tid=threadIdx.x;
	int index=blockIdx.x*(blockSize*2)+tid;
	int gridSize=blockSize*2*gridDim.x;
	Type *inputRow;

	Type tempSum=0;

	if(inMultiPitch)
	{
		inputRow=(Type *)((char *)input+blockIdx.y*inputPitch);
	}
	else
	{
		inputRow=input+blockIdx.y;
	}

	while(index<rowLength)
	{
		tempSum+=inputRow[index]+inputRow[index+blockSize];
		index+=gridSize;
	}
	sData[tid]=tempSum;
	__syncthreads();

	if(blockSize>=512)
	{
		if(tid<256)
			sData[tid]+=sData[tid+256];
		__syncthreads();
	}

	if(blockSize>=256)
	{
		if(tid<128)
			sData[tid]+=sData[tid+128];
		__syncthreads();
	}

	if(blockSize>=128)
	{
		if(tid<64)
			sData[tid]+=sData[tid+64];
		__syncthreads();
	}

	if(tid<32)
	{
		volatile Type* sMem = sData;
		if(blockSize>=64)
			sMem[tid]+=sMem[tid+32];
		if(blockSize>=32)
			sMem[tid]+=sMem[tid+16];
		if(blockSize>=16)
			sMem[tid]+=sMem[tid+8];
		if(blockSize>=8)
			sMem[tid]+=sMem[tid+4];
		if(blockSize>=4)
			sMem[tid]+=sMem[tid+2];
		if(blockSize>=2)
			sMem[tid]+=sMem[tid+1];
	}
	if(tid==0)
	{
		Type *outputRow;
		if(outMultiPitch)
		{
			outputRow=(Type *)((char *)output+blockIdx.y*outputPitch);
		}
		else
		{
			outputRow=output+blockIdx.y;
		}
		outputRow[blockIdx.x]=sData[0];
	}
}



template<typename Type>
__global__ void broadcastValue(Type *val, Type *outArr)
{
	int i=blockIdx.x*blockDim.x+threadIdx.x;

	outArr[i]=*val;
}
//**---------------end common kernels---------**

//**---------------analysis kernels-----------**

//__global__ void updatePSHGPU(unsigned int *apBuf, unsigned int *psh, unsigned int pshP,
//		int nBins, int tsPerBin, unsigned int extrashift)
//{
//	int index=blockIdx.x*blockDim.x+threadIdx.x;
//	unsigned int tempBuf=apBuf[index]<<extrashift;
//
//	for(int i=0; i<nBins; i++)
//	{
//		unsigned int *pshRow;
//		unsigned int tempCount;
//
//		pshRow=(unsigned int *)((char *)psh+i*pshP);
//		tempCount=0;
//		for(int j=0; j<tsPerBin; j++)
//		{
//			tempCount+=(tempBuf&0x80000000)>0;
//			tempBuf=tempBuf<<1;
//		}
//		pshRow[index]+=tempCount;
//	}
//}
//**---------------end analysis kernels--------**


//**---------------kernel calls---------------**

void callTestKernel(cudaStream_t &st, float *a, float *b, float *c)
{
	testKernel<<<1, 128>>>(a, b, c);
}

void callGRActKernel(cudaStream_t &st, unsigned int numBlocks, unsigned int numGRPerBlock,
		float *vGPU, float *gKCaGPU, float *threshGPU,
		ct_uint32_t *apBufGPU, ct_uint8_t *apOutGRGPU, ct_uint32_t *apGRGPU,
		float *gESumGPU, float *gISumGPU,
		float gLeak, float eLeak, float eGOIn,
		float threshBase, float threshMax, float threshDecay)
{
	calcActivityGRGPU<<<numBlocks, numGRPerBlock, 0, st>>>(vGPU, gKCaGPU, threshGPU,
			apBufGPU, apOutGRGPU, apGRGPU,
			gESumGPU, gISumGPU, gLeak, eLeak, eGOIn,
			threshBase, threshMax, threshDecay);
}

template<typename Type, bool inMultiP, bool outMultiP>
void callSumKernel(cudaStream_t &st, Type *inGPU, size_t inGPUP, Type *outSumGPU, size_t outSumGPUP,
		unsigned int nOutCells, unsigned int nOutCols, unsigned int rowLength)
{
	unsigned int numElementsPerBlock;
	dim3 dimGrid(nOutCols, nOutCells);

	numElementsPerBlock=rowLength/nOutCols;
//	std::cout<<numElementsPerBlock<<std::endl;

	if(numElementsPerBlock>=2048)
	{
		sumInputsNew<Type, 512, 512, inMultiP, outMultiP><<<dimGrid, 512, 0, st>>>
				(inGPU, inGPUP, outSumGPU, outSumGPUP, rowLength);
	}
	else if(numElementsPerBlock>=512)
	{
		sumInputsNew<Type, 128, 128, inMultiP, outMultiP><<<dimGrid, 128, 0, st>>>
				(inGPU, inGPUP, outSumGPU, outSumGPUP, rowLength);
	}
	else if(numElementsPerBlock>=128)
	{
		sumInputsNew<Type, 32, 64, inMultiP, outMultiP><<<dimGrid, 32, 0, st>>>
				(inGPU, inGPUP, outSumGPU, outSumGPUP, rowLength);
	}
	else if(numElementsPerBlock>=32)
	{
		sumInputsNew<Type, 8, 64, inMultiP, outMultiP><<<dimGrid, 8, 0, st>>>
				(inGPU, inGPUP, outSumGPU, outSumGPUP, rowLength);
	}
	else
	{
		sumInputsNew<Type, 2, 64, inMultiP, outMultiP><<<dimGrid, 2, 0, st>>>
				(inGPU, inGPUP, outSumGPU, outSumGPUP, rowLength);
	}
}

template<typename Type>
void callBroadcastKernel(cudaStream_t &st, Type *broadcastVal, Type *outArray,
		unsigned int nBlocks, unsigned int rowLength)
{
	broadcastValue<Type><<<nBlocks, rowLength/nBlocks, 0, st>>>(broadcastVal, outArray);
}

void callSumGRGOOutKernel(cudaStream_t &st, unsigned int numBlocks, unsigned int numGOPerBlock,
		unsigned int numGROutRows, ct_uint32_t *grInGOGPU,  size_t grInGOGPUPitch, ct_uint32_t *grInGOSGPU)
{
	sumGRGOOutGPU<<<numBlocks, numGOPerBlock, 0, st>>>(numGROutRows, grInGOGPU, grInGOGPUPitch, grInGOSGPU);
}

void callUpdateInGROPKernel(cudaStream_t &st, unsigned int numBlocks, unsigned int numGRPerBlock,
		unsigned int numInCells, ct_uint32_t *apInGPU, float *gGPU, size_t gGPUP,
		ct_uint32_t *conInGRGPU, size_t conInGRGPUP,
		ct_int32_t *numInPerGRGPU, float *gSumGPU, float gDecay, float gInc)
{
	updateGRInOPGPU<<<numBlocks, numGRPerBlock, numInCells*sizeof(ct_uint32_t), st>>>
			(numInCells/numGRPerBlock, apInGPU, gGPU, gGPUP, conInGRGPU, conInGRGPUP, numInPerGRGPU,
			gSumGPU, gDecay, gInc);
}


//void callUpdateInGRKernel(cudaStream_t &st, unsigned int nBlocks, unsigned int nThreadsPerB,
//		unsigned int *apInGPU, float *gGPU, unsigned int gGPUP,
//		unsigned int *conInGRGPU, unsigned int conInGRGPUP,
//		int *numInPerGRGPU, float *gSumGPU, float gDecay, float gInc)
//{
//	updateGRInGPU<<<nBlocks, nThreadsPerB, 0, st>>>(apInGPU, gGPU, gGPUP,
//			conInGRGPU, conInGRGPUP, numInPerGRGPU, gSumGPU, gDecay, gInc);
//}

//void callupdateInSumGRKernel(cudaStream_t &st, unsigned int nBlocks, unsigned int nThreadsPerB,
//		float *gGPU, float gDecay, float gInc, float inSum)
//{
//	updateGRSumInGPU<<<nBlocks, nThreadsPerB, 0, st>>>(gGPU, gDecay, gInc, inSum);
//}

void callUpdatePFBCSCOutKernel(cudaStream_t &st, unsigned int numBlocks, unsigned int numGRPerBlock,
		ct_uint32_t *apBufGPU, ct_uint32_t *delayMaskGPU,
		ct_uint32_t *inPFBCGPU, size_t inPFBCGPUPitch, unsigned int numPFInPerBCP2,
		ct_uint32_t *inPFSCGPU, size_t inPFSCGPUPitch, unsigned int numPFInPerSCP2)
{
	updatePFBCSCOutGPU<<<numBlocks, numGRPerBlock, 0, st>>>(apBufGPU, delayMaskGPU,
			inPFBCGPU, inPFBCGPUPitch, 1<<numPFInPerBCP2, numPFInPerBCP2,
			inPFSCGPU, inPFSCGPUPitch, 1<<numPFInPerSCP2, numPFInPerSCP2);
}

void callUpdatePFPCOutKernel(cudaStream_t &st, unsigned int numBlocks, unsigned int numGRPerBlock,
		ct_uint32_t *apBufGPU, ct_uint32_t *delayMaskGPU,
		float *pfPCSynWGPU, float *inPFPCGPU, size_t inPFPCGPUPitch, unsigned int numPFInPerPCP2)
{
	updatePFPCOutGPU<<<numBlocks, numGRPerBlock, 0, st>>>(apBufGPU, delayMaskGPU, pfPCSynWGPU,
			inPFPCGPU, inPFPCGPUPitch, 1<<numPFInPerPCP2, numPFInPerPCP2);
}

void callUpdateGROutGOKernel(cudaStream_t &st, unsigned int numBlocks, unsigned int numGRPerBlock, unsigned int numGO,
		ct_uint32_t *apBufGPU, ct_uint32_t *grInGOGPU, ct_uint32_t grInGOGPUPitch,
		ct_uint32_t *delayMasksGPU, ct_uint32_t delayMasksGPUPitch,
		ct_uint32_t *conGRtoGOGPU, size_t conGRtoGOGPUPitch,
		ct_int32_t *numGOPerGRGPU)
{
	updateGRGOOutGPU<<<numBlocks, numGRPerBlock, numGO*sizeof(ct_uint32_t), st>>>(apBufGPU, grInGOGPU, grInGOGPUPitch,
			delayMasksGPU, delayMasksGPUPitch, conGRtoGOGPU, conGRtoGOGPUPitch, numGOPerGRGPU, numGO/numGRPerBlock);
}
void callUpdateGRHistKernel(cudaStream_t &st, unsigned int numBlocks, unsigned int numGRPerBlock,
		ct_uint32_t *apBufGPU, ct_uint64_t *historyGPU, ct_uint32_t apBufGRHistMask)
{
		updateGRHistory<<<numBlocks, numGRPerBlock, 0, st>>>(apBufGPU, historyGPU, apBufGRHistMask);
}

void callUpdatePFPCPlasticityIOKernel(cudaStream_t &st, unsigned int numBlocks, unsigned int numGRPerBlock,
		float *synWeightGPU, ct_uint64_t *historyGPU, unsigned int pastBinNToCheck,
		int offSet, float pfPCPlastStep)
{
	ct_uint64_t mask;
	mask=1;
	mask=mask<<pastBinNToCheck;
		updatePFPCSynIO<<<numBlocks, numGRPerBlock, 0, st>>>(synWeightGPU, historyGPU,
				((ct_uint64_t)1)<<pastBinNToCheck, offSet, pfPCPlastStep);
}

//void callUpdatePSHGPU(unsigned int *apBufGPU, unsigned int *pshGPU, unsigned int pshGPUP,
//		int nBins, int tsPerBin, unsigned int extrashift, unsigned int nBlocks, unsigned int nThreadPerB)
//{
//	updatePSHGPU<<<nBlocks, nThreadPerB>>>(apBufGPU, pshGPU, pshGPUP, nBins, tsPerBin, extrashift);
//}

//**---------------end kernel calls------------**

////template initializations
//
template void callSumKernel<float, true, false>
		(cudaStream_t &st, float *inPFGPU, size_t inPFGPUP, float *outPFSumGPU, size_t outPFSumGPUP,
		unsigned int nOutCells, unsigned int nOutCols, unsigned int rowLength);
template void callSumKernel<ct_uint32_t, true, false>
		(cudaStream_t &st, ct_uint32_t *inPFGPU, size_t inPFGPUP, ct_uint32_t *outPFSumGPU, size_t outPFSumGPUP,
		unsigned int nOutCells, unsigned int nOutCols, unsigned int rowLength);
template void callSumKernel<ct_uint32_t, false, false>
		(cudaStream_t &st, ct_uint32_t *inPFGPU, size_t inPFGPUP, ct_uint32_t *outPFSumGPU, size_t outPFSumGPUP,
		unsigned int nOutCells, unsigned int nOutCols, unsigned int rowLength);


template void callBroadcastKernel<ct_uint32_t>
(cudaStream_t &st, ct_uint32_t *broadCastVal, ct_uint32_t *outArray, unsigned int nBlocks, unsigned int rowLength);
