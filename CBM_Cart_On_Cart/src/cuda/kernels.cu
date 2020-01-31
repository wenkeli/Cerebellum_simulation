/*
 * kernels.cu
 *
 *  Created on: Jun 6, 2011
 *      Author: consciousness
 */

#include "../../includes/cuda/kernels.h"
//**-----------------GR Kernels------------------**
__global__ void calcActivityGRGPU(float *vm, float *gKCa,
		float *thresh, unsigned int *apBuf, unsigned char *apOutGR,
		float *gESum, float *gISum,
		float gLeak, float eLeak, float eGOIn,
		float threshBase, float threshMax, float threshDecay)
{
	float tempV;
	float tempGKCa;
	float tempThresh;
	unsigned int tempAP;

	int i=blockIdx.x*blockDim.x+threadIdx.x;

	tempGKCa=gKCa[i];
	tempV=vm[i];
	tempV=tempV+(gLeak+tempGKCa*tempGKCa*tempGKCa*tempGKCa)*(eLeak-tempV);
	tempGKCa=tempGKCa*0.9999f;

	tempV=tempV-(gESum[i])*tempV+(gISum[i])*(eGOIn-tempV);
	tempThresh=thresh[i];
	tempThresh=tempThresh+(threshBase-tempThresh)*threshDecay;
	tempAP=tempV>tempThresh;
	thresh[i]=tempAP*threshMax+(!tempAP)*tempThresh;
	gKCa[i]=tempAP*(tempGKCa+0.003f)+(!tempAP)*tempGKCa;
	apBuf[i]=(apBuf[i]<<1)|(tempAP);
	apOutGR[i]=tempAP;
	vm[i]=tempV;
}

template <unsigned int nGO>
__global__ void updateGRGOOutGPU(unsigned int *apBuf,
		unsigned int *goOut, unsigned int goOutPitch,
		unsigned int *delay, unsigned int delayPitch,
		unsigned int *con, unsigned int conPitch,
		int *numSyn)
{
	__shared__ unsigned int sGO[nGO+1];
	int tid=threadIdx.x;
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int *conRow;
	unsigned int *delayRow;
	unsigned int *goRow=(unsigned int *)((char *)goOut+blockIdx.x*goOutPitch);

	int tempNS=numSyn[index];
	unsigned int tempOut;

	sGO[tid]=0;

	__syncthreads();
	for(int i=0; i<tempNS; i++)
	{
		conRow=(unsigned int *)((char *)con+i*conPitch);
		delayRow=(unsigned int *)((char *)delay+i*delayPitch);

		tempOut=(apBuf[index]&delayRow[index])>0;

		if(tempOut>0)
		{
			atomicAdd(&sGO[conRow[index]], 1);
		}
	}
	__syncthreads();
	goRow[tid]=sGO[tid];
}

template <unsigned int nCols>
__global__ void sumGRGOOutGPU(unsigned int *goOut, unsigned int goOutPitch, unsigned int *goOutSum)
{
	unsigned int *goOutCol;
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int tempSum;

	tempSum=0;
	for(int i=0; i<nCols; i++)
	{
		goOutCol=(unsigned int *)((char *)goOut+i*goOutPitch);

		tempSum+=goOutCol[index];
	}

	goOutSum[index]=tempSum;
}

template <unsigned int nCellIn>
__global__ void updateGRInGPU(unsigned int *apIn,
		float *g, unsigned int gPitch,
		unsigned int *conFromIn, unsigned int conFromInPitch,
		int *numInPerGR, float *gSum, float gDecay, float gIncConst)
{
	__shared__ unsigned int sAPIn[nCellIn+1];

	int tid=threadIdx.x;
	int index=blockIdx.x*blockDim.x+threadIdx.x;

	unsigned int *conRow;
	float *gRow;
	int tempNSyn=numInPerGR[index];
	float tempGSum=0;

	sAPIn[tid]=apIn[tid];
	__syncthreads();

	for(int i=0; i<tempNSyn; i++)
	{
		float tempG;

		gRow=(float *)((char *)g+i*gPitch);
		conRow=(unsigned int *)((char *)conFromIn+i*conFromInPitch);
		tempG=gRow[index]*gDecay+gIncConst*(sAPIn[conRow[index]]);

		tempGSum+=tempG;
		gRow[index]=tempG;
	}
	gSum[index]=tempGSum;
}

__global__ void updateGRHistory(unsigned int *apBuf, unsigned long *apHist)
{
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	unsigned long tempHist=apHist[i]<<1;
	apHist[i]=tempHist|((apBuf[i]&0x0000001F)>0)*0x00000001;
}

template<unsigned int numPFInPerBC, unsigned int numPFInPerSC>
__global__ void updatePFBCSCOutGPU(unsigned int *apBuf, unsigned int *delay,
		unsigned int *pfBC, unsigned int pfBCPitch,
		unsigned int *pfSC, unsigned int pfSCPitch)
{
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int tempOut;
	unsigned int *pfBCRow=(unsigned int *)((char *)pfBC+(index/numPFInPerBC)*pfBCPitch);
	unsigned int *pfSCRow=(unsigned int *)((char *)pfSC+(index/numPFInPerSC)*pfSCPitch);

	tempOut=(apBuf[index]&delay[index])>0;

	pfBCRow[index%numPFInPerBC]=tempOut;
	pfSCRow[index%numPFInPerSC]=tempOut;
}

template<unsigned int numPFInPerPC>
__global__ void updatePFPCOutGPU(unsigned int *apBuf, unsigned int *delay,
		float *synWeight, float *pfPC, unsigned int pfPCPitch)
{
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int tempOut;
	float *pfPCRow=(float *)((char *)pfPC+(index/numPFInPerPC)*pfPCPitch);

	tempOut=(apBuf[index]&delay[index])>0;

	pfPCRow[index%numPFInPerPC]=synWeight[index]*tempOut;
}

__global__ void updateGROutForHGPU(unsigned char *output, unsigned int *apBuf)
{
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	output[index]=apBuf[index]&0x00000001;
}
//**---------------end GR Kernels-------------------**

//**---------------IO kernels-----------------**
__global__ void updatePFPCSynIO(float *synWPFPC, unsigned long *historyGR, unsigned int offset, float plastStep)
{
	int i=blockIdx.x*blockDim.x+threadIdx.x+offset;
//	synWPFPC[i]=synWPFPC[i]+checkBinGR[i]*(doLTD*PFPCLTDDECPF+(!doLTD)*PFPCLTPINCPF);
	synWPFPC[i]=synWPFPC[i]+((historyGR[i]&0x0000008000000000)>0)*plastStep;

	synWPFPC[i]=(synWPFPC[i]>0)*synWPFPC[i];
	synWPFPC[i]=(synWPFPC[i]>1)+(synWPFPC[i]<=1)*synWPFPC[i];
}

//**---------------end IO kernels-------------**

//**---------------common kernels-------------**
template <unsigned int blockSize, typename Type, bool inMultiPitch, bool outMultiPitch>
__global__ void sumInputsNew(Type *input, unsigned int inputPitch, Type *output, unsigned int outputPitch, unsigned int rowLength)
{
	__shared__ Type sData[blockSize];

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

//**---------------end common kernels---------**

//**---------------analysis kernels-----------**

__global__ void updatePSHGPU(unsigned int *apBuf, unsigned int *psh, unsigned int pshP,
		int nBins, int tsPerBin, unsigned int extrashift)
{
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int tempBuf=apBuf[index]<<extrashift;

	for(int i=0; i<nBins; i++)
	{
		unsigned int *pshRow;
		unsigned int tempCount;

		pshRow=(unsigned int *)((char *)psh+i*pshP);
		tempCount=0;
		for(int j=0; j<tsPerBin; j++)
		{
			tempCount+=(tempBuf&0x80000000)>0;
			tempBuf=tempBuf<<1;
		}
		pshRow[index]+=tempCount;
	}
}
//**---------------end analysis kernels--------**


//**---------------kernel calls---------------**

void callGRActKernel(cudaStream_t &st, unsigned int nBlocks, unsigned int nThreadPerB, float *vGPU, float *gKCaGPU, float *threshGPU,
		unsigned int *apBufGPU, unsigned char *apOutGRGPU, float *gESumGPU, float *gISumGPU,
		float gLeak, float eLeak, float eGOIn,
		float threshBase, float threshMax, float threshDecay)
{
	calcActivityGRGPU<<<nBlocks, nThreadPerB, 0, st>>>(vGPU, gKCaGPU, threshGPU, apBufGPU, apOutGRGPU, gESumGPU, gISumGPU,
			gLeak, eLeak, eGOIn, threshBase, threshMax, threshDecay);
}

template<typename Type, unsigned int blockSize, bool inMultiP, bool outMultiP>
void callSumPFKernel(cudaStream_t &st, Type *inPFGPU, unsigned int inPFGPUP, Type *outPFSumGPU, unsigned int outPFSumGPUP,
		unsigned int nOutCells, unsigned int nOutCols, unsigned int rowLength)
{
	dim3 dimGrid(nOutCols, nOutCells);
	sumInputsNew<blockSize, Type, inMultiP, outMultiP><<<dimGrid, blockSize, 0, st>>>(inPFGPU, inPFGPUP, outPFSumGPU, outPFSumGPUP, rowLength);
}

template<unsigned int nCols, unsigned int nBlocks, unsigned int nThreadPerB>
void callSumGRGOOutKernel(cudaStream_t &st, unsigned int *grInGOGPU, unsigned int grInGOGPUPitch,
		unsigned int *grInGOSGPU)
{
	sumGRGOOutGPU<nCols><<<nBlocks, nThreadPerB, 0, st>>>(grInGOGPU, grInGOGPUPitch, grInGOSGPU);
}

template<unsigned int nCellIn, unsigned int nBlocks, unsigned int nThreadsPerB>
void callUpdateInGRKernel(cudaStream_t &st, unsigned int *apInGPU, float *gGPU, unsigned int gGPUP,
		unsigned int *conInGRGPU, unsigned int conInGRGPUP,
		int *numInPerGRGPU, float *gSumGPU, float gDecay, float gInc)
{
	updateGRInGPU<nCellIn><<<nBlocks, nThreadsPerB, 0, st>>>(apInGPU, gGPU, gGPUP, conInGRGPU, conInGRGPUP, numInPerGRGPU, gSumGPU, gDecay, gInc);
}

template<unsigned int numPFInPerBC, unsigned int numPFInPerSC, unsigned int numBlocks, unsigned int numThreadsPerB>
void callUpdatePFBCSCOutKernel(cudaStream_t &st, unsigned int *apBufGPU, unsigned int *delayMaskGPU,
		unsigned int *inPFBCGPU, unsigned int inPFBCGPUPitch,
		unsigned int *inPFSCGPU, unsigned int inPFSCGPUPitch)
{
	updatePFBCSCOutGPU<numPFInPerBC, numPFInPerSC><<<numBlocks, numThreadsPerB, 0, st>>>(apBufGPU, delayMaskGPU,
			inPFBCGPU, inPFBCGPUPitch, inPFSCGPU, inPFSCGPUPitch);
}

template<unsigned int numPFInPerPC, unsigned int numBlocks, unsigned int numThreadsPerB>
void callUpdatePFPCOutKernel(cudaStream_t &st, unsigned int *apBufGPU, unsigned int *delayMaskGPU,
		float *pfPCSynWGPU, float *inPFPCGPU, unsigned int inPFPCGPUPitch)
{
	updatePFPCOutGPU<numPFInPerPC><<<numBlocks, numThreadsPerB, 0, st>>>(apBufGPU, delayMaskGPU, pfPCSynWGPU,
			inPFPCGPU, inPFPCGPUPitch);
}

template<unsigned int nGO, unsigned int nRows, unsigned int threadsPerRow>
void callUpdateGROutGOKernel(cudaStream_t &st, unsigned int *apBufGPU,
		unsigned int *grInGOGPU, unsigned int grInGOGPUPitch,
		unsigned int *delayMasksGPU, unsigned int delayMasksGPUPitch,
		unsigned int *conGRtoGOGPU, unsigned int conGRtoGOGPUPitch,
		int *numGOPerGRGPU)
{
	updateGRGOOutGPU<nGO><<<nRows, threadsPerRow, 0, st>>>(apBufGPU, grInGOGPU, grInGOGPUPitch,
			delayMasksGPU, delayMasksGPUPitch, conGRtoGOGPU, conGRtoGOGPUPitch, numGOPerGRGPU);
}

template<unsigned int numBlocks, unsigned int numThreadsPerB>
void callUpdateGRHistKernel(cudaStream_t &st, unsigned int *apBufGPU, unsigned long *historyGPU)
{
		updateGRHistory<<<numBlocks, numThreadsPerB, 0, st>>>(apBufGPU, historyGPU);
}

template<unsigned int numBlocks, unsigned int numThreadsPerB>
void callUpdatePFPCPlasticityIOKernel(cudaStream_t &st, int plastTimerIO, float *synWeightGPU,
		unsigned long *historyGPU, int offSet, float pfPCLTDStep, float pfPCLTPStep)
{
	if(plastTimerIO<0)
	{
		updatePFPCSynIO<<<numBlocks, numThreadsPerB, 0, st>>>(synWeightGPU, historyGPU, offSet, pfPCLTDStep);
	}
	else if(plastTimerIO>100)
	{
		updatePFPCSynIO<<<numBlocks, numThreadsPerB, 0, st>>>(synWeightGPU, historyGPU, offSet, pfPCLTPStep);
	}
}

void callUpdatePSHGPU(unsigned int *apBufGPU, unsigned int *pshGPU, unsigned int pshGPUP,
		int nBins, int tsPerBin, unsigned int extrashift, unsigned int nBlocks, unsigned int nThreadPerB)
{
	updatePSHGPU<<<nBlocks, nThreadPerB>>>(apBufGPU, pshGPU, pshGPUP, nBins, tsPerBin, extrashift);
}

//**---------------end kernel calls------------**

//template initializations
template void callUpdatePFPCPlasticityIOKernel<256, 1024>
		(cudaStream_t &st, int plastTimerIO, float *synWeightGPU,
		unsigned long *historyGPU, int offSet, float pfPCLTDStep, float pfPCLTPStep);

template void callUpdatePFPCOutKernel<32768, 1024, 1024>
		(cudaStream_t &st, unsigned int *apBufGPU, unsigned int *delayMaskGPU,
		float *pfPCSynWGPU, float *inPFPCGPU, unsigned int inPFPCGPUPitch);

template void callSumPFKernel<float, 512, true, false>
		(cudaStream_t &st, float *inPFGPU, unsigned int inPFGPUP, float *outPFSumGPU, unsigned int outPFSumGPUP,
		unsigned int nOutCells, unsigned int nOutCols, unsigned int rowLength);

template void callSumPFKernel<unsigned int, 512, true, false>
		(cudaStream_t &st, unsigned int *inPFGPU, unsigned int inPFGPUP, unsigned int *outPFSumGPU, unsigned int outPFSumGPUP,
		unsigned int nOutCells, unsigned int nOutCols, unsigned int rowLength);

template void callSumGRGOOutKernel<1024, 2, 512>
		(cudaStream_t &st, unsigned int *grInGOGPU, unsigned int grInGOGPUPitch,
		unsigned int *grInGOSGPU);

template void callUpdateInGRKernel<1024, 1024, 1024>
		(cudaStream_t &st, unsigned int *apInGPU, float *gGPU, unsigned int gGPUP,
		unsigned int *conInGRGPU, unsigned int conInGRGPUP,
		int *numInPerGRGPU, float *gSumGPU, float gDecay, float gInc);

template void callUpdatePFBCSCOutKernel<8192, 2048, 1024, 1024>
		(cudaStream_t &st, unsigned int *apBufGPU, unsigned int *delayMaskGPU,
		unsigned int *inPFBCGPU, unsigned int inPFBCGPUPitch,
		unsigned int *inPFSCGPU, unsigned int inPFSCGPUPitch);

template void callUpdateGROutGOKernel<1024, 1024, 1024>
		(cudaStream_t &st, unsigned int *apBufGPU,
		unsigned int *grInGOGPU, unsigned int grInGOGPUPitch,
		unsigned int *delayMasksGPU, unsigned int delayMasksGPUPitch,
		unsigned int *conGRtoGOGPU, unsigned int conGRtoGOGPUPitch,
		int *numGOPerGRGPU);

template void callUpdateGRHistKernel<2048, 512>
		(cudaStream_t &st, unsigned int *apBufGPU, unsigned long *historyGPU);
