/*
 * grKernels.cpp
 *
 *  Created on: Dec 3, 2009
 *      Author: wen
 */

#include "../includes/grKernels.h"
#include "commonCUDAKernels.cu"

/*updates excitatory and inhibitatory input conductances
inputsGR is the array that contains inputs from the mossy fiber and golgi cells
each byte in the inputsGR array cotains the complete input to a granule cell (to
achieve data compression). The bits in the byte is arranged in the following
manner, from most significant digit to least significant digit:
bit0: input from golgi cell to dendrite 4
bit1: input from golgi cell to dendrite 3
bit2: input from golgi cell to dendrite 2
bit3: input from golgi cell to dendrite 1
bit4: input from MF to dendrite 4
bit5: input from MF to dendrite 3
bit6: input from MF to dendrite 2
bit7: input from MF to dendrite 1
the code (inputsGR[i]&0xN)>>M performs a bitwise _and_ operation with the hex value N
and shifts the result to the right by M bits. This results in a 1 if the bit at the
position specified by N is 1, and 0 otherwise.
*/
template <unsigned char shiftN, unsigned char bitMask> __global__ void calcExG(unsigned char *input, float *exG, float *exGInc)
{
	__shared__ float inc[CUDAGRNUMTHREAD];
	__shared__ float org[CUDAGRNUMTHREAD];

	int tid=threadIdx.x;
	int index=blockIdx.x*blockDim.x+tid;
	inc[tid]=exGInc[index]*((input[index]&bitMask)>>shiftN);
	org[tid]=exG[index]*GEDECAYGR;
//	int i=blockIdx.x*blockDim.x+threadIdx.x;
//	exG[i]=exG[i]*GENMDADECAYGR+exGInc[i]*(float)(((input[i]&bitMask)>>shiftN));
	exG[index]=org[tid]+inc[tid];
}

template <unsigned char shiftN, unsigned char bitMask> __global__ void calcInG(unsigned char *input, float *inG)
{
	__shared__ float inc[CUDAGRNUMTHREAD];
	__shared__ float org[CUDAGRNUMTHREAD];

	int tid=threadIdx.x;
	int i=blockIdx.x*blockDim.x+threadIdx.x;
//	inG[i]=inG[i]*GIDECAYGR+GICONSTGR*(float)(((input[i]&bitMask)>>shiftN));
	inc[tid]=GICONSTINCGR*((input[i]&bitMask)>>shiftN);
	org[tid]=inG[i]*GIDECAYGR;
	inG[i]=org[tid]+inc[tid];
}

__global__ void sumG(float *g1, float *g2, float *g3, float *g4, float *gSum)
{
	__shared__ float s1[CUDAGRNUMTHREAD];
	__shared__ float s2[CUDAGRNUMTHREAD];
	int tid=threadIdx.x;
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	s1[tid]=g1[i]+g2[i];
	s2[tid]=g3[i]+g4[i];
	gSum[i]=s1[tid]+s2[tid];
//	gSum[i]=g1[i]+g2[i]+g3[i]+g4[i];
}

//__global__ void calcinputG(unsigned char *input,
//		float *inG1, float *inG2, float *inG3, float *inG4,
//		float *exG1, float *exG2,  float *exG3, float *exG4,
//		float *exGInc1, float *exGInc2, float *exGInc3, float *exGInc4,
//		float *vGR, float *gKCa, float *threshGR, float *threshBaseGR, unsigned char *apGR)
//{
//	__shared__ unsigned char s_input[CUDAGRNUMTHREAD];
//	__shared__ float s_inG1[CUDAGRNUMTHREAD];
//	__shared__ float s_inG2[CUDAGRNUMTHREAD];
//	__shared__ float s_inG3[CUDAGRNUMTHREAD];
//	__shared__ float s_inG4[CUDAGRNUMTHREAD];
//	__shared__ float s_exG1[CUDAGRNUMTHREAD];
//	__shared__ float s_exG2[CUDAGRNUMTHREAD];
//	__shared__ float s_exG3[CUDAGRNUMTHREAD];
//	__shared__ float s_exG4[CUDAGRNUMTHREAD];
//
//	int tid=threadIdx.x;
//	int i=blockIdx.x*blockDim.x+tid;
//
//	s_input[tid]=input[i];
//
//	s_inG1[tid]=inG1[i]*GIDECAYGR+GICONSTGR*((s_input[tid]&0x10)>>4);
//	s_inG2[tid]=inG2[i]*GIDECAYGR+GICONSTGR*((s_input[tid]&0x20)>>5);
//	s_inG3[tid]=inG3[i]*GIDECAYGR+GICONSTGR*((s_input[tid]&0x40)>>6);
//	s_inG4[tid]=inG4[i]*GIDECAYGR+GICONSTGR*((s_input[tid]&0x80)>>7);
//
//	s_exG1[tid]=exG1[i]*GENMDADECAYGR+exGInc1[i]*(s_input[tid]&0x01);
//	s_exG2[tid]=exG2[i]*GENMDADECAYGR+exGInc2[i]*((s_input[tid]&0x02)>>1);
//	s_exG3[tid]=exG3[i]*GENMDADECAYGR+exGInc3[i]*((s_input[tid]&0x04)>>2);
//	s_exG4[tid]=exG4[i]*GENMDADECAYGR+exGInc4[i]*((s_input[tid]&0x08)>>3);
//
//	exGSum[i]=s_exG1[tid]+s_exG2[tid]+s_exG3[tid]+s_exG4[tid];
//	inGSum[i]=s_inG1[tid]+s_inG2[tid]+s_inG3[tid]+s_inG4[tid];
//
//	exG1[i]=s_exG1[tid];
//	exG2[i]=s_exG2[tid];
//	exG3[i]=s_exG3[tid];
//	exG4[i]=s_exG4[tid];
//
//	inG1[i]=s_inG1[tid];
//	inG2[i]=s_inG2[tid];
//	inG3[i]=s_inG3[tid];
//	inG4[i]=s_inG4[tid];
//}

__global__ void calcVmAndThresh(float *vGR, float *gKCa, float *gE, float *gI, float *threshGR, float *threshBaseGR)
{
	__shared__ float gk[CUDAGRNUMTHREAD];
	__shared__ float v[CUDAGRNUMTHREAD];
//	__shared__ float vdec[CUDAGRNUMTHREAD];

	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int tid=threadIdx.x;

	gk[tid]=gKCa[i];
	v[tid]=vGR[i];
	v[tid]=v[tid]+(GLEAKGR+gk[tid]*gk[tid]*gk[tid]*gk[tid])*(ELEAKGR-v[tid]);
	gKCa[i]=gk[tid]*0.9999f;
	vGR[i]=v[tid]-gE[i]*v[tid]+gI[i]*(EGOGR-v[tid]);

//	vGR[i]=vGR[i]+(GLEAKGR+gKCa[i]*gKCa[i]*gKCa[i]*gKCa[i])*(ELEAKGR-vGR[i]);
//	gKCa[i]=gKCa[i]*0.9999f;
//	vGR[i]=vGR[i]-(gE[i])*vGR[i]+(gI[i])*(EGABAGR-vGR[i]);
	threshGR[i]=threshGR[i]+(threshBaseGR[i]-threshGR[i])*THRESHDECAYGR;
}

__global__ void calcAPAndThresh(float *vGR, float *threshGR, unsigned char *apGR, float *gKCa)
{
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	apGR[i]=vGR[i]>threshGR[i];

	threshGR[i]=apGR[i]*THRESHMAXGR+(!apGR[i])*threshGR[i];
	gKCa[i]=apGR[i]*(gKCa[i]+0.003f)+(!apGR[i])*gKCa[i];


}


__global__ void updateHistory(unsigned char *apGR, unsigned char *historyGR, int binPitch, unsigned short binN, unsigned char resetBin)
{
	unsigned char *updateBinGR=(unsigned char *)((char *)historyGR+binN*binPitch);
	int i=blockIdx.x*blockDim.x+threadIdx.x;

//	if(resetBin)
//	{
//		updateBinGR[i]=0;
//	}

	updateBinGR[i]=((!resetBin) && updateBinGR[i]) || apGR[i];

//	updateBinGR[i]+=apGR[i];
}



__global__ void calcActivity(float *vGR, float *gKCa,
		float *gENMDAGR1, float *gENMDAGR2, float *gENMDAGR3, float *gENMDAGR4,
		float *gEMFIncGR1, float *gEMFIncGR2, float *gEMFIncGR3, float *gEMFIncGR4,
		float *gIGR1, float *gIGR2, float *gIGR3, float *gIGR4,
		unsigned char *inputsGR, unsigned char *apGR, float *threshGR, float *threshBaseGR)
{
	/*in the GPU, this specifies which cell to calculate for
	CUDAGRNUMTHREAD is defined to be 512
	blockIdx.x ranges from 0 to 2047 and threadIdx.x ranges from 0 to 511
	*/
	int i=blockIdx.x*CUDAGRNUMTHREAD+threadIdx.x;
	unsigned char input;
	float tGENMDA1, tGENMDA2, tGENMDA3, tGENMDA4;
	float tGI1, tGI2, tGI3, tGI4;
	float tempV;
	float tempGKCA;
	float tempThresh;
	unsigned char tempAP;

	tempGKCA=gKCa[i];
	tempV=vGR[i]+(GLEAKGR+tempGKCA*tempGKCA*tempGKCA)*(ELEAKGR-vGR[i]);
	tempGKCA=tempGKCA*0.9999f;

	input=inputsGR[i];
	tGENMDA1=gENMDAGR1[i]*GEDECAYGR+gEMFIncGR1[i]*(float)(input&0x01);
	tGENMDA2=gENMDAGR2[i]*GEDECAYGR+gEMFIncGR2[i]*(float)((input&0x02)>>1);
	tGENMDA3=gENMDAGR3[i]*GEDECAYGR+gEMFIncGR3[i]*(float)((input&0x04)>>2);
	tGENMDA4=gENMDAGR4[i]*GEDECAYGR+gEMFIncGR4[i]*(float)((input&0x08)>>3);

	tGI1=gIGR1[i]*GIDECAYGR+GICONSTINCGR*(float)((input&0x10)>>4);
	tGI2=gIGR2[i]*GIDECAYGR+GICONSTINCGR*(float)((input&0x20)>>5);
	tGI3=gIGR3[i]*GIDECAYGR+GICONSTINCGR*(float)((input&0x40)>>6);
	tGI4=gIGR4[i]*GIDECAYGR+GICONSTINCGR*(float)((input&0x80)>>7);

	tempV=tempV-(tGENMDA1+tGENMDA2+tGENMDA3+tGENMDA4)*tempV+(tGI1+tGI2+tGI3+tGI3)*(EGOGR-tempV);

	tempThresh=threshGR[i];
	tempThresh=tempThresh+(threshBaseGR[i]-tempThresh)*THRESHDECAYGR;
	tempAP=tempV>tempThresh;

	threshGR[i]=tempAP*THRESHMAXGR+(!tempAP)*tempThresh;
	gKCa[i]=tempAP*(tempGKCA+0.003f)+(!tempAP)*tempGKCA;

	apGR[i]=(apGR[i]<<1)|(tempAP);
	vGR[i]=tempV;

	gENMDAGR1[i]=tGENMDA1;
	gENMDAGR2[i]=tGENMDA2;
	gENMDAGR3[i]=tGENMDA3;
	gENMDAGR4[i]=tGENMDA4;

	gIGR1[i]=tGI1;
	gIGR2[i]=tGI2;
	gIGR3[i]=tGI3;
	gIGR4[i]=tGI4;
}

__global__ void calcOutPut(unsigned char *apBuf, unsigned char *delayGOMask1, unsigned char *delayGOMask2, unsigned char *delayBCPCSCMask, unsigned char *apOut)
{
	int i=blockIdx.x*CUDAGRNUMTHREAD+threadIdx.x;
	unsigned char tempOut;
	unsigned char tempBuf;
	tempBuf=apBuf[i];
	tempOut=tempBuf&0x01;

//	tempOut=tempOut|(((tempBuf&0x01))*0x02);
//	tempOut=tempOut|(((tempBuf&0x01))*0x04);
//	apOut[i]=tempOut|(((tempBuf&0x01))*0x08);
	tempOut=tempOut|(((tempBuf&delayGOMask1[i]))*0x02);
	tempOut=tempOut|(((tempBuf&delayGOMask2[i]))*0x04);
	apOut[i]=tempOut|(((tempBuf&delayBCPCSCMask[i]))*0x08);
}

void runGRKernels(short t)
{
//	calcExG<0, 0x01><<<CUDAGRNUMTBLOCK, CUDAGRNUMTHREAD>>>(inputsGRGPU, gEGR1GPU, gEIncGR1GPU);
//	calcExG<1, 0x02><<<CUDAGRNUMTBLOCK, CUDAGRNUMTHREAD>>>(inputsGRGPU, gEGR2GPU, gEIncGR2GPU);
//	calcExG<2, 0x04><<<CUDAGRNUMTBLOCK, CUDAGRNUMTHREAD>>>(inputsGRGPU, gEGR3GPU, gEIncGR3GPU);
//	calcExG<3, 0x08><<<CUDAGRNUMTBLOCK, CUDAGRNUMTHREAD>>>(inputsGRGPU, gEGR4GPU, gEIncGR4GPU);
//
//	calcInG<4, 0x10><<<CUDAGRNUMTBLOCK, CUDAGRNUMTHREAD>>>(inputsGRGPU, gIGR1GPU);
//	calcInG<5, 0x20><<<CUDAGRNUMTBLOCK, CUDAGRNUMTHREAD>>>(inputsGRGPU, gIGR2GPU);
//	calcInG<6, 0x40><<<CUDAGRNUMTBLOCK, CUDAGRNUMTHREAD>>>(inputsGRGPU, gIGR3GPU);
//	calcInG<7, 0x80><<<CUDAGRNUMTBLOCK, CUDAGRNUMTHREAD>>>(inputsGRGPU, gIGR4GPU);
//	cudaThreadSynchronize();
//
//	sumG<<<CUDAGRNUMTBLOCK, CUDAGRNUMTHREAD>>>(gEGR1GPU, gEGR2GPU, gEGR3GPU, gEGR4GPU, gEGRGPUSum);
//	sumG<<<CUDAGRNUMTBLOCK, CUDAGRNUMTHREAD>>>(gIGR1GPU, gIGR2GPU, gIGR3GPU, gIGR4GPU, gIGRGPUSum);
//
////	calcinputG<<<CUDAGRNUMTBLOCK, CUDAGRNUMTHREAD>>>(inputsGRGPU, gIGR1GPU, gIGR2GPU, gIGR3GPU, gIGR4GPU,
////			gENMDAGR1GPU, gENMDAGR2GPU, gENMDAGR3GPU, gENMDAGR4GPU,
////			gEMFIncGR1GPU, gEMFIncGR2GPU, gEMFIncGR3GPU, gEMFIncGR4GPU,
////			gENMDAGRGPUSum, gIGRGPUSum);
//	cudaThreadSynchronize();
//
//	calcVmAndThresh<<<CUDAGRNUMTBLOCK, CUDAGRNUMTHREAD>>>(vGRGPU, gKCaGRGPU, gEGRGPUSum, gIGRGPUSum, threshGRGPU, threshBaseGRGPU);
//	cudaThreadSynchronize();
//
//	calcAPAndThresh<<<CUDAGRNUMTBLOCK, CUDAGRNUMTHREAD>>>(vGRGPU, threshGRGPU, apGRGPU, gKCaGRGPU);
//	cudaThreadSynchronize();

	calcActivity<<<CUDAGRNUMTBLOCK, CUDAGRNUMTHREAD>>>(vGRGPU, gKCaGRGPU,
			gEGR1GPU, gEGR2GPU, gEGR3GPU, gEGR4GPU,
			gEIncGR1GPU, gEIncGR2GPU, gEIncGR3GPU, gEIncGR4GPU,
			gIGR1GPU, gIGR2GPU, gIGR3GPU, gIGR4GPU,
			inputsGRGPU, apBufGRGPU, threshGRGPU, threshBaseGRGPU);
	cudaThreadSynchronize();

	calcOutPut<<<CUDAGRNUMTBLOCK, CUDAGRNUMTHREAD>>>(apBufGRGPU, delayGOMask1GRGPU, delayGOMask2GRGPU, delayBCPCSCMaskGRGPU, apOutGRGPU);
	cudaThreadSynchronize();
	//update history bin of GR cells

	if((t%HISTBINWIDTHGR)==0)
	{
		histBinNGR++;
		histBinNGR=histBinNGR%NUMHISTBINSGR;
	}
	updateHistory<<<CUDAGRNUMTBLOCK, CUDAGRNUMTHREAD>>>(apOutGRGPU, historyGRGPU, histGRGPUPitch, histBinNGR, (unsigned char)((t%HISTBINWIDTHGR)==0));
	//end update
	cudaThreadSynchronize();
}


//	calcActivity<<<CUDAGRNUMTBLOCK, CUDAGRNUMTHREAD>>>(vGRGPU, gKCaGRGPU,
//			gENMDAGR1GPU, gENMDAGR2GPU, gENMDAGR3GPU, gENMDAGR4GPU,
//			gEMFIncGR1GPU, gEMFIncGR2GPU, gEMFIncGR3GPU, gEMFIncGR4GPU,
//			gIGR1GPU, gIGR2GPU, gIGR3GPU, gIGR4GPU,
//			inputsGRGPU, apGRGPU, threshGRGPU, threshBaseGRGPU);

