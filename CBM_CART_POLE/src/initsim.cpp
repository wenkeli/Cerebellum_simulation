/*
 * initsim.cpp
 *
 *  Created on: Feb 25, 2009
 *      Author: wen
 */

#include "../includes/initsim.h"

void resetVars()
{
	//initialize connectivity variables
	CRandomSFMT0 randGen(time(NULL));

	//initialize GR input compression
	compMask[0]=0x01;
	compMask[1]=0x02;
	compMask[2]=0x04;
	compMask[3]=0x08;
	compMask[4]=0x10;
	compMask[5]=0x20;
	compMask[6]=0x40;
	compMask[7]=0x80;
	//end compression initialization

	memset(numSynMFtoGR, 0, NUMMF*sizeof(short));
	memset(numSynMFtoGO, 0, NUMMF*sizeof(char));
	memset(numSynGOtoGR, 0, NUMMF*sizeof(short));
	memset(numSynGRtoGO, 0, NUMMF*sizeof(char));

	for(int i=0; i<NUMMF; i++)
	{
		for(int j=0; j<MFGRSYNPERMF; j++)
		{
			conMFtoGR[i][j]=NUMGR*4;
		}
		for(int j=0; j<MFGOSYNPERMF; j++)
		{
			conMFtoGO[i][j]=NUMGO;
		}
	}
	for(int i=0; i<NUMGO; i++)
	{
		for(int j=0; j<GOGRSYNPERGO; j++)
		{
			conGOtoGR[i][j]=NUMGR*4;
		}
	}
	for(int i=0; i<NUMGR; i++)
	{
		for(int j=0; j<GRGOSYNPERGR; j++)
		{
			conGRtoGO[i][j]=NUMGO;
		}
	}
	//end connectivity variables

	//initialize mossy fiber variables
	for(int i=0; i<NUMMF; i++)
	{
		typeMFs[i]=MFBG;
		threshMF[i]=1;
	}
	memset(bgFreqContsMF, 0, NUMCONTEXTS*NUMMF*sizeof(float));
	memset(incFreqMF, 0, NUMMF*sizeof(float));
	memset(csStartMF, 0, NUMMF*sizeof(short));
	memset(csEndMF, 0, NUMMF*sizeof(short));
//	memset(threshMF, 1, NUMMF*sizeof(float));
	// Nothing is on to begin with
	memset(apMF, false, NUMMF*sizeof(bool));
	memset(csOnMF, false, NUMMF*sizeof(bool));
	memset(histMF, false, NUMMF*sizeof(bool));
	//end mossy fiber variables

	//initialize golgi cell variables
	for(int i=0; i<NUMGO; i++)
	{
		vGO[i]=ELEAKGO-10+randGen.Random()*20;
		threshGO[i]=threshBaseInitGO-10+randGen.Random()*20;
		threshBaseGO[i]=threshBaseInitGO;
		gMFIncGO[i]=gMFIncInitGO;//*7;
		gGRIncGO[i]=gGRIncInitGO;//*7;
	}

	memset(apGO, false, NUMGO*sizeof(bool));
	memset(inputMFGO, 0, NUMGO*sizeof(short));
	memset(inputGRGO, 0, NUMGO*sizeof(short));
	memset(gMFGO, 0, NUMGO*sizeof(float));
	memset(gGRGO, 0, NUMGO*sizeof(float));
	memset(gMGluRGO, 0, NUMGO*sizeof(float));
	memset(gMGluRIncGO, 0, NUMGO*sizeof(float));
	memset(mGluRGO, 0, NUMGO*sizeof(float));
	memset(gluGO, 0, NUMGO*sizeof(float));

	//initialize purkinje cell variables
	memset(inputSumPFPC, 0, NUMPC*sizeof(float));
	memset(inputBCPC, 0, NUMPC*sizeof(unsigned char));
	memset(gPFPC, 0, NUMPC*sizeof(float));
	memset(gBCPC, 0, NUMPC*sizeof(float));
	memset(gSCPC, 0, NUMPC*SCPCSYNPERPC*sizeof(float));
	memset(apPC, false, NUMPC*sizeof(bool));
	memset(histAllAPPC, 0, NUMHISTBINSPC*sizeof(short));
	for(int i=0; i<NUMPC; i++)
	{
		vPC[i]=ELEAKPC;
		threshBasePC[i]=THRESHBASEPC;
		threshPC[i]=threshBasePC[i];
	}
	for(int i=0; i<NUMGR; i++)
	{
		pfSynWeightPC[i]=PFSYNWINITNC;
	}

	//initialize basket cell variables
	memset(inputSumPFBC, 0, NUMBC*sizeof(unsigned short));
	memset(inputPCBC, 0, NUMBC*sizeof(unsigned char));
	memset(gPFBC, 0, NUMBC*sizeof(float));
	memset(gPCBC, 0, NUMBC*sizeof(float));
	memset(apBC, false, NUMBC*sizeof(bool));
	for(int i=0; i<NUMPC; i++)
	{
		vBC[i]=ELEAKBC;
		threshBC[i]=THRESHBASEBC;
	}

	//initalize stellate cell variables
	memset(inputSumPFSC, 0, NUMSC*sizeof(unsigned short));
	memset(gPFSC, 0, NUMSC*sizeof(float));
	memset(apSC, false, NUMSC*sizeof(bool));
	for(int i=0; i<NUMSC; i++)
	{
		vSC[i]=ELEAKSC;
		threshSC[i]=THRESHBASESC;
	}

	//initialize inferior olivary cell variables
	memset(inputNCIO, false, NUMIO*NCIOSYNPERIO*sizeof(bool));
	memset(gNCIO, 0, NUMIO*NCIOSYNPERIO*sizeof(float));
	memset(gHIO, 0, NUMIO*sizeof(float));
	memset(gLtCaIO, 0, NUMIO*sizeof(float));
	memset(caIO, 0, NUMIO*sizeof(float));
	memset(gKCaIO, 0, NUMIO*sizeof(float));
	memset(gLtCaHIO, 0, NUMIO*sizeof(float));
	memset(vCoupIO, 0, NUMIO*sizeof(float));
	memset(apIO, 0, NUMIO*sizeof(bool));
	memset(plasticityPFPCTimerIO, 0, NUMIO*sizeof(short));
	for(int i=0; i<NUMIO; i++)
	{
		threshIO[i]=THRESHBASEIO;
		vIO[i]=ELEAKIO;
	}

	//initialize nucleus cells
	memset(inputPCNC, false, NUMNC*PCNCSYNPERNC*sizeof(bool));
	memset(gPCNC, 0, NUMNC*PCNCSYNPERNC*sizeof(float));
	memset(inputMFNC, false, NUMNC*MFNCSYNPERNC*sizeof(bool));
	memset(mfNMDANC, 0, NUMNC*MFNCSYNPERNC*sizeof(float));
	memset(mfAMPANC, 0, NUMNC*MFNCSYNPERNC*sizeof(float));
	memset(gMFNMDANC, 0, NUMNC*MFNCSYNPERNC*sizeof(float));
	memset(gMFAMPANC, 0, NUMNC*MFNCSYNPERNC*sizeof(float));
	memset(apNC, false, NUMNC*sizeof(bool));
	memset(synIOReleasePNC, 0, NUMNC*sizeof(float));
	memset(mfSynWChangeNC, 0, NUMNC*MFNCSYNPERNC*sizeof(float));
	for(int i=0; i<NUMNC; i++)
	{
		for(int j=0; j<PCNCSYNPERNC; j++)
		{
			gPCScaleNC[i][j]=GPCSCALEAVGNC*(1+(randGen.Random()-0.5)*0.2);
		}

		for(int j=0; j<MFNCSYNPERNC; j++)
		{
			mfSynWeightNC[i][j]=MFSYNWINITNC;
		}

		threshNC[i]=THRESHBASENC;
		vNC[i]=ELEAKNC;
	}

	//initialize simulation variables
	activeContext=0;
	activeCS=0;//1

	//initialize analysis variables
	memset(pshGR, 0, NUMGR*PSHNUMBINS*sizeof(unsigned short));
	pshGRMax=0;

	memset(pshGO, 0, NUMGO*PSHNUMBINS*sizeof(unsigned short));
	pshGOMax=0;

	memset(pshMF, 0, NUMMF*PSHNUMBINS*sizeof(unsigned short));
	pshMFMax=0;

	memset(pshPC, 0, NUMPC*PSHNUMBINS*sizeof(unsigned int));
	pshPCMax=0;

	memset(pshBC, 0, NUMBC*PSHNUMBINS*sizeof(unsigned int));
	pshBCMax=0;

	memset(pshSC, 0, NUMSC*PSHNUMBINS*sizeof(unsigned int));
	pshSCMax=0;

	memset(spikeSumGO, 0, NUMGO*sizeof(unsigned int));
	memset(spikeSumGR, 0, NUMGR*sizeof(unsigned int));
	memset(spikeSumSC, 0, NUMSC*sizeof(unsigned int));
	memset(spikeSumBC, 0, NUMBC*sizeof(unsigned int));
	memset(spikeSumPC, 0, NUMPC*sizeof(unsigned int));
	memset(spikeSumNC, 0, NUMNC*sizeof(unsigned int));
}

void resetActiveCS(char aCS)
{
	activeCS=aCS;
	updateMFCSOn();
}

void initSim()
{

}

void initCUDA()
{
	initGRCUDA();
	initPCCUDA();
	initBCCUDA();
	initSCCUDA();
}

void initGRCUDA()
{
	CRandomSFMT0 randGen(time(NULL));

	cudaSetDevice(0);

	//allocate memory for GPU
	cudaMalloc((void **)&vGRGPU, NUMGR*sizeof(float));
	cudaMalloc((void **)&gKCaGRGPU, NUMGR*sizeof(float));

	cudaMalloc((void **)&gEGR1GPU, NUMGR*sizeof(float));
	cudaMalloc((void **)&gEGR2GPU, NUMGR*sizeof(float));
	cudaMalloc((void **)&gEGR3GPU, NUMGR*sizeof(float));
	cudaMalloc((void **)&gEGR4GPU, NUMGR*sizeof(float));
	cudaMalloc((void **)&gEGRGPUSum, NUMGR*sizeof(float));

	cudaMalloc((void **)&gEIncGR1GPU, NUMGR*sizeof(float));
	cudaMalloc((void **)&gEIncGR2GPU, NUMGR*sizeof(float));
	cudaMalloc((void **)&gEIncGR3GPU, NUMGR*sizeof(float));
	cudaMalloc((void **)&gEIncGR4GPU, NUMGR*sizeof(float));

	cudaMalloc((void **)&gIGR1GPU, NUMGR*sizeof(float));
	cudaMalloc((void **)&gIGR2GPU, NUMGR*sizeof(float));
	cudaMalloc((void **)&gIGR3GPU, NUMGR*sizeof(float));
	cudaMalloc((void **)&gIGR4GPU, NUMGR*sizeof(float));
	cudaMalloc((void **)&gIGRGPUSum, NUMGR*sizeof(float));

	cudaMalloc((void **)&inputsGRGPU, NUMGR*sizeof(unsigned char));
	cudaMalloc((void **)&apOutGRGPU, NUMGR*sizeof(unsigned char));
	cudaMalloc((void **)&apBufGRGPU, NUMGR*sizeof(unsigned char));
	cudaMalloc((void **)&threshGRGPU, NUMGR*sizeof(float));
	cudaMalloc((void **)&threshBaseGRGPU, NUMGR*sizeof(float));

	cudaMallocPitch((void **)&historyGRGPU, (size_t *)&histGRGPUPitch, NUMGR*sizeof(unsigned char), NUMHISTBINSGR);

	//variables for conduction delays
	cudaMalloc((void **)&delayGOMask1GRGPU, NUMGR*sizeof(unsigned char));
	cudaMalloc((void **)&delayGOMask2GRGPU, NUMGR*sizeof(unsigned char));
	cudaMalloc((void **)&delayBCPCSCMaskGRGPU, NUMGR*sizeof(unsigned char));
	//end conduction delay
	//end GPU memory allocation

	//initialize GR GPU variables
	{
		//temp variables to assign values to for copying over to the GPU
		float vGRIni[NUMGR];
		float gKCaGRIni[NUMGR];
		float gENMDAGRIni[NUMGR];
		float gEMFIncGRIni[NUMGR];
		float gIGRIni[NUMGR];

		unsigned char inputsGRIni[NUMGR];
		unsigned char apGRIni[NUMGR];

		float threshGRIni[NUMGR];
		float threshBaseGRIni[NUMGR];

		//conduction delay variables
		unsigned char delayGOMask1GRIni[NUMGR];
		unsigned char delayGOMask2GRIni[NUMGR];
		unsigned char delayBCPCSCMaskGRIni[NUMGR];

		//initialize values to copy over to GPU
		for(int i=0; i<NUMGR; i++)
		{
			vGRIni[i]=ELEAKGR;//-10+randGen.Random()*20;
			threshGRIni[i]=threshBaseInitGR;//-10+randGen.Random()*20;
			threshBaseGRIni[i]=threshBaseInitGR;
			gEMFIncGRIni[i]=gEIncInitGR;

			delayBCPCSCMaskGRIni[i]=delayBCPCSCMaskGR[i];
			delayGOMask1GRIni[i]=delayGOMasksGR[i][0]; //0x01;
			delayGOMask2GRIni[i]=delayGOMasksGR[i][1]; //0x01;
		}
		memset(gKCaGRIni, 0, NUMGR*sizeof(float));
		memset(apGRIni, 0, NUMGR*sizeof(unsigned char));
		memset(inputsGRIni, 0, NUMGR*sizeof(unsigned char));
		memset(gIGRIni, 0, NUMGR*sizeof(float));
		memset(gENMDAGRIni, 0, NUMGR*sizeof(float));
		//end value initialization

		//copying to GPU
//		cudaSetDevice(0);
		cudaMemcpy(vGRGPU, vGRIni, NUMGR*sizeof(float), cudaMemcpyHostToDevice);

		cudaMemcpy(gKCaGRGPU, gKCaGRIni, NUMGR*sizeof(float), cudaMemcpyHostToDevice);

		cudaMemcpy(gEGR1GPU, gENMDAGRIni, NUMGR*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(gEGR2GPU, gENMDAGRIni, NUMGR*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(gEGR3GPU, gENMDAGRIni, NUMGR*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(gEGR4GPU, gENMDAGRIni, NUMGR*sizeof(float), cudaMemcpyHostToDevice);

		cudaMemcpy(gEIncGR1GPU, gEMFIncGRIni, NUMGR*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(gEIncGR2GPU, gEMFIncGRIni, NUMGR*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(gEIncGR3GPU, gEMFIncGRIni, NUMGR*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(gEIncGR4GPU, gEMFIncGRIni, NUMGR*sizeof(float), cudaMemcpyHostToDevice);

		cudaMemcpy(gIGR1GPU, gIGRIni, NUMGR*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(gIGR2GPU, gIGRIni, NUMGR*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(gIGR3GPU, gIGRIni, NUMGR*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(gIGR4GPU, gIGRIni, NUMGR*sizeof(float), cudaMemcpyHostToDevice);

		cudaMemcpy(inputsGRGPU, inputsGRIni, NUMGR*sizeof(unsigned char), cudaMemcpyHostToDevice);
		cudaMemcpy(apOutGRGPU, apGRIni, NUMGR*sizeof(unsigned char), cudaMemcpyHostToDevice);
		cudaMemcpy(apBufGRGPU, apGRIni, NUMGR*sizeof(unsigned char), cudaMemcpyHostToDevice);

		cudaMemcpy(threshGRGPU, threshGRIni, NUMGR*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(threshBaseGRGPU, threshBaseGRIni, NUMGR*sizeof(float), cudaMemcpyHostToDevice);

		cudaMemcpy(delayGOMask1GRGPU, delayGOMask1GRIni, NUMGR*sizeof(unsigned char), cudaMemcpyHostToDevice);
		cudaMemcpy(delayGOMask2GRGPU, delayGOMask2GRIni, NUMGR*sizeof(unsigned char), cudaMemcpyHostToDevice);
		cudaMemcpy(delayBCPCSCMaskGRGPU, delayBCPCSCMaskGRIni, NUMGR*sizeof(unsigned char), cudaMemcpyHostToDevice);
		//end copying to GPU
#ifdef GPUDEBUG
		for(int i=0; i<2; i++)
		{
			cout<<vGRIni[i]<<" ";
		}
		cout<<endl;
		memset(vGRIni, 0, NUMGR*sizeof(float));
		cout<<vGRIni[0]<<endl;
		cudaMemcpy(vGRIni, vGRGPU, NUMGR*sizeof(float), cudaMemcpyDeviceToHost);
		cout<<vGRIni[0]<<endl;
#endif
	}
	//end initialize granule cell variables
}

void initPCCUDA()
{
	cudaSetDevice(0);
	//allocate GPU memory
	cudaMalloc((void **)&pfSynWeightPCGPU, NUMGR*sizeof(float));
	cudaMallocPitch((void **)&inputPFPCGPU, (size_t *)&iPFPCGPUPitch, PFPCSYNPERPC*sizeof(float), NUMPC);
	cudaMallocPitch((void **)&tempSumPFPCGPU, (size_t *)&tempSumPFPCGPUPitch, 32*sizeof(float), NUMPC); //TODO: make 32 more generalized
	cudaMalloc((void **)&inputSumPFPCGPU, NUMPC*sizeof(float));
	//end GPU allocation

	//initialize purkinje cell variables
	{
//		float pfSynWeightPCIni[NUMGR];
//		for(int i=0; i<NUMGR; i++)
//		{
//			pfSynWeightPCIni[i]=PFSYNWINITNC;//0.5;
//		}

		cudaMemcpy(pfSynWeightPCGPU, pfSynWeightPC, NUMGR*sizeof(float), cudaMemcpyHostToDevice);
	}
	//end initialization
}

void initBCCUDA()
{
	cudaSetDevice(0);
	//allocate GPU memory
	cudaMallocPitch((void **)&inputPFBCGPU, (size_t *)&iPFBCGPUPitch, PFBCSYNPERBC*sizeof(unsigned short), NUMBC);
	cudaMallocPitch((void **)&tempSumPFBCGPU, (size_t *)&tempSumPFBCGPUPitch, 8*sizeof(unsigned short), NUMBC);
	cudaMalloc((void **)&inputSumPFBCGPU, NUMBC*sizeof(unsigned short));
	//end GPU allocation

	//initialize Basket cell variables

	//end initialization
}

void initSCCUDA()
{
	cudaSetDevice(0);
	//allocate GPU memory
	cudaMallocPitch((void **)&inputPFSCGPU, (size_t *)&iPFSCGPUPitch, PFSCSYNPERSC*sizeof(unsigned short), NUMSC);
	cudaMallocPitch((void **)&tempSumPFSCGPU, (size_t *)&tempSumPFSCGPUPitch, 2*sizeof(unsigned short), NUMSC);
	cudaMalloc((void **)&inputSumPFSCGPU, NUMSC*sizeof(unsigned short));
	//end GPU allocation

	//initialize Basket cell variables

	//end initialization
}

void assignMF()
{
	CRandomSFMT0 randGen(time(NULL));
//	int numMFAssign=0;
//
//	for(int i=0; i<NUMMF; i++)
//	{
//		typeMFs[i]=MFBG;
//	}
//
//	while(numMFAssign<NUMMF*MFPROPTONIC1)
//	{
//		int mfInd=(int)(randGen.Random()*NUMMF);
//		if(mfInd>=NUMMF)
//		{
//			mfInd=NUMMF-1;
//		}
//		if(typeMFs[mfInd]==MFBG)
//		{
//
//			typeMFs[mfInd]=MFCS1TON;//mossyFibers[mfInd].setMFType(MFCS1TON, &randGen);
//			numMFAssign++;
//		}
//	}
//
//	numMFAssign=0;
//	while(numMFAssign<NUMMF*MFPROPTONIC2)
//	{
//		int mfInd=(int)(randGen.Random()*NUMMF);
//		if(mfInd>=NUMMF)
//		{
//			mfInd=NUMMF-1;
//		}
//		if(typeMFs[mfInd]==MFBG)
//		{
//			typeMFs[mfInd]=MFCS2TON;//mossyFibers[mfInd].setMFType(MFCS2TON, &randGen);
//			numMFAssign++;
//		}
//	}
//
//	numMFAssign=0;
//	while(numMFAssign<NUMMF*MFPROPTONIC3)
//	{
//		int mfInd=(int)(randGen.Random()*NUMMF);
//		if(mfInd>=NUMMF)
//		{
//			mfInd=NUMMF-1;
//		}
//		if(typeMFs[mfInd]==MFBG)
//		{
//			typeMFs[mfInd]=MFCS2TON;//mossyFibers[mfInd].setMFType(MFCS3TON, &randGen);
//			numMFAssign++;
//		}
//	}
//
//	numMFAssign=0;
//	while(numMFAssign<NUMMF*MFPROPTONIC4)
//	{
//		int mfInd=(int)(randGen.Random()*NUMMF);
//		if(mfInd>=NUMMF)
//		{
//			mfInd=NUMMF-1;
//		}
//		if(typeMFs[mfInd]==MFBG)
//		{
//			typeMFs[mfInd]=MFCS4TON;//mossyFibers[mfInd].setMFType(MFCS4TON, &randGen);
//			numMFAssign++;
//		}
//	}
//
//	numMFAssign=0;
//	while(numMFAssign<NUMMF*MFPROPPHASIC1)
//	{
//		int mfInd=(int)(randGen.Random()*NUMMF);
//		if(mfInd>=NUMMF)
//		{
//			mfInd=NUMMF-1;
//		}
//		if(typeMFs[mfInd]==MFBG)
//		{
//			typeMFs[mfInd]=MFCS1PHA;//mossyFibers[mfInd].setMFType(MFCS1PHA, &randGen);
//			numMFAssign++;
//		}
//	}
//
//	numMFAssign=0;
//	while(numMFAssign<NUMMF*MFPROPPHASIC2)
//	{
//		int mfInd=(int)(randGen.Random()*NUMMF);
//		if(mfInd>=NUMMF)
//		{
//			mfInd=NUMMF-1;
//		}
//		if(typeMFs[mfInd]==MFBG)
//		{
//			typeMFs[mfInd]=MFCS2PHA;//mossyFibers[mfInd].setMFType(MFCS2PHA, &randGen);
//			numMFAssign++;
//		}
//	}
//
//	numMFAssign=0;
//	while(numMFAssign<NUMMF*MFPROPPHASIC3)
//	{
//		int mfInd=(int)(randGen.Random()*NUMMF);
//		if(mfInd>=NUMMF)
//		{
//			mfInd=NUMMF-1;
//		}
//		if(typeMFs[mfInd]==MFBG)
//		{
//			typeMFs[mfInd]=MFCS3PHA;//mossyFibers[mfInd].setMFType(MFCS3PHA, &randGen);
//			numMFAssign++;
//		}
//	}
//
//	numMFAssign=0;
//	while(numMFAssign<NUMMF*MFPROPPHASIC4)
//	{
//		int mfInd=(int)(randGen.Random()*NUMMF);
//		if(mfInd>=NUMMF)
//		{
//			mfInd=NUMMF-1;
//		}
//		if(typeMFs[mfInd]==MFBG)
//		{
//			typeMFs[mfInd]=MFCS4PHA;//mossyFibers[mfInd].setMFType(MFCS4PHA, &randGen);
//			numMFAssign++;
//		}
//	}
//
//	numMFAssign=0;
//	while(numMFAssign<NUMMF*MFPROPCONTEXT1)
//	{
//		int mfInd=(int)(randGen.Random()*NUMMF);
//		if(mfInd>=NUMMF)
//		{
//			mfInd=NUMMF-1;
//		}
//		if(typeMFs[mfInd]==MFBG)
//		{
//			typeMFs[mfInd]=MFCONT1;//mossyFibers[mfInd].setMFType(MFCONT1, &randGen);
//			numMFAssign++;
//		}
//	}
//
//	numMFAssign=0;
//	while(numMFAssign<NUMMF*MFPROPCONTEXT2)
//	{
//		int mfInd=(int)(randGen.Random()*NUMMF);
//		if(mfInd>=NUMMF)
//		{
//			mfInd=NUMMF-1;
//		}
//		if(typeMFs[mfInd]==MFBG)
//		{
//			typeMFs[mfInd]=MFCONT2;//mossyFibers[mfInd].setMFType(MFCONT2, &randGen);
//			numMFAssign++;
//		}
//	}
//
//#ifdef DEBUG
//	for(int i=0; i<NUMMF; i++)
//	{
//
//		cout<<"MF #"<<i<<" type: "<<(int)typeMFs[i]<<endl;
//	}
//#endif

	initMF(randGen);
//	updateMFCSOn();
}

void initMF(CRandomSFMT0 &randGen)
{
	for(int i=0; i<NUMCONTEXTS; i++)
	{
		for(int j=0; j<NUMMF; j++)
		{
			float tempRand;
			bool isMFBG;

			isMFBG=(typeMFs[j]==MFBG);
			tempRand=randGen.Random();
			tempRand=tempRand*(isMFBG*(MFBGNDFREQMAX-MFBGNDFREQMIN)+(!isMFBG)*(MFCSBGNDFREQMAX-MFCSBGNDFREQMIN));
			bgFreqContsMF[i][j]=((isMFBG*MFBGNDFREQMIN+(!isMFBG)*MFCSBGNDFREQMIN)+tempRand)*(TIMESTEP/1000);
		}
	}

//
//	for(int i=0; i<NUMMF; i++)
//	{
//		incFreqMF[i]=0;
//
//		if(typeMFs[i]==MFCS1TON)
//		{
//			incFreqMF[i]=MFTONICFREQINC1*(TIMESTEP/1000);
//			csStartMF[i]=csOnset[0]-1;
//			csEndMF[i]=csStartMF[i]+csDuration[0];
//		}
//
//		else if(typeMFs[i]==MFCS2TON)
//		{
//			incFreqMF[i]=MFTONICFREQINC2*(TIMESTEP/1000);
//			csStartMF[i]=csOnset[1]-1;
//			csEndMF[i]=csStartMF[i]+csDuration[1];
//		}
//
//		else if(typeMFs[i]==MFCS3TON)
//		{
//			incFreqMF[i]=MFTONICFREQINC3*(TIMESTEP/1000);
//			csStartMF[i]=csOnset[2]-1;
//			csEndMF[i]=csStartMF[i]+csDuration[2];
//		}
//
//		else if(typeMFs[i]==MFCS4TON)
//		{
//			incFreqMF[i]=MFTONICFREQINC4*(TIMESTEP/1000);
//			csStartMF[i]=csOnset[3]-1;
//			csEndMF[i]=csStartMF[i]+csOnset[3];
//		}
//
//		else if(typeMFs[i]==MFCS1PHA)
//		{
//			incFreqMF[i]=MFPHASFREQINC1*(TIMESTEP/1000);
//			csStartMF[i]=csOnset[0]-1;
//			csEndMF[i]=csStartMF[i]+MFPHASICDUR;
//		}
//
//		else if(typeMFs[i]==MFCS2PHA)
//		{
//			incFreqMF[i]=MFPHASFREQINC2*(TIMESTEP/1000);
//			csStartMF[i]=csOnset[1]-1;
//			csEndMF[i]=csStartMF[i]+MFPHASICDUR;
//		}
//
//		else if(typeMFs[i]==MFCS3PHA)
//		{
//			incFreqMF[i]=MFPHASFREQINC3*(TIMESTEP/1000);
//			csStartMF[i]=csOnset[2]-1;
//			csEndMF[i]=csStartMF[i]+MFPHASICDUR;
//		}
//
//		else if(typeMFs[i]==MFCS4PHA)
//		{
//			incFreqMF[i]=MFPHASFREQINC4*(TIMESTEP/1000);
//			csStartMF[i]=csOnset[3]-1;
//			csEndMF[i]=csStartMF[i]+MFPHASICDUR;
//		}
//
//		else if(typeMFs[i]==MFCONT1)
//		{
//			bgFreqContsMF[0][i]=MFCONTFREQMIN1+randGen.Random()*(MFCONTFREQMAX1-MFCONTFREQMIN1);
//			bgFreqContsMF[0][i]=bgFreqContsMF[0][i]*(TIMESTEP/1000);
//
//			bgFreqContsMF[1][i]=MFBGNDFREQMIN+randGen.Random()*(MFBGNDFREQMAX-MFBGNDFREQMIN);
//			bgFreqContsMF[1][i]=bgFreqContsMF[1][i]*(TIMESTEP/1000);
//		}
//
//		else if(typeMFs[i]==MFCONT2)
//		{
//			bgFreqContsMF[1][i]=MFCONTFREQMIN2+randGen.Random()*(MFCONTFREQMAX2-MFCONTFREQMIN2);
//			bgFreqContsMF[1][i]=bgFreqContsMF[1][i]*(TIMESTEP/1000);
//
//			bgFreqContsMF[0][i]=MFBGNDFREQMIN+randGen.Random()*(MFBGNDFREQMAX-MFBGNDFREQMIN);
//			bgFreqContsMF[0][i]=bgFreqContsMF[0][i]*(TIMESTEP/1000);
//		}
//
//	}
}

void updateMFCSOn()
{
	memset(csOnMF, false, NUMMF*(sizeof(bool)));

	for(short i=0; i<NUMMF; i++)
	{
		if(typeMFs[i]==activeCS+1 || typeMFs[i]==activeCS+5)
		{
			csOnMF[i]=true;
		}
	}
}
