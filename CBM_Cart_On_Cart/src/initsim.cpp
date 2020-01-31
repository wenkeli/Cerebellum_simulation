/*
 * initsim.cpp
 *
 *  Created on: Feb 25, 2009
 *      Author: wen
 */

#include "../includes/initsim.h"

void initCUDA()
{
	cudaError_t error;
	cudaSetDevice(0);
	for(int i=0; i<CUDANUMSTREAMS; i++)
	{
		cudaStreamCreate(&streams[i]);
	}
	error=cudaGetLastError();
	cout<<"CUDA dev: "<<cudaGetErrorString(error)<<endl;

//	inputNetwork->initCUDA();
//
//	//must have this code occur only after initBCCUDA
//	for(int i=0; i<NUMMZONES; i++)
//	{
//		zones[i]->initCUDA(inputNetwork->exportPFBCSum(), inputNetwork->exportApBufGRGPU(),
//				inputNetwork->exportDelayBCPCSCMaskGPU(), inputNetwork->exportHistGRGPU());
//		error=cudaGetLastError();
//		cout<<"CUDA mz "<<i<<" init: "<<cudaGetErrorString(error)<<endl;
//	}
}

void newSim()
{
	unsigned int dummy;

	initCUDA();
	for(int i=0; i<NUMMZONES; i++)
	{
	#if defined EYELID
		errMod[i] =new ECErrorInput(MAXUSDR, 0, TIMESTEP, TSUNITINS, USONSET);
	#endif
	#if defined CARTPOLE
		errMod[i] = new CPErrorInput(MAXUSDR, 0, TIMESTEP, TSUNITINS);
	#endif
		cout<<"error input module init"<<endl;
		//initialize nucleus cells
	#if defined EYELID
		outputMod[i]=new ECOutput(NUMNC, TIMESTEP, TSUNITINS);
	#endif
	#if defined CARTPOLE
		outputMod[i] = new CPOutput(NUMNC, TIMESTEP, TSUNITINS);
	#endif
		cout<<"output module init"<<endl;
	}

#if defined EYELID
	externalMod=new DummyExternal(TIMESTEP, TSUNITINS, errMod, outputMod, NUMMZONES);
#endif
#if defined CARTPOLE
	externalMod = new CartPole(TIMESTEP, TSUNITINS, errMod, outputMod, NUMMZONES);
#endif
	cout<<"external module init"<<endl;
#if defined EYELID
	mfMod=new ECMFInput(NUMMF, csOnset[0], csOnset[0]+csDuration[0], csOnset[0]+MFPHASICDUR,
		 		MFPROPTONIC1, MFPROPPHASIC1, MFPROPCONTEXT1, TIMESTEP, TSUNITINS);
#endif
#if defined CARTPOLE
	mfMod = new CPMFInput(NUMMF, TIMESTEP, TSUNITINS, (CartPole*) externalMod);
#endif
	cout<<"mf module init"<<endl;

//	inputNetwork=new InNet(mfMod->exportApMF(dummy));
//	inputNetwork=new InNetNoGO(mfMod->exportApMF(dummy));
//	inputNetwork=new InNetNoGRGO(mfMod->exportApMF(dummy));
//	inputNetwork=new InNetNoMFGO(mfMod->exportApMF(dummy));
	inputNetwork=new InNetSparseGRGO(mfMod->exportApMF(dummy));

	for(int i=0; i<NUMMZONES; i++)
	{
		zones[i]=new MZone(inputNetwork->exportApSC(), mfMod->exportApMF(dummy), inputNetwork->exportHistMF(),
				inputNetwork->exportPFBCSum(), inputNetwork->exportApBufGRGPU(),
				inputNetwork->exportDelayBCPCSCMaskGPU(), inputNetwork->exportHistGRGPU());
	}
	cout<<"mz modules init"<<endl;


	//PSH initialization
	cout<<"PSH initialization"<<endl;

	mfPSH=new PSHAnalysis(InNet::numMF, inputNetwork->exportApBufMF(),
			PSHPRESTIMNUMBINS, PSHSTIMNUMBINS, PSHPOSTSTIMNUMBINS,
			PSHBINWIDTH, APBUFWIDTH, NUMBINSINAPBUF);
	goPSH=new PSHAnalysis(InNet::numGO, inputNetwork->exportApBufGO(),
			PSHPRESTIMNUMBINS, PSHSTIMNUMBINS, PSHPOSTSTIMNUMBINS,
			PSHBINWIDTH, APBUFWIDTH, NUMBINSINAPBUF);
	grPSH=new PSHAnalysisGPU(InNet::numGR, inputNetwork->exportApBufGRGPU(),
			PSHPRESTIMNUMBINS, PSHSTIMNUMBINS, PSHPOSTSTIMNUMBINS,
			PSHBINWIDTH, APBUFWIDTH, NUMBINSINAPBUF, 2048, 512);
	scPSH=new PSHAnalysis(InNet::numSC, inputNetwork->exportApBufSC(),
			PSHPRESTIMNUMBINS, PSHSTIMNUMBINS, PSHPOSTSTIMNUMBINS,
			PSHBINWIDTH, APBUFWIDTH, NUMBINSINAPBUF);

	for(int i=0; i<NUMMZONES; i++)
	{
		bcPSH[i]=new PSHAnalysis(MZone::numBC, zones[i]->exportApBufBC(),
				PSHPRESTIMNUMBINS, PSHSTIMNUMBINS, PSHPOSTSTIMNUMBINS,
				PSHBINWIDTH, APBUFWIDTH, NUMBINSINAPBUF);
		pcPSH[i]=new PSHAnalysis(MZone::numPC, zones[i]->exportApBufPC(),
				PSHPRESTIMNUMBINS, PSHSTIMNUMBINS, PSHPOSTSTIMNUMBINS,
				PSHBINWIDTH, APBUFWIDTH, NUMBINSINAPBUF);
		ioPSH[i]=new PSHAnalysis(MZone::numIO, zones[i]->exportApBufIO(),
				PSHPRESTIMNUMBINS, PSHSTIMNUMBINS, PSHPOSTSTIMNUMBINS,
				PSHBINWIDTH, APBUFWIDTH, NUMBINSINAPBUF);
		ncPSH[i]=new PSHAnalysis(MZone::numNC, zones[i]->exportApBufNC(),
				PSHPRESTIMNUMBINS, PSHSTIMNUMBINS, PSHPOSTSTIMNUMBINS,
				PSHBINWIDTH, APBUFWIDTH, NUMBINSINAPBUF);
	}
	cout<<"done"<<endl;
}

void readSimIn(ifstream &infile)
{
	unsigned int dummy;

	initCUDA();
	for(int i=0; i<NUMMZONES; i++)
	{
	#if defined EYELID
		errMod[i]=new ECErrorInput(infile);
	#endif
	#if defined CARTPOLE
		errMod[i]=new CPErrorInput(infile);
	#endif
		cout<<"error input module init"<<endl;
		//initialize nucleus cells
	#if defined EYELID
		outputMod[i]=new ECOutput(infile);
	#endif
	#if defined CARTPOLE
		outputMod[i]=new CPOutput(infile);
	#endif
		cout<<"output module init"<<endl;
	}

#if defined EYELID
	externalMod=new DummyExternal(infile, errMod, outputMod, NUMMZONES);
#endif
#if defined CARTPOLE
	externalMod=new CartPole(infile, errMod, outputMod, NUMMZONES);
#endif
	cout<<"external module init"<<endl;
#if defined EYELID
//		 		MFPROPTONIC1, MFPROPPHASIC1, MFPROPCONTEXT1, TIMESTEP, TSUNITINS);
	mfMod=new ECMFInput(infile);
#endif
#if defined CARTPOLE
	mfMod=new CPMFInput(infile, (CartPole *) externalMod);
#endif
	cout<<"mf module init"<<endl;

//	inputNetwork=new InNet(infile, mfMod->exportApMF(dummy));
//	inputNetwork=new InNetNoGO(infile, mfMod->exportApMF(dummy));
//	inputNetwork=new InNetNoGRGO(infile, mfMod->exportApMF(dummy));
//	inputNetwork=new InNetNoMFGO(infile, mfMod->exportApMF(dummy));
	inputNetwork=new InNetSparseGRGO(infile, mfMod->exportApMF(dummy));

	for(int i=0; i<NUMMZONES; i++)
	{
		zones[i]=new MZone(infile, inputNetwork->exportApSC(), mfMod->exportApMF(dummy), inputNetwork->exportHistMF(),
				inputNetwork->exportPFBCSum(), inputNetwork->exportApBufGRGPU(),
				inputNetwork->exportDelayBCPCSCMaskGPU(), inputNetwork->exportHistGRGPU());
	}
	cout<<"mz modules init"<<endl;

	cout<<"PSH initialization"<<endl;

	mfPSH=new PSHAnalysis(InNet::numMF, inputNetwork->exportApBufMF(),
			PSHPRESTIMNUMBINS, PSHSTIMNUMBINS, PSHPOSTSTIMNUMBINS,
			PSHBINWIDTH, APBUFWIDTH, NUMBINSINAPBUF);
	goPSH=new PSHAnalysis(InNet::numGO, inputNetwork->exportApBufGO(),
			PSHPRESTIMNUMBINS, PSHSTIMNUMBINS, PSHPOSTSTIMNUMBINS,
			PSHBINWIDTH, APBUFWIDTH, NUMBINSINAPBUF);
	grPSH=new PSHAnalysisGPU(InNet::numGR, inputNetwork->exportApBufGRGPU(),
			PSHPRESTIMNUMBINS, PSHSTIMNUMBINS, PSHPOSTSTIMNUMBINS,
			PSHBINWIDTH, APBUFWIDTH, NUMBINSINAPBUF, 2048, 512);
	scPSH=new PSHAnalysis(InNet::numSC, inputNetwork->exportApBufSC(),
			PSHPRESTIMNUMBINS, PSHSTIMNUMBINS, PSHPOSTSTIMNUMBINS,
			PSHBINWIDTH, APBUFWIDTH, NUMBINSINAPBUF);

	for(int i=0; i<NUMMZONES; i++)
	{
		bcPSH[i]=new PSHAnalysis(MZone::numBC, zones[i]->exportApBufBC(),
				PSHPRESTIMNUMBINS, PSHSTIMNUMBINS, PSHPOSTSTIMNUMBINS,
				PSHBINWIDTH, APBUFWIDTH, NUMBINSINAPBUF);
		pcPSH[i]=new PSHAnalysis(MZone::numPC, zones[i]->exportApBufPC(),
				PSHPRESTIMNUMBINS, PSHSTIMNUMBINS, PSHPOSTSTIMNUMBINS,
				PSHBINWIDTH, APBUFWIDTH, NUMBINSINAPBUF);
		ioPSH[i]=new PSHAnalysis(MZone::numIO, zones[i]->exportApBufIO(),
				PSHPRESTIMNUMBINS, PSHSTIMNUMBINS, PSHPOSTSTIMNUMBINS,
				PSHBINWIDTH, APBUFWIDTH, NUMBINSINAPBUF);
		ncPSH[i]=new PSHAnalysis(MZone::numNC, zones[i]->exportApBufNC(),
				PSHPRESTIMNUMBINS, PSHSTIMNUMBINS, PSHPOSTSTIMNUMBINS,
				PSHBINWIDTH, APBUFWIDTH, NUMBINSINAPBUF);
	}
	cout<<"done"<<endl;
}

void writeSimOut(ofstream &outfile)
{
	cout<<"writing sim out..."<<endl;
	for(int i=0; i<NUMMZONES; i++)
	{
		errMod[i]->exportState(outfile);
		outputMod[i]->exportState(outfile);
	}

	externalMod->exportState(outfile);

	mfMod->exportState(outfile);

	inputNetwork->exportState(outfile);

	for(int i=0; i<NUMMZONES; i++)
	{
		zones[i]->exportState(outfile);
	}
	cout<<"done"<<endl;
}

void writePSHOut(ofstream &outfile)
{
	cout<<"writing psh"<<endl;
	mfPSH->exportPSH(outfile);
	goPSH->exportPSH(outfile);
	grPSH->exportPSH(outfile);
	scPSH->exportPSH(outfile);

	for(int i=0; i<NUMMZONES; i++)
	{
		bcPSH[i]->exportPSH(outfile);
		pcPSH[i]->exportPSH(outfile);
		ioPSH[i]->exportPSH(outfile);
		ncPSH[i]->exportPSH(outfile);
	}

	cout<<"done"<<endl;
}

void cleanSim()
{
        delete randGen;
	delete[] errMod;
	delete[] outputMod;

	delete externalMod;
	delete mfMod;
	delete inputNetwork;
	delete[] zones;

	delete mfPSH;
	delete goPSH;
	delete grPSH;
	delete scPSH;
	delete[] bcPSH;
	delete[] pcPSH;
	delete[] ioPSH;
	delete[] ncPSH;
	cudaDeviceReset();
}


