/*
 * mzone.cpp
 *
 *  Created on: Jun 14, 2011
 *      Author: consciousness
 */

#include "../../includes/mzonemodules/mzone.h"
#include "../../includes/globalvars.h"

//basket cell const values
const float MZone::eLeakBC=-70;
const float MZone::ePCBC=-70;
const float MZone::gLeakBC=0.2/(6-TIMESTEP);
const float MZone::gPFDecayTBC=4.15;
const float MZone::gPFDecayBC=exp(-TIMESTEP/MZone::gPFDecayTBC);
const float MZone::gPCDecayTBC=5;
const float MZone::gPCDecayBC=exp(-TIMESTEP/MZone::gPCDecayTBC);
const float MZone::threshDecayTBC=10;
const float MZone::threshDecayBC=1-exp(-TIMESTEP/MZone::threshDecayTBC);
const float MZone::threshBaseBC=-50;
const float MZone::threshMaxBC=0;
const float MZone::pfIncConstBC=0.00045;//0.00055;//0.0007;
const float MZone::pcIncConstBC=0.002;//0.02;

//purkinje cell const values
const float MZone::pfSynWInitPC=0.5;
const float MZone::eLeakPC=-60;
const float MZone::eBCPC=-80;
const float MZone::eSCPC=-80;
const float MZone::threshMaxPC=-48;
const float MZone::threshBasePC=-60;
const float MZone::threshDecayTPC=5;
const float MZone::threshDecayPC=1-exp(-TIMESTEP/MZone::threshDecayTPC);
const float MZone::gLeakPC=0.2/(6-TIMESTEP);
const float MZone::gPFDecayTPC=4.15;
const float MZone::gPFDecayPC=exp(-TIMESTEP/MZone::gPFDecayTPC);
const float MZone::gBCDecayTPC=5;
const float MZone::gBCDecayPC=exp(-TIMESTEP/MZone::gBCDecayTPC);
const float MZone::gSCDecayTPC=4.15;
const float MZone::gSCDecayPC=exp(-TIMESTEP/MZone::gSCDecayTPC);
const float MZone::gSCIncConstPC=0.00005;//0.0025;
const float MZone::gPFScaleConstPC=0.0000105;//0.0000185;//0.000065;
const float MZone::gBCScaleConstPC=0.0007;//0.0015;//0.037;

//IO cell const values
const float MZone::coupleScaleIO=0.33;//0.17;//0.07;//0.2;//0.04;
const float MZone::eLeakIO=-60;
const float MZone::eNCIO=-80;
const float MZone::gLeakIO=0.03;
const float MZone::gNCDecTSIO=0.5;
const float MZone::gNCDecTTIO=70;
const float MZone::gNCDecT0IO=0.56;
const float MZone::gNCIncScaleIO=0.003;
const float MZone::gNCIncTIO=300;
const float MZone::threshBaseIO=-61;
const float MZone::threshMaxIO=10;
const float MZone::threshTauIO=122;
const float MZone::threshDecayIO=1-exp(-TIMESTEP/MZone::threshTauIO);
// Quick Learning Plasticity Parameters
// Granule-Purkinje synapse long term potentiation/depression step sizes
float MZone::pfPCLTPIncPF=0.001;//0.0001;//0.001;
float MZone::pfPCLTDDecPF=-0.009;//-0.0009;//-0.01;
// Realistic Plasticity Parameters
// float MZone::pfPCLTPIncPF=.0001;
// float MZone::pfPCLTDDecPF=-0.0009;

//nucleus cell const values
const float MZone::eLeakNC=-65;
const float MZone::ePCNC=-80;
const float MZone::mfNMDADecayTNC=50;
const float MZone::mfNMDADecayNC=exp(-TIMESTEP/MZone::mfNMDADecayTNC);
const float MZone::mfAMPADecayTNC=6;
const float MZone::mfAMPADecayNC=exp(-TIMESTEP/MZone::mfAMPADecayTNC);
const float MZone::gMFNMDAIncNC=1-exp(-TIMESTEP/3.0);
const float MZone::gMFAMPAIncNC=1-exp(-TIMESTEP/3.0);
const float MZone::gPCScaleAvgNC=0.15;//4;//0.177;
const float MZone::gPCDecayTNC=4.15;
const float MZone::gPCDecayNC=exp(-TIMESTEP/MZone::gPCDecayTNC);
const float MZone::gLeakNC=0.02;
const float MZone::threshDecayTNC=5;
const float MZone::threshDecayNC=1-exp(-TIMESTEP/MZone::threshDecayTNC);
const float MZone::threshMaxNC=-40;
const float MZone::threshBaseNC=-72;
const float MZone::outIORelPDecTSNC=40;
const float MZone::outIORelPDecTTNC=1;
const float MZone::outIORelPDecT0NC=78;
const float MZone::outIORelPIncScaleNC=0.25;
const float MZone::outIORelPIncTNC=0.8;
const float MZone::mfSynWInitNC=0.005;
const float MZone::mfNCLTDThresh=13;//14;//18;//15;//14;
const float MZone::mfNCLTPThresh=1.7;//2;//6;//4;//2;

// matthew - changed these to be non const
// These are the plasticity parameters for the mossy fiber -> nucleus cell synapse.
// They are the step sizes for LTD/LTP connections eg excitatory inhibitory.
float MZone::mfNCLTDDecNC=-0.0000025; // Plasticity
float MZone::mfNCLTPIncNC=0.0003;//0.003;     // Plasticity


MZone::MZone(const bool *actSCIn, const bool *actMFIn, const bool *hMFIn,
		const unsigned int *pfBCSumIn, unsigned int *actBufGRGPU, unsigned int *delayMaskGRGPU, unsigned long *histGRGPU)
{
	apSCInput=actSCIn;
	apMFInput=actMFIn;
	histMFInput=hMFIn;

	//initialize variables
	//basket cell variables
	for(int i=0; i<numBC; i++)
	{
		inputPCBC[i]=0;
		gPFBC[i]=0;
		gPCBC[i]=0;
		threshBC[i]=threshBaseBC;
		vBC[i]=eLeakBC;
		apBC[i]=false;
		apBufBC[i]=0;
	}

	//purkinje cell variables
	for(int i=0; i<NUMGR; i++)
	{
		pfSynWeightPCH[i]=pfSynWInitPC;
	}
	for(int i=0; i<numPC; i++)
	{
		inputBCPC[i]=0;
		for(int j=0; j<numSCInPerPC; j++)
		{
			inputSCPC[i][j]=0;
			gSCPC[i][j]=0;
		}
		gPFPC[i]=0;
		gBCPC[i]=0;
		vPC[i]=eLeakPC;
		threshPC[i]=threshBasePC;
		apPC[i]=false;
		apBufPC[i]=0;
	}
	for(int i=0; i<numHistBinsPC; i++)
	{
		histAllAPPC[i]=0;
	}
	histSumAllAPPC=0;
	histBinNPC=0;
	allAPPC=0;

	//IO cell variables
	for(int i=0; i<NUMIO; i++)
	{
		threshIO[i]=threshBaseIO;
		vIO[i]=eLeakIO;
		vCoupIO[i]=0;
		apIO[i]=false;
		apBufIO[i]=0;
		for(int j=0; j<numNCInPerIO; j++)
		{
			inputNCIO[i][j]=false;
			gNCIO[i][j]=0;
		}
		pfPlastTimerIO[i]=0;
	}
	errDrive=0;

	//nucleus cell variables
	for(int i=0; i<numNC; i++)
	{
		for(int j=0; j<numPCInPerNC; j++)
		{
			inputPCNC[i][j]=false;
			gPCNC[i][j]=0;
			gPCScaleNC[i][j]=gPCScaleAvgNC;//*(1+(randGen->Random()-0.5)*gPCScaleAvgNC);
		}
		for(int j=0; j<numMFInPerNC; j++)
		{
			inputMFNC[i][j]=false;
			mfSynWNC[i][j]=mfSynWInitNC;
			mfNMDANC[i][j]=0;
			mfAMPANC[i][j]=0;
			gMFNMDANC[i][j]=0;
			gMFAMPANC[i][j]=0;
		}

		threshNC[i]=threshBaseNC;
		vNC[i]=eLeakNC;
		apNC[i]=false;
		apBufNC[i]=0;
		synIOPReleaseNC[i]=0;
	}
	noLTPMFNC=false;
	noLTDMFNC=false;

	//generate connectivities
	assignBCOutPCCon();
	assignPCOutNCCon();
	assignIOCoupleCon();

	initCUDA(pfBCSumIn, actBufGRGPU, delayMaskGRGPU, histGRGPU);
}

MZone::MZone(ifstream &infile, const bool *actSCIn, const bool *actMFIn, const bool *hMFIn,
		const unsigned int *pfBCSumIn, unsigned int *actBufGRGPU, unsigned int *delayMaskGRGPU, unsigned long *histGRGPU)
{
	apMFInput=actMFIn;
	histMFInput=hMFIn;
	apSCInput=actSCIn;

	infile.read((char *)inputPCBC, numBC*sizeof(unsigned char));
	infile.read((char *)gPFBC, numBC*sizeof(float));
	infile.read((char *)gPCBC, numBC*sizeof(float));
	infile.read((char *)threshBC, numBC*sizeof(float));
	infile.read((char *)vBC, numBC*sizeof(float));
	infile.read((char *)apBC, numBC*sizeof(bool));
	infile.read((char *)apBufBC, numBC*sizeof(unsigned int));
	infile.read((char *)bcConBCOutPC, numBC*numPCOutPerBC*sizeof(unsigned char));

	infile.read((char *)pfSynWeightPCH, numGR*sizeof(float));
	infile.read((char *)inputBCPC, numPC*sizeof(unsigned char));
	infile.read((char *)inputSCPC, numPC*numSCInPerPC*sizeof(bool));
	infile.read((char *)gPFPC, numPC*sizeof(float));
	infile.read((char *)gBCPC, numPC*sizeof(float));
	infile.read((char *)gSCPC, numPC*numSCInPerPC*sizeof(float));
	infile.read((char *)vPC, numPC*sizeof(float));
	infile.read((char *)threshPC, numPC*sizeof(float));
	infile.read((char *)apPC, numPC*sizeof(bool));
	infile.read((char *)apBufPC, numPC*sizeof(unsigned int));
	infile.read((char *)pcConPCOutNC, numPC*numNCOutPerPC*sizeof(unsigned int));
	infile.read((char *)histAllAPPC, numHistBinsPC*sizeof(unsigned short));
	infile.read((char *)&histSumAllAPPC, sizeof(unsigned short));
	infile.read((char *)&histBinNPC, sizeof(unsigned char));
	infile.read((char *)&allAPPC, sizeof(short));

	infile.read((char *)&errDrive, sizeof(float));
	infile.read((char *)inputNCIO, numIO*numNCInPerIO*sizeof(bool));
	infile.read((char *)gNCIO, numIO*numNCInPerIO*sizeof(float));
	infile.read((char *)threshIO, numIO*sizeof(float));
	infile.read((char *)vIO, numIO*sizeof(float));
	infile.read((char *)vCoupIO, numIO*sizeof(float));
	infile.read((char *)apIO, numIO*sizeof(bool));
	infile.read((char *)apBufIO, numIO*sizeof(unsigned int));
	infile.read((char *)conIOCouple, numIO*numIOCoupInPerIO*sizeof(unsigned char));
	infile.read((char *)pfPlastTimerIO, numIO*sizeof(int));

	infile.read((char *)inputPCNC, numNC*numPCInPerNC*sizeof(bool));
	infile.read((char *)gPCNC, numNC*numPCInPerNC*sizeof(float));
	infile.read((char *)gPCScaleNC, numNC*numPCInPerNC*sizeof(float));
	infile.read((char *)inputMFNC, numNC*numMFInPerNC*sizeof(bool));
	infile.read((char *)mfSynWNC, numNC*numMFInPerNC*sizeof(float));
	infile.read((char *)mfNMDANC, numNC*numMFInPerNC*sizeof(float));
	infile.read((char *)mfAMPANC, numNC*numMFInPerNC*sizeof(float));
	infile.read((char *)gMFNMDANC, numNC*numMFInPerNC*sizeof(float));
	infile.read((char *)gMFAMPANC, numNC*numMFInPerNC*sizeof(float));
	infile.read((char *)threshNC, numNC*sizeof(float));
	infile.read((char *)vNC, numNC*sizeof(float));
	infile.read((char *)apNC, numNC*sizeof(bool));
	infile.read((char *)apBufNC, numNC*sizeof(unsigned int));
	infile.read((char *)synIOPReleaseNC, numNC*sizeof(float));
	infile.read((char *)&noLTPMFNC, sizeof(bool));
	infile.read((char *)&noLTDMFNC, sizeof(bool));

	cout<<"mzone read: "<<vBC[0]<<" "<<vBC[numBC-1]<<" "<<vPC[0]<<" "<<vPC[numPC-1]
			<<" "<<vIO[0]<<" "<<vIO[numIO-1]<<" "<<vNC[0]<<" "<<vNC[numNC-1]
			<<" "<<pfSynWeightPCH[0]<<" "<<pfSynWeightPCH[numGR-1]<<" "
			<<mfSynWNC[0][0]<<" "<<mfSynWNC[numNC-1][numMFInPerNC-1]<<endl;

	initCUDA(pfBCSumIn, actBufGRGPU, delayMaskGRGPU, histGRGPU);
}

void MZone::exportState(ofstream &outfile)
{
	outfile.write((char *)inputPCBC, numBC*sizeof(unsigned char));
	outfile.write((char *)gPFBC, numBC*sizeof(float));
	outfile.write((char *)gPCBC, numBC*sizeof(float));
	outfile.write((char *)threshBC, numBC*sizeof(float));
	outfile.write((char *)vBC, numBC*sizeof(float));
	outfile.write((char *)apBC, numBC*sizeof(bool));
	outfile.write((char *)apBufBC, numBC*sizeof(unsigned int));
	outfile.write((char *)bcConBCOutPC, numBC*numPCOutPerBC*sizeof(unsigned char));

//	cpyPFPCSynW();
	outfile.write((char *)pfSynWeightPCH, numGR*sizeof(float));
	outfile.write((char *)inputBCPC, numPC*sizeof(unsigned char));
	outfile.write((char *)inputSCPC, numPC*numSCInPerPC*sizeof(bool));
	outfile.write((char *)gPFPC, numPC*sizeof(float));
	outfile.write((char *)gBCPC, numPC*sizeof(float));
	outfile.write((char *)gSCPC, numPC*numSCInPerPC*sizeof(float));
	outfile.write((char *)vPC, numPC*sizeof(float));
	outfile.write((char *)threshPC, numPC*sizeof(float));
	outfile.write((char *)apPC, numPC*sizeof(bool));
	outfile.write((char *)apBufPC, numPC*sizeof(unsigned int));
	outfile.write((char *)pcConPCOutNC, numPC*numNCOutPerPC*sizeof(unsigned int));
	outfile.write((char *)histAllAPPC, numHistBinsPC*sizeof(unsigned short));
	outfile.write((char *)&histSumAllAPPC, sizeof(unsigned short));
	outfile.write((char *)&histBinNPC, sizeof(unsigned char));
	outfile.write((char *)&allAPPC, sizeof(short));

	outfile.write((char *)&errDrive, sizeof(float));
	outfile.write((char *)inputNCIO, numIO*numNCInPerIO*sizeof(bool));
	outfile.write((char *)gNCIO, numIO*numNCInPerIO*sizeof(float));
	outfile.write((char *)threshIO, numIO*sizeof(float));
	outfile.write((char *)vIO, numIO*sizeof(float));
	outfile.write((char *)vCoupIO, numIO*sizeof(float));
	outfile.write((char *)apIO, numIO*sizeof(bool));
	outfile.write((char *)apBufIO, numIO*sizeof(unsigned int));
	outfile.write((char *)conIOCouple, numIO*numIOCoupInPerIO*sizeof(unsigned char));
	outfile.write((char *)pfPlastTimerIO, numIO*sizeof(int));

	outfile.write((char *)inputPCNC, numNC*numPCInPerNC*sizeof(bool));
	outfile.write((char *)gPCNC, numNC*numPCInPerNC*sizeof(float));
	outfile.write((char *)gPCScaleNC, numNC*numPCInPerNC*sizeof(float));
	outfile.write((char *)inputMFNC, numNC*numMFInPerNC*sizeof(bool));
	outfile.write((char *)mfSynWNC, numNC*numMFInPerNC*sizeof(float));
	outfile.write((char *)mfNMDANC, numNC*numMFInPerNC*sizeof(float));
	outfile.write((char *)mfAMPANC, numNC*numMFInPerNC*sizeof(float));
	outfile.write((char *)gMFNMDANC, numNC*numMFInPerNC*sizeof(float));
	outfile.write((char *)gMFAMPANC, numNC*numMFInPerNC*sizeof(float));
	outfile.write((char *)threshNC, numNC*sizeof(float));
	outfile.write((char *)vNC, numNC*sizeof(float));
	outfile.write((char *)apNC, numNC*sizeof(bool));
	outfile.write((char *)apBufNC, numNC*sizeof(unsigned int));
	outfile.write((char *)synIOPReleaseNC, numNC*sizeof(float));
	outfile.write((char *)&noLTPMFNC, sizeof(bool));
	outfile.write((char *)&noLTDMFNC, sizeof(bool));
}

void MZone::assignBCOutPCCon()
{
	for(int i=0; i<numPC; i++)
	{
		bcConBCOutPC[i*bcToPCRatio][0]=((i+1)%numPC+numPC)%numPC;
		bcConBCOutPC[i*bcToPCRatio][1]=((i-1)%numPC+numPC)%numPC;
		bcConBCOutPC[i*bcToPCRatio][2]=((i+2)%numPC+numPC)%numPC;
		bcConBCOutPC[i*bcToPCRatio][3]=((i-2)%numPC+numPC)%numPC;

		bcConBCOutPC[i*bcToPCRatio+1][0]=((i+1)%numPC+numPC)%numPC;
		bcConBCOutPC[i*bcToPCRatio+1][1]=((i-1)%numPC+numPC)%numPC;
		bcConBCOutPC[i*bcToPCRatio+1][2]=((i+3)%numPC+numPC)%numPC;
		bcConBCOutPC[i*bcToPCRatio+1][3]=((i-3)%numPC+numPC)%numPC;

		bcConBCOutPC[i*bcToPCRatio+2][0]=((i+3)%numPC+numPC)%numPC;
		bcConBCOutPC[i*bcToPCRatio+2][1]=((i-3)%numPC+numPC)%numPC;
		bcConBCOutPC[i*bcToPCRatio+2][2]=((i+6)%numPC+numPC)%numPC;
		bcConBCOutPC[i*bcToPCRatio+2][3]=((i-6)%numPC+numPC)%numPC;

		bcConBCOutPC[i*bcToPCRatio+3][0]=((i+4)%numPC+numPC)%numPC;
		bcConBCOutPC[i*bcToPCRatio+3][1]=((i-4)%numPC+numPC)%numPC;
		bcConBCOutPC[i*bcToPCRatio+3][2]=((i+9)%numPC+numPC)%numPC;
		bcConBCOutPC[i*bcToPCRatio+3][3]=((i-9)%numPC+numPC)%numPC;
	}
}

void MZone::assignPCOutNCCon()
{
	char numSynPCPerNC[numNC];
//	char ncConPC[numPC][numNCOutPerPC];
	memset(numSynPCPerNC, 0, numNC*sizeof(char));
//	memset(ncConPC, -1, numPC*numNCOutPerPC*sizeof(char));

	//assign PC to NC connections
	for(int i=0; i<numPC; i++)
	{
		pcConPCOutNC[i][0]=(unsigned int)(i/(numPCInPerNC/numNCOutPerPC))*numPCInPerNC+(unsigned int)(i%(numPCInPerNC/numNCOutPerPC));
//		ncConPC[i][0]=(char)(i/(numPCInPerNC/numNCOutPerPC));
		for(int j=1; j<numNCOutPerPC; j++)
		{
			while(true)
			{
				int indSynNC;
//				bool ncPrevConned;

//				ncPrevConned=false;
				indSynNC=randGen->Random()*numNC*numPCInPerNC;

				if((indSynNC%numPCInPerNC)<(numPCInPerNC/numNCOutPerPC))
				{
					continue;
				}
//				for(int k=0; k<j; k++)
//				{
//					if(indSynNC/numPCInPerNC==ncConPC[i][k])
//					{
//						ncPrevConned=true;
//						break;
//					}
//				}
//				if(ncPrevConned)
//				{
//					continue;
//				}
				if(numSynPCPerNC[indSynNC/numPCInPerNC]<(numPCInPerNC-(numPCInPerNC/numNCOutPerPC)))
				{
					numSynPCPerNC[indSynNC/numPCInPerNC]++;
//					ncConPC[i][j]=indSynNC/numPCInPerNC;
					pcConPCOutNC[i][j]=indSynNC;
					break;
				}
			}
		}
	}
//	pcConPCOutNC[numPC-1][0]=(unsigned int)((numPC-1)/(numPCInPerNC/numNCOutPerPC))*numPCInPerNC+(unsigned int)((numPC-1)%(numPCInPerNC/numNCOutPerPC));
//	for(int i=0; i<numNC; i++)
//	{
//
//	}

}

void MZone::assignIOCoupleCon()
{
	for(int i=0; i<numIO; i++)
	{
		unsigned char tempInInd;

		tempInInd=0;
		for(int j=0; j<numIOCoupInPerIO; j++)
		{
			if(tempInInd==i)
			{
				tempInInd++;
			}
			conIOCouple[i][j]=tempInInd;//((i+1)%numIO+numIO)%numIO;
			tempInInd++;
		}
	}
}

void MZone::initCUDA(const unsigned int *pfBCSumIn, unsigned int *actBufGRGPU, unsigned int *delayMaskGRGPU, unsigned long *histGRGPU)
{
	sumPFBCInput=pfBCSumIn;
	apBufGRGPU=actBufGRGPU;
	delayBCPCSCMaskGRGPU=delayMaskGRGPU;
	historyGRGPU=histGRGPU;
	//allocate host cuda memory
	cudaMallocHost((void **)&inputSumPFPCMZH, numPC*sizeof(float));

	//allocate device cuda memory
	cudaMalloc((void **)&pfSynWeightPCGPU, numGR*sizeof(float));
	cudaMallocPitch((void **)&inputPFPCGPU, (size_t *)&inputPFPCGPUPitch,
			numPFInPerPC*sizeof(float), numPC);
	cudaMalloc((void **)&inputSumPFPCMZGPU, numPC*sizeof(float));

	//initialize host cuda memory
	for(int i=0; i<numPC; i++)
	{
		inputSumPFPCMZH[i]=0;
	}
	//initialize device cuda memory
	cudaMemcpy(pfSynWeightPCGPU, pfSynWeightPCH, numGR*sizeof(float), cudaMemcpyHostToDevice);
}

void MZone::cpyPFPCSynWCUDA()
{
	cudaMemcpy(pfSynWeightPCH, pfSynWeightPCGPU, numGR*sizeof(float), cudaMemcpyDeviceToHost);
}

MZone::~MZone()
{
	//free cuda host memory
	cudaFreeHost(inputSumPFPCMZH);

	//free cuda device memory
	cudaFree(pfSynWeightPCGPU);
	cudaFree(inputPFPCGPU);
	cudaFree(inputSumPFPCMZGPU);

}

void MZone::calcPCActivities()
{
//	float temp;
	for(int i=0; i<numPC; i++)
	{
		float gSCPCSum;

//		cout<<inputSumPFPCMZH[i]<<" ";
		gPFPC[i]=gPFPC[i]+inputSumPFPCMZH[i]*gPFScaleConstPC;
		gPFPC[i]=gPFPC[i]*gPFDecayPC;//gPFScaleConstPC;//GPFSCALECONSTPC; //? TODO: is that the decay?
		gBCPC[i]=gBCPC[i]+inputBCPC[i]*gBCScaleConstPC;//GBCSCALECONSTPC;
		gBCPC[i]=gBCPC[i]*gBCDecayPC;//gBCScaleConstPC;//GBCSCALECONSTPC; //? TODO: where's the decay?

		gSCPCSum=0;
		//TODO: refactor SC input to PC to make it consistent with everything else
		for(int j=0; j<numSCInPerPC; j++)
		{
			gSCPC[i][j]=gSCPC[i][j]+gSCIncConstPC*(1-gSCPC[i][j])*inputSCPC[i][j];//apSC[i*SCPCSYNPERPC+j];
			gSCPC[i][j]=gSCPC[i][j]*gSCDecayPC;//GSCDECAYPC;
			gSCPCSum+=gSCPC[i][j];
		}
//		temp=gSCPCSum;

		vPC[i]=vPC[i]+(gLeakPC*(eLeakPC-vPC[i]))-(gPFPC[i]*vPC[i])+(gBCPC[i]*(eBCPC-vPC[i]))+(gSCPCSum*(eSCPC-vPC[i]));
		threshPC[i]=threshPC[i]+(threshDecayPC*(threshBasePC-threshPC[i]));

		apPC[i]=vPC[i]>threshPC[i];
		apBufPC[i]=(apBufPC[i]<<1)|(apPC[i]*0x00000001);

		threshPC[i]=apPC[i]*threshMaxPC+(!apPC[i])*threshPC[i];
		allAPPC=allAPPC+apPC[i];

	}
//	cout<<endl;

//	cout<<temp<<" "<<gPFPC[31]<<" "<<inputSumPFPCH[31]<<" "<<gBCPC[31]<<" "<<(int)(inputBCPC[31])<<" "<<apPC[31]<<endl;

	memset(inputBCPC, 0, numPC*sizeof(unsigned char));
}

void MZone::calcBCActivities()
{
	for(int i=0; i<numBC; i++)
	{
		gPFBC[i]=gPFBC[i]+(sumPFBCInput[i]*pfIncConstBC);
		gPFBC[i]=gPFBC[i]*gPFDecayBC;
		gPCBC[i]=gPCBC[i]+(inputPCBC[i]*pcIncConstBC);
		gPCBC[i]=gPCBC[i]*gPCDecayBC;

		vBC[i]=vBC[i]+(gLeakBC*(eLeakBC-vBC[i]))-(gPFBC[i]*vBC[i])+(gPCBC[i]*(ePCBC-vBC[i]));
		threshBC[i]=threshBC[i]+threshDecayBC*(threshBaseBC-threshBC[i]);
		apBC[i]=vBC[i]>threshBC[i];
		apBufBC[i]=(apBufBC[i]<<1)|(apBC[i]*0x00000001);

		threshBC[i]=apBC[i]*threshMaxBC+(!apBC[i])*(threshBC[i]);
	}
//	cout<<gPFBC[127]<<" "<<inputSumPFBCH[127]<<" "<<gPCBC[127]<<" "<<inputPCBC[127]<<vBC[127]<<apBC[127]<<endl;
//	cout<<"diagPFBCH: "<<pfbcsumdiag<<" diagPFBCGPU: "<<inputSumPFBCH[0]<<endl;

	memset(inputPCBC, 0, numBC*sizeof(unsigned char));
//	memset(sumPFBCInput, 0, numBC*sizeof(unsigned short));
}

void MZone::calcIOActivities()
{
	for(int i=0; i<numIO; i++)
	{
		float gNCSum;
//		float gHMax;
//		float gHTau;
//		float gLtCaHMax;
//		float gLtCaM;

//		cout<<threshIO[i]<<" ";
		//calculate DCN input conductance
		gNCSum=0;
		for(int j=0; j<numNCInPerIO; j++)
		{
			gNCIO[i][j]=gNCIO[i][j]*exp(-TIMESTEP/(-gNCDecTSIO*exp(-gNCIO[i][j]/gNCDecTTIO)+gNCDecT0IO));
			gNCIO[i][j]=gNCIO[i][j]+inputNCIO[i][j]*gNCIncScaleIO*exp(-gNCIO[i][j]/gNCIncTIO);
			gNCSum=gNCSum+gNCIO[i][j];
		}

		gNCSum=gNCSum/3.1;

		vIO[i]=vIO[i]+gLeakIO*(eLeakIO-vIO[i])+gNCSum*1.5*(eNCIO-vIO[i])+errDrive;

		apIO[i]=vIO[i]>threshIO[i];
		apBufIO[i]=(apBufIO[i]<<1)|(apIO[i]*0x00000001);

		threshIO[i]=threshMaxIO*apIO[i]+(!apIO[i])*(threshIO[i]+threshDecayIO*(threshBaseIO-threshIO[i]));
//		cout<<vIO[i]<<" "<<threshIO[i]<<" "<<gNCSum<<" "<<apIO[i]<<" "<<threshDecayIO<<endl;

//		//calculate gH
//		gHMax=1/(1+exp((vIO[i]+GHMAXVIO)/8));
//		gHTau=exp(0.033*(vIO[i]+GHTAUVIO))/(0.011*(1+exp(0.083*(vIO[i]+GHTAUVIO))));
//		gHIO[i]=gHIO[i]+(gHMax-gHIO[i])/gHTau;
//
//		//calculate gLtCa
//		gLtCaHMax=1/(1+exp((vIO[i]+GLTCAHMAXVIO)/8.6));
//		gLtCaHIO[i]=gLtCaHIO[i]+(gLtCaHMax-gLtCaHIO[i])/GLTCAHTIO;
//		gLtCaM=1/powf(1+exp((GLTCAMMAXVIO-vIO[i])/4.2), 3);
//		gLtCaIO[i]=gLtCaM*gLtCaHIO[i]*4;
//
//		//calculate [Ca]
//		caIO[i]=caIO[i]+(gLtCaIO[i]>0)*(1-caIO[i])*gLtCaIO[i]*GLTCAHTIO;
//		caIO[i]=caIO[i]*CADECAYIO;
//
//		//calculate gKCa
//		gKCaIO[i]=gKCaIO[i]+(1-gKCaIO[i])*(caIO[i]-0.2)*0.05;
//		gKCaIO[i]=(!(gKCaIO[i]<0))*gKCaIO[i];
//
//		//calculate vm
//		vIO[i]=vIO[i]+GLEAKIO*(ELEAKIO-vIO[i])+
//			gHIO[i]*0.04*(EHIO-vIO[i])+
//			gNCSum*1.5*(ENCIO-vIO[i])+
//			gKCaIO[i]*0.1*(EKCAIO-vIO[i])+
//			gLtCaIO[i]*(ECAIO-vIO[i]);//+
////			MAXUSDR*(t==USONSET);
//
////		cout<<vIO[i]<<endl;
//		//calculate ap, thresh and [Ca]
//		apIO[i]=vIO[i]>threshIO[i];
//		threshIO[i]=-20*apIO[i]+(!apIO[i])*(threshIO[i]+(0.2*(THRESHBASEIO-threshIO[i])));
//		caIO[i]=caIO[i]+apIO[i]*(1-caIO[i])*0.2;
	}
//	cout<<endl;

	memset(inputNCIO, false, numIO*numNCInPerIO*sizeof(bool));
}

void MZone::calcNCActivities()
{
	for(int i=0; i<numNC; i++)
	{
		float gMFNMDASum;
		float gMFAMPASum;
		float gPCNCSum;

		gMFNMDASum=0;
		gMFAMPASum=0;
		for(int j=0; j<numMFInPerNC; j++)
		{
			mfNMDANC[i][j]=mfNMDANC[i][j]*mfNMDADecayNC+inputMFNC[i][j]*mfSynWNC[i][j]*(1-mfNMDANC[i][j]);
			gMFNMDANC[i][j]=gMFNMDANC[i][j]+gMFNMDAIncNC*(mfNMDANC[i][j]-gMFNMDANC[i][j]);
			gMFNMDASum=gMFNMDASum+gMFNMDANC[i][j];

			mfAMPANC[i][j]=mfAMPANC[i][j]*mfAMPADecayNC+inputMFNC[i][j]*mfSynWNC[i][j]*(1-mfAMPANC[i][j]);
			gMFAMPANC[i][j]=gMFAMPANC[i][j]+gMFAMPAIncNC*(mfAMPANC[i][j]-gMFAMPANC[i][j]);
			gMFAMPASum=gMFAMPASum+gMFAMPANC[i][j];
		}
		gMFNMDASum=gMFNMDASum*TIMESTEP/((float)numMFInPerNC);
		gMFAMPASum=gMFAMPASum*TIMESTEP/((float)numMFInPerNC);
		gMFNMDASum=gMFNMDASum*vNC[i]/(-80.0f);

		gPCNCSum=0;
		for(int j=0; j<numPCInPerNC; j++)
		{
			gPCNC[i][j]=gPCNC[i][j]*gPCDecayNC+inputPCNC[i][j]*gPCScaleNC[i][j]*(1-gPCNC[i][j]);
			gPCNCSum=gPCNCSum+gPCNC[i][j];
		}
		gPCNCSum=gPCNCSum*TIMESTEP/((float)numPCInPerNC);

//		cout<<inputMFNC[i][0]<<" "<<mfSynWeightNC[i][0]<<" "<<mfNMDANC[i][0]<<" "<<inputMFNC[i][0]*mfSynWeightNC[i][0]*(1-mfNMDANC[i][0])<<" "<<gMFNMDANC[i][0]<<" "<<mfAMPANC[i][0]<<" "<<gMFAMPANC[i][0]<<endl;

		vNC[i]=vNC[i]+gLeakNC*(eLeakNC-vNC[i])-(gMFNMDASum+gMFAMPASum)*vNC[i]+gPCNCSum*(ePCNC-vNC[i]);

		threshNC[i]=threshNC[i]+threshDecayNC*(threshBaseNC-threshNC[i]);
		apNC[i]=vNC[i]>threshNC[i];
		apBufNC[i]=(apBufNC[i]<<1)|(apNC[i]*0x00000001);

		threshNC[i]=apNC[i]*threshMaxNC+(!apNC[i])*threshNC[i];
	}
//	cout<<"-----------"<<endl;

	memset(inputMFNC, false, numNC*numMFInPerNC*sizeof(bool));
	memset(inputPCNC, false, numNC*numPCInPerNC*sizeof(bool));
}

void MZone::updatePCOut()
{
	for(int i=0; i<numPC; i++)
	{
		for(int j=0; j<numBCOutPerPC; j++)
		{
			int indBC;

			indBC=i*bcToPCRatio-6+j;

			indBC=(indBC%numBC+numBC)%numBC;
			inputPCBC[indBC]=inputPCBC[indBC]+apPC[i];
		}

		for(int j=0; j<numNCOutPerPC; j++)
		{
			inputPCNC[pcConPCOutNC[i][j]/numPCInPerNC][pcConPCOutNC[i][j]%numPCInPerNC]=apPC[i];
		}
	}
}

void MZone::updateBCPCOut()
{
	for(int i=0; i<numBC; i++)
	{
		if(apBC[i])
		{
			for(int j=0; j<numPCOutPerBC; j++)
			{
				inputBCPC[bcConBCOutPC[i][j]]++;
			}
		}
	}
}

void MZone::updateSCPCOut()
{
	for(int i=0; i<numSC; i++)
	{
		inputSCPC[i/numSCInPerPC][i%numSCInPerPC]=apSCInput[i];
	}
}

void MZone::updateIOOut()
{
	for(int i=0; i<numIO; i++)
	{
		pfPlastTimerIO[i]=(!apIO[i])*(pfPlastTimerIO[i]+1)+apIO[i]*pfLTDTimerStartIO;
	}
}

void MZone::updateIOCouple()
{
	for(int i=0; i<numIO; i++)
	{
		vCoupIO[i]=0;
		for(int j=0; j<numIOCoupInPerIO; j++)
		{
			vCoupIO[i]=vCoupIO[i]+coupleScaleIO*(vIO[conIOCouple[i][j]]-vIO[i]);
		}
	}
	for(int i=0; i<numIO; i++)
	{
		vIO[i]=vIO[i]+vCoupIO[i];
	}
}

void MZone::updateNCOut()
{
	for(int i=0; i<NUMNC; i++)
	{
		synIOPReleaseNC[i]=synIOPReleaseNC[i]*exp(-TIMESTEP/(outIORelPDecTSNC*exp(-synIOPReleaseNC[i]/outIORelPDecTTNC)+outIORelPDecT0NC));
		synIOPReleaseNC[i]=synIOPReleaseNC[i]+apNC[i]*outIORelPIncScaleNC*exp(-synIOPReleaseNC[i]/outIORelPIncTNC);
	}


	for(int i=0; i<NUMIO; i++)
	{
		for(int j=0; j<NUMNCINPERIO; j++)
		{
			inputNCIO[i][j]=(randGen->Random()<synIOPReleaseNC[j]);
		}
	}
}

void MZone::updateMFNCOut()
{
//	for(int i=0; i<numNC; i++)
//	{
//		for(int j=0; j<numMFInPerNC; j++)
//		{
//			inputMFNC[i][j]=apMFInput[j];
//		}
//	}
	for(int i=0; i<numMF; i++)
	{
		inputMFNC[i/numMFInPerNC][i%numMFInPerNC]=apMFInput[i];
	}
}

void MZone::updateMFNCSyn(short t)
{
	bool reset;
	float avgAllAPPC;
	bool doLTD;
	bool doLTP;

	reset=(t%histBinWidthPC==0);
	if(!reset)
	{
		return;
	}
	histSumAllAPPC=histSumAllAPPC-histAllAPPC[histBinNPC]+allAPPC;
	histAllAPPC[histBinNPC]=allAPPC;
	allAPPC=0;
	histBinNPC++;
	histBinNPC=histBinNPC%numHistBinsPC;

	avgAllAPPC=((float)histSumAllAPPC)/numHistBinsPC;

//	cout<<"avgAllAPPC: "<<avgAllAPPC<<endl;

	doLTD=false;
	doLTP=false;
	if(avgAllAPPC>=mfNCLTDThresh && !noLTDMFNC)
	{
		doLTD=true;
		noLTDMFNC=true;
	}
	else if(avgAllAPPC<mfNCLTDThresh)
	{
		noLTDMFNC=false;
	}

	if(avgAllAPPC<=mfNCLTPThresh && !noLTPMFNC)
	{
		doLTP=true;
		noLTPMFNC=true;
	}
	else if(avgAllAPPC>mfNCLTPThresh)
	{
		noLTPMFNC=false;
	}

//	cout<<"MFNC plasticity: "<<avgAllAPPC<<" "<<doLTP<<" "<<doLTD<<endl;
	for(int i=0; i<NUMNC; i++)
	{
		for(int j=0; j<numMFInPerNC; j++)
		{
			float synWDelta;
			synWDelta=histMFInput[i*numMFInPerNC+j]*(doLTD*mfNCLTDDecNC+doLTP*mfNCLTPIncNC);
			mfSynWNC[i][j]=mfSynWNC[i][j]+synWDelta;
			mfSynWNC[i][j]=(mfSynWNC[i][j]>0)*mfSynWNC[i][j];
			mfSynWNC[i][j]=(mfSynWNC[i][j]<=1)*mfSynWNC[i][j]+(mfSynWNC[i][j]>1);
		}
//		cout<<endl<<mfSynWChangeNC[i][0]<<" "<<mfSynWeightNC[i][0]<<endl;
	}
}

void MZone::runPFPCOutCUDA(cudaStream_t &st)
{
	callUpdatePFPCOutKernel<numPFInPerPC, 1024, 1024>(st, apBufGRGPU, delayBCPCSCMaskGRGPU, pfSynWeightPCGPU, inputPFPCGPU, inputPFPCGPUPitch);
}

void MZone::runPFPCSumCUDA(cudaStream_t &st)
{
	callSumPFKernel<float, 512, true, false>
		(st, inputPFPCGPU, inputPFPCGPUPitch, inputSumPFPCMZGPU, 1, numPC, 1, numPFInPerPC);
}

void MZone::cpyPFPCSumCUDA(cudaStream_t &st)
{
	cudaMemcpyAsync(inputSumPFPCMZH, inputSumPFPCMZGPU, numPC*sizeof(float), cudaMemcpyDeviceToHost, st);
}

void MZone::runPFPCPlastCUDA(cudaStream_t *sts, short t)
{
	if(t%histBinWidthGR==0)
	{
		for(int i=0; i<numIO; i++)
		{
			callUpdatePFPCPlasticityIOKernel<256, 1024>(sts[i], pfPlastTimerIO[i], pfSynWeightPCGPU, historyGRGPU, i*numGR/numIO, pfPCLTDDecPF, pfPCLTPIncPF);
		}
	}
}

void MZone::exportActsPCBCDisp(SCBCPCActs &actSt)
{
	for(int i=0; i<numPC; i++)
	{
		actSt.apPC[i]=apPC[i];
		actSt.vPC[i]=vPC[i];
	}

	for(int i=0; i<numBC; i++)
	{
		actSt.apBC[i]=apBC[i];
	}
}

void MZone::exportActsIONCPCDisp(IONCPCActs &actSt)
{
	for(int i=0; i<numIO; i++)
	{
		actSt.apIO[i]=apIO[i];
		actSt.vIO[i]=vIO[i];
	}

	for(int i=0; i<numNC; i++)
	{
		actSt.apNC[i]=apNC[i];
		actSt.vNC[i]=vNC[i];
	}

	for(int i=0; i<numPC; i++)
	{
		actSt.apPC[i]=apPC[i];
		actSt.vPC[i]=vPC[i];
	}
}

void MZone::disablePlasticity()
{
    pfPCLTPIncPF = 0;
    pfPCLTDDecPF = 0;
    mfNCLTDDecNC = 0;
    mfNCLTPIncNC = 0;
}

void MZone::annealPlasticity(float decay)
{
    pfPCLTPIncPF *= decay;
    pfPCLTDDecPF *= decay;
}
