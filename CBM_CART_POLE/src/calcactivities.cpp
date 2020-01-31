/*
 * calcactivities.cpp
 *
 *  Created on: Feb 12, 2010
 *      Author: wen
 */

#include "../includes/calcactivities.h"

/**
 * Determines the MF Firings rates for a given cart-pole paramter.
 * This is based on normal distributions distributed evenly over a
 * range.
 */
void calcCartPoleMFRates(float mfRates[], int numMFs, float minVal, float maxVal, float currentVal) {
	float range = maxVal - minVal;
	float interval = range / numMFs;
	float pos = minVal + interval / 2.0;
	for (int i = 0; i < numMFs; i++) {
		float mean = pos;
		float x = currentVal;
		// This should give us reasonably overlapped gaussians
		float variance = interval / 2.0; // This may need tuning
		// Formula for normal distribution: http://en.wikipedia.org/wiki/Normal_distribution
		float value = exp(-1 * ((x-mean)*(x-mean))/(2*(variance*variance))) / sqrt(2 * M_PI * (variance*variance));
		mfRates[i] = value;
		pos += interval;
	}
}

void updateMFRates()
{
	float poleAngleMFRates[NUMPOLANGLEMF];
	float poleVelocityMFRates[NUMPOLVELMF];
	float cartPosMFRates[NUMCARTPOSMF];
	float cartVelocityMFRates[NUMCARTVELMF];

	calcCartPoleMFRates(poleAngleMFRates, NUMPOLANGLEMF, MIN_POLE_ANGLE, MAX_POLE_ANGLE, 0);
	calcCartPoleMFRates(poleVelocityMFRates, NUMPOLVELMF, MIN_POLE_VELOCITY, MAX_POLE_VELOCITY, 0);
	calcCartPoleMFRates(cartPosMFRates, NUMCARTPOSMF, MIN_CART_POS, MAX_CART_POS, 0);
	calcCartPoleMFRates(cartVelocityMFRates, NUMCARTVELMF, MIN_CART_VELOCITY, MAX_CART_VELOCITY, 0);
}

//functions to calculate the activity of each cell type
void calcMFActivity(short t, CRandomSFMT0 &randGen)
{
	static bool tempIncMF[NUMMF];
	for(int i=0; i<NUMMF; i++)
	{
		// Updates thresholds based on threshDecay = ~.22
		// thresMFs are initialized to 1. Basically this
		// increases threshMF with limit of 1.
		threshMF[i]=threshMF[i]+(1-threshMF[i])*threshDecayMF;
	}

//	for(int i=0; i<NUMMF; i++)
//	{
//		// Determines if we should temporarily increase this MF's activity.
//		// Some MFs respond to tone and other dont.
//		tempIncMF[i]=csOnMF[i] && t>=csStartMF[i] && t<csEndMF[i];
//	}

	for(int i=0; i<NUMMF; i++)
	{
		// Determine if MF[i] is active.
//		apMF[i]=randGen.Random()<((tempIncMF[i])*incFreqMF[i]+bgFreqContsMF[activeContext][i])*threshMF[i];
		apMF[i]=randGen.Random()<(incFreqMF[i]+bgFreqContsMF[activeContext][i])*threshMF[i];
		// Record if this MF is active.
		histMF[i]=histMF[i] || apMF[i];
	}

	for(int i=0; i<NUMMF; i++)
	{
		// If MF cell is active (apMF = true) then
		// we reset its threshold to 0. (It will then
		// slowly work its way back to 1.)
		threshMF[i]=(!apMF[i])*threshMF[i];
	}
}

void calcGRActivity(short t)
{
#ifdef GPUDEBUG
	float vGRCheck[NUMGR+NUMGRPAD];
	for(int j=0; j<1024; j++)
	{
		cout<<(int)inputsGRH[j]<<" ";
	}
	cout<<endl;
#endif

	cudaMemcpy(inputsGRGPU, inputsGRH, NUMGR*sizeof(unsigned char), cudaMemcpyHostToDevice);
	runGRKernels(t);
	cudaMemcpy(apOutGR, apOutGRGPU, NUMGR*sizeof(unsigned char), cudaMemcpyDeviceToHost);

	memset(inputsGRH, 0, NUMGR*sizeof(unsigned char));

#ifdef GPUDEBUG
	cudaMemcpy(vGRCheck, vGRGPU, NUMGR*sizeof(float), cudaMemcpyDeviceToHost);

	for(int i=0; i<2; i++)
	{
		cout<<vGRCheck[i]<<" ";
	}
	cout<<endl;
#endif
}
void calcGOActivity()
{
	for(int i=0; i<NUMGO; i++)
	{
		gMFGO[i]=inputMFGO[i]*gMFIncGO[i]+gMFGO[i]*gMFDecayGO;
		gGRGO[i]=inputGRGO[i]*gGRIncGO[i]+gGRGO[i]*gGRDecayGO;

		gluGO[i]=gluGO[i]*gluDecayGO+inputGRGO[i]*gluScaleGO*exp(-1.5*gluGO[i]);
		mGluRGO[i]=mGluRGO[i]*mGluRDecayGO+gluGO[i]*mGluRScaleGO*exp(-mGluRGO[i]);
		gMGluRIncGO[i]=gMGluRIncGO[i]*gMGluRDecayGO+mGluRGO[i]*gMGluRIncScaleGO*exp(-gMGluRIncGO[i]);
		gMGluRGO[i]=gMGluRGO[i]*gMGluRDecayGO+gMGluRIncGO[i]*gMGluRScaleGO;

		threshGO[i]=threshGO[i]+(threshBaseGO[i]-threshGO[i])*threshDecayGO;
		vGO[i]=vGO[i]+(gLeakGO*(ELEAKGO-vGO[i]))+(gMGluRGO[i]*(EMGLURGO-vGO[i]))-(gMFGO[i]+gGRGO[i])*vGO[i];

		apGO[i]=vGO[i]>threshGO[i];

		threshGO[i]=apGO[i]*threshMaxGO+(!apGO[i])*threshGO[i];
	}

	memset(inputMFGO, 0, NUMGO*sizeof(unsigned short));
	memset(inputGRGO, 0, NUMGO*sizeof(unsigned short));
}

void calcPCActivity()
{
	runPCKernels();
	cudaMemcpy(inputSumPFPC, inputSumPFPCGPU, NUMPC*sizeof(float), cudaMemcpyDeviceToHost);
	for(int i=0; i<NUMPC; i++)
	{
		float gSCPCSum;

		gPFPC[i]=gPFPC[i]+inputSumPFPC[i];
		gPFPC[i]=gPFPC[i]*GPFSCALECONSTPC; //? TODO: is that the decay?
		gBCPC[i]=gBCPC[i]+inputBCPC[i]*GBCSCALECONSTPC;
		gBCPC[i]=gBCPC[i]*GBCSCALECONSTPC; //? TODO: where's the decay?

		gSCPCSum=0;
		//TODO: refactor SC input to PC to make it consistent with everything else
		for(int j=0; j<SCPCSYNPERPC; j++)
		{
			gSCPC[i][j]=gSCPC[i][j]+GSCINCCONSTPC*(1-gSCPC[i][j])*apSC[i*SCPCSYNPERPC+j];
			gSCPC[i][j]=gSCPC[i][j]*GSCDecayPC;
			gSCPCSum+=gSCPC[i][j];
		}

		vPC[i]=vPC[i]+(GLEAKPC*(ELEAKPC-vPC[i]))-(gPFPC[i]*vPC[i])+(gBCPC[i]*(EBCPC-vPC[i]))+(gSCPCSum*(ESCPC-vPC[i]));
		threshPC[i]=threshPC[i]+(THRESHDECAYPC*(threshBasePC[i]-threshPC[i]));

		apPC[i]=vPC[i]>threshPC[i];
		threshPC[i]=apPC[i]*THRESHMAXPC+(!apPC[i])*threshPC[i];
		allAPPC=allAPPC+apPC[i];
	}

	memset(inputBCPC, 0, NUMPC*sizeof(unsigned char));
//	memset(inputSumPFPC, 0, NUMPC*sizeof(float));
}

void calcBCActivity()
{
	runBCKernels();
	cudaMemcpy(inputSumPFBC, inputSumPFBCGPU, NUMBC*sizeof(unsigned short), cudaMemcpyDeviceToHost);

	for(int i=0; i<NUMBC; i++)
	{
		gPFBC[i]=gPFBC[i]+(inputSumPFBC[i]*PFINCCONSTBC);
		gPFBC[i]=gPFBC[i]*GPFDECAYBC;
		gPCBC[i]=gPCBC[i]+(inputPCBC[i]*PCINCCONSTBC);
		gPCBC[i]=gPCBC[i]*GPCDECAYBC;

		vBC[i]=vBC[i]+(GLEAKBC*(ELEAKBC-vBC[i]))-(gPFBC[i]*vBC[i])+(gPCBC[i]*(EPCBC-vBC[i]));
		threshBC[i]=threshBC[i]+THRESHDECAYBC*(THRESHBASEBC-threshBC[i]);
		apBC[i]=vBC[i]>threshBC[i];
		threshBC[i]=apBC[i]*THRESHMAXBC+(!apBC[i])*(threshBC[i]);
	}

//	cout<<"vBC: "<<vBC[1]<<endl;
//	cout<<"threshBC: "<<threshBC[1]<<endl;

	memset(inputPCBC, 0, NUMBC*sizeof(unsigned char));
	memset(inputSumPFBC, 0, NUMBC*sizeof(unsigned short));
}
void calcSCActivity()
{
	runSCKernels();
	cudaMemcpy(inputSumPFSC, inputSumPFSCGPU, NUMSC*sizeof(unsigned short), cudaMemcpyDeviceToHost);

	for(int i=0; i<NUMSC; i++)
	{
		gPFSC[i]=gPFSC[i]+(inputSumPFSC[i]*PFINCCONSTSC);
		gPFSC[i]=gPFSC[i]*GPFDECAYSC;

		vSC[i]=vSC[i]+(GLEAKSC*(ELEAKSC-vSC[i]))-gPFSC[i]*vSC[i];

		apSC[i]=vSC[i]>threshSC[i];
		threshSC[i]=threshSC[i]+THRESHDECAYSC*(THRESHBASESC-threshSC[i]);
		threshSC[i]=apSC[i]*THRESHMAXSC+(!apSC[i])*(threshSC[i]);
	}

//	cout<<"vSC: "<<vSC[1]<<endl;
//	cout<<"threshSC: "<<threshSC[1]<<endl;
//	cout<<"apSC: "<<apSC[1]<<endl;
//	if(apSC[1])
//		cout<<"apSC[1]"<<endl;

	memset(inputSumPFSC, 0, NUMSC*sizeof(unsigned short));
}

void calcIOActivity(short t)
{
	for(int i=0; i<NUMIO; i++)
	{
		float gNCSum;
//		float gHMax;
//		float gHTau;
//		float gLtCaHMax;
//		float gLtCaM;

		//calculate DCN input conductance
		gNCSum=0;
		for(int j=0; j<NCIOSYNPERIO; j++)
		{
			gNCIO[i][j]=gNCIO[i][j]*exp(-TIMESTEP/(-GNCDECTSIO*exp(-gNCIO[i][j]/GNCDECTTIO)+GNCDECT0IO));
			gNCIO[i][j]=gNCIO[i][j]+inputNCIO[i][j]*GNCINCSCALEIO*exp(-gNCIO[i][j]/GNCINCTIO);
			gNCSum=gNCSum+gNCIO[i][j];
		}

		gNCSum=gNCSum/10000;

		vIO[i]=vIO[i]+GLEAKIO*(ELEAKIO-vIO[i])+gNCSum*1.5*(ENCIO-vIO[i])+MAXUSDR*(t==USONSET);

		apIO[i]=vIO[i]>threshIO[i];
		threshIO[i]=THRESHMAXIO*apIO[i]+(!apIO[i])*(threshIO[i]+THRESHDECAYIO*(THRESHBASEIO-threshIO[i]));
//		cout<<vIO[i]<<" "<<threshIO[i]<<" "<<gNCSum<<endl;

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

	memset(inputNCIO, false, NUMIO*NCIOSYNPERIO*sizeof(bool));
}

void calcNCActivity()
{
	for(int i=0; i<NUMNC; i++)
	{
		float gMFNMDASum;
		float gMFAMPASum;
		float gPCNCSum;

		gMFNMDASum=0;
		gMFAMPASum=0;
		for(int j=0; j<MFNCSYNPERNC; j++)
		{
			mfNMDANC[i][j]=mfNMDANC[i][j]*MFNMDADECAYNC+inputMFNC[i][j]*mfSynWeightNC[i][j]*(1-mfNMDANC[i][j]);
			gMFNMDANC[i][j]=gMFNMDANC[i][j]+GMFNMDAINCNC*(mfNMDANC[i][j]-gMFNMDANC[i][j]);
			gMFNMDASum=gMFNMDASum+gMFNMDANC[i][j];

			mfAMPANC[i][j]=mfAMPANC[i][j]*MFAMPADECAYNC+inputMFNC[i][j]*mfSynWeightNC[i][j]*(1-mfAMPANC[i][j]);
			gMFAMPANC[i][j]=gMFAMPANC[i][j]+GMFAMPAINCNC*(mfAMPANC[i][j]-gMFAMPANC[i][j]);
			gMFAMPASum=gMFAMPASum+gMFAMPANC[i][j];
		}
		gMFNMDASum=gMFNMDASum*TIMESTEP/((float)MFNCSYNPERNC);
		gMFAMPASum=gMFAMPASum*TIMESTEP/((float)MFNCSYNPERNC);
		gMFNMDASum=gMFNMDASum*vNC[i]/(-80.0f);

		gPCNCSum=0;
		for(int j=0; j<PCNCSYNPERNC; j++)
		{
			gPCNC[i][j]=gPCNC[i][j]*GPCDECAYNC+inputPCNC[i][j]*gPCScaleNC[i][j]*(1-gPCNC[i][j]);
			gPCNCSum=gPCNCSum+gPCNC[i][j];
		}
		gPCNCSum=gPCNCSum*TIMESTEP/((float)PCNCSYNPERNC);

//		cout<<inputMFNC[i][0]<<" "<<mfSynWeightNC[i][0]<<" "<<mfNMDANC[i][0]<<" "<<inputMFNC[i][0]*mfSynWeightNC[i][0]*(1-mfNMDANC[i][0])<<" "<<gMFNMDANC[i][0]<<" "<<mfAMPANC[i][0]<<" "<<gMFAMPANC[i][0]<<endl;

		vNC[i]=vNC[i]+GLEAKNC*(ELEAKNC-vNC[i])-(gMFNMDASum+gMFAMPASum)*vNC[i]+gPCNCSum*(EPCNC-vNC[i]);

		threshNC[i]=threshNC[i]+THRESHDECAYNC*(THRESHBASENC-threshNC[i]);
		apNC[i]=vNC[i]>threshNC[i];

		threshNC[i]=apNC[i]*THRESHMAXNC+(!apNC[i])*threshNC[i];
	}
//	cout<<"-----------"<<endl;

	memset(inputMFNC, false, NUMNC*MFNCSYNPERNC*sizeof(bool));
	memset(inputPCNC, false, NUMNC*PCNCSYNPERNC*sizeof(bool));
}

//functions to update the output targets of each cell type
//according to the connectivity matrix
void updateMFOut()
{
	accessConnLock.lock();
	for(int i=0; i<NUMMF; i++)
	{
		if(apMF[i])
		{
			for(int j=0; j<numMFtoGRN[i]; j++)
			{
				inputsGRH[conMFtoGRN[i][j]/4]=(inputsGRH[conMFtoGRN[i][j]/4]|compMask[conMFtoGRN[i][j]%4]);
			}
//			for(int j=0; j<MFGRSYNPERMF; j++)
//			{
//				inputsGRH[conMFtoGR[i][j]/4]=(inputsGRH[conMFtoGR[i][j]/4]|compMask[conMFtoGR[i][j]%4]);
//			}

			inputMFNC[i/MFNCSYNPERNC][i%MFNCSYNPERNC]=true;
		}
	}

	for(int i=0; i<NUMMF; i++)
	{
		if(apMF[i])
		{
			for(int j=0; j<numMFtoGON[i]; j++)
			{
				inputMFGO[conMFtoGON[i][j]]++;
			}
//			for(int j=0; j<MFGOSYNPERMF; j++)
//			{
//				inputMFGO[conMFtoGO[i][j]]++;
//			}
		}
	}
	accessConnLock.unlock();
}

void updateGOOut()
{
	accessConnLock.lock();
	for(int i=0; i<NUMGO; i++)
	{
		if(apGO[i])
		{
			for(int j=0; j<numGOtoGRN[i]; j++)
			{
				inputsGRH[conGOtoGRN[i][j]/4]=(inputsGRH[conGOtoGRN[i][j]/4]|compMask[conGOtoGRN[i][j]%4+4]);
			}
//			for(int j=0; j<GOGRSYNPERGO; j++)
//			{
//				inputsGRH[conGOtoGR[i][j]/4]=(inputsGRH[conGOtoGR[i][j]/4]|compMask[conGOtoGR[i][j]%4+4]);
//			}
		}
	}
	accessConnLock.unlock();
}
void updateGROut()
{
	accessConnLock.lock();
	for(int i=0; i<NUMGR; i++)
	{
		if((apOutGR[i]&0x06)>0)
		{
			inputGRGO[conGRtoGO[i][0]]+=(apOutGR[i]&0x02)>>1;
			inputGRGO[conGRtoGO[i][1]]+=(apOutGR[i]&0x04)>>2;
		}
//		if((apOutGR[i]&0x02)>0)
//		{
//				inputGRGO[conGRtoGO[i][0]]++;
//		}
//		if((apOutGR[i]&0x04)>0)
//		{
//				inputGRGO[conGRtoGO[i][1]]++;
//		}
	}
	accessConnLock.unlock();
}

void updatePCOut()
{
	accessConnLock.lock();
	for(int i=0; i<NUMPC; i++)
	{
		for(int j=0; j<PCBCSYNPERPC; j++)
		{
			int indBC;

			indBC=i*NUMBCPERPC-6+j;

			indBC=(indBC%NUMBC+NUMBC)%NUMBC;
			inputPCBC[indBC]=inputPCBC[indBC]+apPC[i];
		}

		for(int j=0; j<PCNCSYNPERPC; j++)
		{
			inputPCNC[conPCtoNC[i][j]/PCNCSYNPERNC][conPCtoNC[i][j]%PCNCSYNPERNC]=apPC[i];
		}
	}
	accessConnLock.unlock();
}

void updateBCOut()
{
	accessConnLock.lock();
	for(int i=0; i<NUMBC; i++)
	{
		if(apBC[i])
		{
			for(int j=0; j<BCPCSYNPERBC; j++)
			{
				inputBCPC[conBCtoPC[i][j]]++;
			}
		}
	}
	accessConnLock.unlock();
}

void updateIOOut()
{
	for(int i=0; i<NUMIO; i++)
	{
		plasticityPFPCTimerIO[i]=(!apIO[i])*plasticityPFPCTimerIO[i]+apIO[i]*PFLTDTIMERSTARTIO;
	}
}

void updateIOCouple()
{
	accessConnLock.lock();
	for(int i=0; i<NUMIO; i++)
	{
		for(int j=0; j<IOCOUPSYNPERIO; j++)
		{
			vCoupIO[i]=COUPLESCALEIO*(vIO[conIOCouple[i][j]]-vIO[i]);
		}
		vIO[i]=vIO[i]+vCoupIO[i];
	}
	accessConnLock.unlock();
}

void updateNCOut(CRandomSFMT0 &randGen)
{
	for(int i=0; i<NUMNC; i++)
	{
		synIOReleasePNC[i]=synIOReleasePNC[i]*exp(-TIMESTEP/(IORELPDECTSNC*exp(-synIOReleasePNC[i]/IORELPDECTTNC)+IORELPDECT0NC));
		synIOReleasePNC[i]=synIOReleasePNC[i]+apNC[i]*IORELPINCSCALENC*exp(-synIOReleasePNC[i]/IORELPINCTNC);
	}


	for(int i=0; i<NUMIO; i++)
	{
		for(int j=0; j<NCIOSYNPERIO; j++)
		{
			inputNCIO[i][j]=(randGen.Random()<synIOReleasePNC[j]);
		}
	}
}

void updateMFNCSyn(short t)
{
	bool reset;
	float avgAllAPPC;
	bool doLTD;
	bool doLTP;

	reset=(t%HISTBINWIDTHPC==0);
	if(!reset)
	{
		return;
	}
	histSumAllAPPC=histSumAllAPPC-histAllAPPC[histBinNPC]+allAPPC;
	histAllAPPC[histBinNPC]=allAPPC;
	allAPPC=0;
	histBinNPC++;
	histBinNPC=histBinNPC%NUMHISTBINSPC;

	avgAllAPPC=((float)histSumAllAPPC)/NUMHISTBINSPC;

//	cout<<avgAllAPPC<<endl;

	doLTD=false;
	doLTP=false;
	if(avgAllAPPC>=MFNCLTDTHRESH && !noLTDMFNC)
	{
		doLTD=true;
		noLTDMFNC=true;
	}
	else if(avgAllAPPC<MFNCLTDTHRESH)
	{
		noLTDMFNC=false;
	}

	if(avgAllAPPC<=MFNCLTPTHRESH && !noLTPMFNC)
	{
		doLTP=true;
		noLTPMFNC=true;
	}
	else if(avgAllAPPC>MFNCLTPTHRESH)
	{
		noLTPMFNC=false;
	}

//	cout<<"MFNC plasticity: "<<doLTP<<" "<<doLTD<<endl;
	for(int i=0; i<NUMNC; i++)
	{
		for(int j=0; j<MFNCSYNPERNC; j++)
		{
			mfSynWChangeNC[i][j]=histMF[i*MFNCSYNPERNC+j]*(doLTD*MFNCLTDDECNC+doLTP*MFNCLTPINCNC);
			mfSynWeightNC[i][j]=mfSynWeightNC[i][j]+mfSynWChangeNC[i][j];
			mfSynWeightNC[i][j]=(mfSynWeightNC[i][j]>0)*mfSynWeightNC[i][j];
			mfSynWeightNC[i][j]=(mfSynWeightNC[i][j]<=1)*mfSynWeightNC[i][j]+(mfSynWeightNC[i][j]>1);
			histMF[i*MFNCSYNPERNC+j]=false;
		}
//		cout<<endl<<mfSynWChangeNC[i][0]<<" "<<mfSynWeightNC[i][0]<<endl;
	}
}

void updatePFPCSyn(short t)
{
	if(t%HISTBINWIDTHGR==0)
	{
//		runIOKernels();
	}
}

void calcCellActivities(short time, CRandomSFMT0 &randGen)
{
	calcMFActivity(time, randGen);
	updateMFOut();

	calcGOActivity();
	updateGOOut();

	calcGRActivity(time);
	updateGROut();

	calcPCActivity();
	updatePCOut();

	calcBCActivity();
	updateBCOut();

	calcSCActivity();

	updateIOCouple();
	calcIOActivity(time);
	updateIOOut();

	calcNCActivity();
	updateNCOut(randGen);

	updateMFNCSyn(time);
	updatePFPCSyn(time);
}
