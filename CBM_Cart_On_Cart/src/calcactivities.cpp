/*
 * calcactivities.cpp
 *
 *  Created on: Feb 12, 2010
 *      Author: wen
 */

#include "../includes/calcactivities.h"

void calcCellActivities(short time, CRandomSFMT0 &randGen)
{
	cudaError_t error;
//	error=cudaGetLastError();
//	cout<<"CUDA block1: "<<cudaGetErrorString(error)<<endl;

	cudaThreadSynchronize();
	//calc gr activity
	inputNetwork->runGRActivitiesCUDA(streams[0]);

	//sumPFPC
	for(int i=0; i<NUMMZONES; i++)
	{
		zones[i]->runPFPCSumCUDA(streams[i+1]);
	}
	//sumPFBC
	inputNetwork->runSumPFBCCUDA(streams[2]);

	//sumPFSC
	inputNetwork->runSumPFSCCUDA(streams[3]);

	//sumGROutGO
	inputNetwork->runSumGRGOOutCUDA(streams[4]);

	for(int i=0; i<NUMMZONES; i++)
	{
		zones[i]->runPFPCPlastCUDA(&(streams[1]), time);
	}

	//cp mf async
	inputNetwork->cpyAPMFHosttoGPUCUDA(streams[6]);
	//cp go async
	inputNetwork->cpyAPGOHosttoGPUCUDA(streams[7]);

	for(int i=0; i<NUMMZONES; i++)
	{
		zones[i]->calcPCActivities();
		zones[i]->calcBCActivities();
	}

	inputNetwork->calcSCActivities();

	//sync
	cudaThreadSynchronize();
//	cout<<"block 2"<<endl;
	//updateMFInGR
	inputNetwork->runUpdateMFInGRCUDA(streams[0]);

	//updateGOInGR
	inputNetwork->runUpdateGOInGRCUDA(streams[1]);

	//updatePFOut
	for(int i=0; i<NUMMZONES; i++)
	{
		zones[i]->runPFPCOutCUDA(streams[i+2]);
		zones[i]->cpyPFPCSumCUDA(streams[i+2]);
	}

	inputNetwork->runUpdatePFBCSCOutCUDA(streams[4]);


	//updateGROutGO
	inputNetwork->runUpdateGROutGOCUDA(streams[7]);
//	cout<<"block 2.5"<<endl;

	//cp PFBCSum async
	inputNetwork->cpyPFBCSumGPUtoHostCUDA(streams[5]);

	//cp PFSCSum async
	inputNetwork->cpyPFSCSumGPUtoHostCUDA(streams[6]);

	//cp GROutGOSum
	inputNetwork->cpyGRGOSumGPUtoHostCUDA(streams[3]);

	//updateGRHist
	inputNetwork->runUpdateGRHistoryCUDA(streams[4], time);


//	cout<<"block 2.75"<<endl;

	mfMod->calcActivity(time, 0);
//	cout<<"block 2.76"<<endl;
	inputNetwork->updateMFActivties();

	inputNetwork->calcGOActivities();

	inputNetwork->updateMFtoGOOut();


//	cout<<"block 2.85"<<endl;
	for(int i=0; i<NUMMZONES; i++)
	{
		errMod[i]->calcActivity(time, 0);
//		cout<<"block 2.86"<<endl;
		zones[i]->setErrDrive(errMod[i]->exportErr());
		zones[i]->calcIOActivities();
//		cout<<"block 2.87"<<endl;

		zones[i]->calcNCActivities();
		outputMod[i]->setApNCIn(zones[i]->exportNCAct());
		outputMod[i]->calcOutput();
//		cout<<"block 2.88"<<endl;

		zones[i]->updateMFNCOut();
		zones[i]->updateBCPCOut();
		zones[i]->updateSCPCOut();
		zones[i]->updatePCOut();
//		cout<<"block 2.89"<<endl;
		zones[i]->updateIOOut();
		zones[i]->updateNCOut();
		zones[i]->updateIOCouple();
		zones[i]->updateMFNCSyn(time);
	}
//	cout<<"block 2.95"<<endl;
	inputNetwork->resetMFHist(time);

	externalMod->run();
}
