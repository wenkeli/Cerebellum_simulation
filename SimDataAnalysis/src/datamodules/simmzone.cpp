/*
 * simmzone.cpp
 *
 *  Created on: Aug 15, 2011
 *      Author: consciousness
 */

#include "../../includes/datamodules/simmzone.h"

SimMZone::SimMZone(ifstream &infile)
{
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
			<<mfSynWNC[0][0]<<" "<<mfSynWNC[numNC-1][numMFInPerNC-1]<<" "
			<<synIOPReleaseNC[0]<<" :"<<noLTPMFNC<<endl;
}
