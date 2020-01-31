/*
 * activityparams.cpp
 *
 *  Created on: Oct 11, 2012
 *      Author: varicella
 */

#include "../../CBMStateInclude/params/activityparams.h"

using namespace std;

ActivityParams::ActivityParams(fstream &infile)
{
	//Assumes that file is in the following format:
	//key\tvalue\n
	//key\tvalue\n
//	map<string,float> tempMap;

//	string line;
//	//loop through file and add key/value pair to map
//	//** this is done to remove the necessity of order in the original file
//	while(getline(infile,line))
//	{
//		tempMap[line.substr(0,line.find_first_of("\t"))]=atof(line.substr(line.find_first_of("\t"),line.size()).c_str());
//	}

	string key;
	float val;

	while(true)
	{
		infile>>key>>val;

		if(key.compare("activityParamEnd")==0)
		{
			break;
		}

		paramMap[key]=val;
	}

	updateParams();
//	msPerTimeStep=paramMap["msPerTimeStep"];
//
//	msPerHistBinMF=paramMap["msPerHistBinMF"];
//	numTSinMFHist=msPerHistBinMF/msPerTimeStep;
//
//	//move elements from map to public variables
////	paramMap.
//	if(paramMap.find("coupleRiRjRatioGO")==paramMap.end())
//	{
//		paramMap["coupleRiRjRatioGO"]=0;
//	}
//
//	if(paramMap.find("goGABAGOGOSynRecTau")==paramMap.end())
//	{
//		paramMap["goGABAGOGOSynRecTau"]=1;
//	}
//
//	if(paramMap.find("goGABAGOGOSynDepF")==paramMap.end())
//	{
//		paramMap["goGABAGOGOSynDepF"]=1;
//	}
//
//	eLeakGO=paramMap["eLeakGO"];
//	eMGluRGO=paramMap["eMGluRGO"];
//	eGABAGO=paramMap["eGABAGO"];
//	threshMaxGO=paramMap["threshMaxGO"];
//	threshRestGO=paramMap["threshBaseGO"];
//	gIncMFtoGO=paramMap["gMFIncGO"];
//	gIncGRtoGO=paramMap["gGRIncGO"];
//	gGABAIncGOtoGO=paramMap["gGOIncGO"];
//	coupleRiRjRatioGO=paramMap["coupleRiRjRatioGO"];
//
//	gMGluRScaleGRtoGO=paramMap["gMGluRScaleGO"];
//	gMGluRIncScaleGO=paramMap["gMGluRIncScaleGO"];
//	mGluRScaleGO=paramMap["mGluRScaleGO"];
//	gluScaleGO=paramMap["gluScaleGO"];
//	gLeakGO=paramMap["rawGLeakGO"]/(6-msPerTimeStep);
//
//	gDecTauMFtoGO=paramMap["gMFDecayTGO"];
//	gDecMFtoGO=exp(-msPerTimeStep/gDecTauMFtoGO);
//
//	gDecTauGRtoGO=paramMap["gGRDecayTGO"];
//	gDecGRtoGO=exp(-msPerTimeStep/gDecTauGRtoGO);
//
//	gGABADecTauGOtoGO=paramMap["gGODecayTGO"];
//	gGABADecGOtoGO=exp(-msPerTimeStep/gGABADecTauGOtoGO);
//
//	//synaptic depression test for GOGABAGO
//	goGABAGOGOSynRecTau=paramMap["goGABAGOGOSynRecTau"];
//	goGABAGOGOSynRec=1-exp(-msPerTimeStep/goGABAGOGOSynRecTau);
//	goGABAGOGOSynDepF=paramMap["goGABAGOGOSynDepF"];
//
//	mGluRDecayGO=paramMap["mGluRDecayGO"];
//	gMGluRIncDecayGO=paramMap["gMGluRIncDecayGO"];
//	gMGluRDecGRtoGO=paramMap["gMGluRDecayGO"];
//	gluDecayGO=paramMap["gluDecayGO"];
//
//	threshDecTauGO=paramMap["threshDecayTGO"];
//	threshDecGO=1-exp(-msPerTimeStep/threshDecTauGO);
//
//
//	eLeakGR=paramMap["eLeakGR"];
//	eGOGR=paramMap["eGOGR"];
//	eMFGR=paramMap["eMFGR"];
//	threshMaxGR=paramMap["threshMaxGR"];
//	threshRestGR=paramMap["threshBaseGR"];
//	gIncMFtoGR=paramMap["gMFIncGR"];
//	gIncGOtoGR=paramMap["gGOIncGR"];
//
//	gDecTauMFtoGR=paramMap["gMFDecayTGR"];
//	gDecMFtoGR=exp(-msPerTimeStep/gDecTauMFtoGR);
//
//	gDecTauGOtoGR=paramMap["gGODecayTGR"];
//	gDecGOtoGR=exp(-msPerTimeStep/gDecTauGOtoGR);
//
//	threshDecTauGR=paramMap["threshDecayTGR"];
//	threshDecGR=1-exp(-msPerTimeStep/threshDecTauGR);
//
//	gLeakGR=paramMap["rawGLeakGR"]/(6-msPerTimeStep);
//
//	msPerHistBinGR=paramMap["msPerHistBinGR"];
//	tsPerHistBinGR=msPerHistBinGR/msPerTimeStep;
//
//	eLeakSC=paramMap["eLeakSC"];
//	gLeakSC=paramMap["rawGLeakSC"]/(6-msPerTimeStep);
//	gDecTauGRtoSC=paramMap["gPFDecayTSC"];
//	gDecGRtoSC=exp(-msPerTimeStep/gDecTauGRtoSC);
//	threshMaxSC=paramMap["threshMaxSC"];
//	threshRestSC=paramMap["threshBaseSC"];
//	threshDecTauSC=paramMap["threshDecayTSC"];
//	threshDecSC=1-exp(-msPerTimeStep/threshDecTauSC);
//	gIncGRtoSC=paramMap["pfIncSC"];
//
//	//**From mzone**
//	eLeakBC=paramMap["eLeakBC"];
//	ePCtoBC=paramMap["ePCBC"];
//	gLeakBC=paramMap["rawGLeakBC"]/(6-msPerTimeStep);
//
//	gDecTauGRtoBC=paramMap["gPFDecayTBC"];
//	gDecGRtoBC=exp(-msPerTimeStep/gDecTauGRtoBC);
//
//	gDecTauPCtoBC=paramMap["gPCDecayTBC"];
//	gDecPCtoBC=exp(-msPerTimeStep/gDecTauPCtoBC);
//
//	threshDecTauBC=paramMap["threshDecayTBC"];
//	threshDecBC=1-exp(-msPerTimeStep/threshDecTauBC);
//
//	threshRestBC=paramMap["threshBaseBC"];
//	threshMaxBC=paramMap["threshMaxBC"];
//	gIncGRtoBC=paramMap["pfIncConstBC"];
//	gIncPCtoBC=paramMap["pcIncConstBC"];
//
//	initSynWofGRtoPC=paramMap["pfSynWInitPC"];
//	eLeakPC=paramMap["eLeakPC"];
//	eBCtoPC=paramMap["eBCPC"];
//	eSCtoPC=paramMap["eSCPC"];
//	threshMaxPC=paramMap["threshMaxPC"];
//	threshRestPC=paramMap["threshBasePC"];
//
//	threshDecTauPC=paramMap["threshDecayTPC"];
//	threshDecPC=1-exp(-msPerTimeStep/threshDecTauPC);
//
//	gLeakPC=paramMap["rawGLeakPC"]/(6-msPerTimeStep);
//
//	gDecTauGRtoPC=paramMap["gPFDecayTPC"];
//	gDecGRtoPC=exp(-msPerTimeStep/gDecTauGRtoPC);
//
//	gDecTauBCtoPC=paramMap["gBCDecayTPC"];
//	gDecBCtoPC=exp(-msPerTimeStep/gDecTauBCtoPC);
//
//	gDecTauSCtoPC=paramMap["gSCDecayTPC"];
//	gDecSCtoPC=exp(-msPerTimeStep/gDecTauSCtoPC);
//
//	gIncSCtoPC=paramMap["gSCIncConstPC"];
//	gIncGRtoPC=paramMap["gPFScaleConstPC"];
//	gIncBCtoPC=paramMap["gBCScaleConstPC"];
//
//	tsPopHistPC=40/msPerTimeStep; //TODO: fixed for now
//	tsPerPopHistBinPC=5/msPerTimeStep;
//	numPopHistBinsPC=tsPopHistPC/tsPerPopHistBinPC;
//
//	coupleRiRjRatioIO=paramMap["coupleScaleIO"];
//	eLeakIO=paramMap["eLeakIO"];
//	eNCtoIO=paramMap["eNCIO"];
//	gLeakIO=paramMap["rawGLeakIO"]/(6-msPerTimeStep);
//	gDecTSofNCtoIO=paramMap["gNCDecTSIO"];
//	gDecTTofNCtoIO=paramMap["gNCDecTTIO"];
//	gDecT0ofNCtoIO=paramMap["gNCDecT0IO"];
//	gIncNCtoIO=paramMap["gNCIncScaleIO"];
//	gIncTauNCtoIO=paramMap["gNCIncTIO"];
//	threshRestIO=paramMap["threshBaseIO"];
//	threshMaxIO=paramMap["threshMaxIO"];
//
//	threshDecTauIO=paramMap["threshDecayTIO"];
//	threshDecIO=1-exp(-msPerTimeStep/threshDecTauIO);
//
//	tsLTDDurationIO=paramMap["msLTDDurationIO"]/msPerTimeStep;
//	tsLTDStartAPIO=paramMap["msLTDStartAPIO"]/msPerTimeStep;
//	tsLTPStartAPIO=paramMap["msLTPStartAPIO"]/msPerTimeStep;
//	synLTPStepSizeGRtoPC=paramMap["grPCLTPIncIO"];
//	synLTDStepSizeGRtoPC=paramMap["grPCLTDDecIO"];
//	grPCHistCheckBinIO=abs(tsLTDStartAPIO/((int)tsPerHistBinGR));
//
//	maxExtIncVIO=paramMap["maxErrDriveIO"];
//
//	eLeakNC=paramMap["eLeakNC"];
//	ePCtoNC=paramMap["ePCNC"];
//
//	gmaxNMDADecTauMFtoNC=paramMap["mfNMDADecayTNC"];
//	gmaxNMDADecMFtoNC=exp(-msPerTimeStep/gmaxNMDADecTauMFtoNC);
//
//	gmaxAMPADecTauMFtoNC=paramMap["mfAMPADecayTNC"];
//	gmaxAMPADecMFtoNC=exp(-msPerTimeStep/gmaxAMPADecTauMFtoNC);
//
//	gNMDAIncMFtoNC=1-exp(-msPerTimeStep/paramMap["rawGMFNMDAIncNC"]);
//	gAMPAIncMFtoNC=1-exp(-msPerTimeStep/paramMap["rawGMFAMPAIncNC"]);
//	gIncAvgPCtoNC=paramMap["gPCScaleAvgNC"];
//
//	gDecTauPCtoNC=paramMap["gPCDecayTNC"];
//	gDecPCtoNC=exp(-msPerTimeStep/gDecTauPCtoNC);
//
//	gLeakNC=paramMap["rawGLeakNC"]/(6-msPerTimeStep);
//
//	threshDecTauNC=paramMap["threshDecayTNC"];
//	threshDecNC=1-exp(-msPerTimeStep/threshDecTauNC);
//
//	threshMaxNC=paramMap["threshMaxNC"];
//	threshRestNC=paramMap["threshBaseNC"];
//	relPDecTSofNCtoIO=paramMap["outIORelPDecTSNC"];
//	relPDecTTofNCtoIO=paramMap["outIORelPDecTTNC"];
//	relPDecT0ofNCtoIO=paramMap["outIORelPDecT0NC"];
//	relPIncNCtoIO=paramMap["outIORelPIncScaleNC"];
//	relPIncTauNCtoIO=paramMap["outIORelPIncTNC"];
//	initSynWofMFtoNC=paramMap["mfSynWInitNC"];
//	synLTDPCPopActThreshMFtoNC=paramMap["mfNCLTDThreshNC"];
//	synLTPPCPopActThreshMFtoNC=paramMap["mfNCLTPThreshNC"];
//	synLTDStepSizeMFtoNC=paramMap["mfNCLTDDecNC"];
//	synLTPStepSizeMFtoNC=paramMap["mfNCLTPIncNC"];
}

void ActivityParams::writeParams(fstream &outfile)
{
	map<string, float>::iterator i;

	for(i=paramMap.begin(); i!=paramMap.end(); i++)
	{
		outfile<<i->first<<" "<<i->second<<endl;
	}

	outfile<<"activityParamEnd 1"<<endl;
}

unsigned int ActivityParams::getMSPerTimeStep()
{
	return msPerTimeStep;
}

void ActivityParams::showParams(ostream &outSt)
{
	outSt<<"msPerTimeStep "<<msPerTimeStep<<endl<<endl;

	outSt<<"msPerHistBinMF "<<msPerHistBinMF<<endl;
	outSt<<"tsPerHistbinMF "<<numTSinMFHist<<endl<<endl;

	outSt<<"coupleRiRjRatioGO "<<coupleRiRjRatioGO<<endl;
	outSt<<"eLeakGO "<<eLeakGO<<endl;
	outSt<<"eMGluRGO "<<eMGluRGO<<endl;
	outSt<<"eGABAGO "<<eGABAGO<<endl;
	outSt<<"threshMaxGO "<<threshMaxGO<<endl;
	outSt<<"threshBaseGO "<<threshRestGO<<endl;
	outSt<<"gMFIncGO "<<gIncMFtoGO<<endl;
	outSt<<"gGRIncGO "<<gIncGRtoGO<<endl;
	outSt<<"gGOIncGO "<<gGABAIncGOtoGO<<endl;
	outSt<<"gMGluRScaleGO "<<gMGluRScaleGRtoGO<<endl;
	outSt<<"mGluRScaleGO "<<mGluRScaleGO<<endl;
	outSt<<"gluScaleGO "<<gluScaleGO<<endl;
	outSt<<"gLeakGO "<<gLeakGO<<endl;
	outSt<<"gMFDecayTGO "<<gDecTauMFtoGO<<endl;
	outSt<<"gMFDecayGO "<<gDecMFtoGO<<endl;
	outSt<<"gGRDecayTGO "<<gDecTauGRtoGO<<endl;
	outSt<<"gGRDecayGO "<<gDecGRtoGO<<endl;
	outSt<<"gGODecayTGO "<<gGABADecTauGOtoGO<<endl;
	outSt<<"gGODecayGO "<<gGABADecGOtoGO<<endl;
	outSt<<"mGluRDecayGO "<<mGluRDecayGO<<endl;
	outSt<<"gMGluRIncDecayGO "<<gMGluRIncDecayGO<<endl;
	outSt<<"gluDecayGO "<<gluDecayGO<<endl;
	outSt<<"threshDecayTGO "<<threshDecTauGO<<endl;
	outSt<<"threshDecayGO "<<threshDecGO<<endl<<endl;

	outSt<<"eLeakGR "<<eLeakGR<<endl;
	outSt<<"eGOGR "<<eGOGR<<endl;
	outSt<<"eMFGR "<<eMFGR<<endl;
	outSt<<"threshMaxGR "<<threshMaxGR<<endl;
	outSt<<"threshBaseGR "<<threshRestGR<<endl;
	outSt<<"gMFIncGR "<<gIncMFtoGR<<endl;
	outSt<<"gGOIncGR "<<gIncGOtoGR<<endl;
	outSt<<"gMFDecayTGR "<<gDecTauMFtoGR<<endl;
	outSt<<"gMFDecayGR "<<gDecMFtoGR<<endl;
	outSt<<"gGODecayTGR "<<gDecTauGOtoGR<<endl;
	outSt<<"gGODecayGR "<<gDecGOtoGR<<endl;
	outSt<<"threshDecayTGR "<<threshDecTauGR<<endl;
	outSt<<"threshDecayGR "<<threshDecGR<<endl;
	outSt<<"gLeakGR "<<gLeakGR<<endl;
	outSt<<"msPerHistBinGR "<<msPerHistBinGR<<endl;
	outSt<<"tsPerHistBinGR "<<tsPerHistBinGR<<endl<<endl;

	outSt<<"eLeakSC "<<eLeakSC<<endl;
	outSt<<"gLeakSC "<<gLeakSC<<endl;
	outSt<<"gPFDecayTSC "<<gDecTauGRtoSC<<endl;
	outSt<<"gPFDecaySC "<<gDecGRtoSC<<endl;
	outSt<<"threshMaxSC "<<threshMaxSC<<endl;
	outSt<<"threshBaseSC "<<threshRestSC<<endl;
	outSt<<"threshDecayTSC "<<threshDecTauSC<<endl;
	outSt<<"threshDecaySC "<<threshDecSC<<endl;
	outSt<<"pfIncSC "<<gIncGRtoSC<<endl<<endl;

	outSt<<"eLeakBC "<<eLeakBC<<endl;
	outSt<<"ePCBC "<<ePCtoBC<<endl;
	outSt<<"gLeakBC "<<gLeakBC<<endl;
	outSt<<"gPFDecayTBC "<<gDecTauGRtoBC<<endl;
	outSt<<"gPFDecayBC "<<gDecGRtoBC<<endl;
	outSt<<"gPCDecayTBC "<<gDecTauPCtoBC<<endl;
	outSt<<"gPCDecayBC "<<gDecPCtoBC<<endl;
	outSt<<"threshBaseBC "<<threshRestBC<<endl;
	outSt<<"threshMaxBC "<<threshMaxBC<<endl;
	outSt<<"threshDecayTBC "<<threshDecTauBC<<endl;
	outSt<<"threshDecayBC "<<threshDecBC<<endl;
	outSt<<"pfIncConstBC "<<gIncGRtoBC<<endl;
	outSt<<"pcIncConstBC "<<gIncPCtoBC<<endl<<endl;

	outSt<<"pfSynWInitPC "<<initSynWofGRtoPC<<endl;
	outSt<<"eLeakPC "<<eLeakPC<<endl;
	outSt<<"eBCPC "<<eBCtoPC<<endl;
	outSt<<"eSCPC "<<eSCtoPC<<endl;
	outSt<<"threshMaxPC "<<threshMaxPC<<endl;
	outSt<<"threshBasePC "<<threshRestPC<<endl;
	outSt<<"threshDecayTPC "<<threshDecTauPC<<endl;
	outSt<<"threshDecayPC "<<threshDecPC<<endl;
	outSt<<"gLeakPC "<<gLeakPC<<endl;
	outSt<<"gPFDecayTPC "<<gDecTauGRtoPC<<endl;
	outSt<<"gPFDecayPC "<<gDecGRtoPC<<endl;
	outSt<<"gBCDecayTPC "<<gDecTauBCtoPC<<endl;
	outSt<<"gBCDecayPC "<<gDecBCtoPC<<endl;
	outSt<<"gSCDecayTPC "<<gDecTauSCtoPC<<endl;
	outSt<<"gSCDecayPC "<<gDecSCtoPC<<endl;
	outSt<<"gSCIncConstPC "<<gIncSCtoPC<<endl;
	outSt<<"gPFScaleConstPC "<<gIncGRtoPC<<endl;
	outSt<<"gBCScaleConstPC "<<gIncBCtoPC<<endl;
	outSt<<"tsPopHistPC "<<tsPopHistPC<<endl;
	outSt<<"tsPerPopHistBinPC "<<tsPerPopHistBinPC<<endl;
	outSt<<"numPopHistBinsPC "<<numPopHistBinsPC<<endl<<endl;

	outSt<<"coupleScaleIO "<<coupleRiRjRatioIO<<endl;
	outSt<<"eLeakIO "<<eLeakIO<<endl;
	outSt<<"eNCIO "<<eNCtoIO<<endl;
	outSt<<"gLeakIO "<<gLeakIO<<endl;
	outSt<<"gNCDecTSIO "<<gDecTSofNCtoIO<<endl;
	outSt<<"gNCDecTTIO "<<gDecTTofNCtoIO<<endl;
	outSt<<"gNCDecT0IO "<<gDecT0ofNCtoIO<<endl;
	outSt<<"gNCIncScaleIO "<<gIncNCtoIO<<endl;
	outSt<<"gNCIncTIO "<<gIncTauNCtoIO<<endl;
	outSt<<"threshBaseIO "<<threshRestIO<<endl;
	outSt<<"threshMaxIO "<<threshMaxIO<<endl;
	outSt<<"threshDecayTIO "<<threshDecTauIO<<endl;
	outSt<<"threshDecayIO "<<threshDecIO<<endl;
	outSt<<"tsLTDDurationIO "<<tsLTDDurationIO<<endl;
	outSt<<"tsLTDStartAPIO "<<tsLTDStartAPIO<<endl;
	outSt<<"tsLTPStartAPIO "<<tsLTPStartAPIO<<endl;
	outSt<<"grPCLTPIncIO "<<synLTPStepSizeGRtoPC<<endl;
	outSt<<"grPCLTDDecIO "<<synLTDStepSizeGRtoPC<<endl;
	outSt<<"grPCHistCheckBinIO "<<grPCHistCheckBinIO<<endl;
	outSt<<"maxErrDriveIO "<<maxExtIncVIO<<endl<<endl;

	outSt<<"eLeakNC "<<eLeakNC<<endl;
	outSt<<"ePCNC "<<ePCtoNC<<endl;
	outSt<<"mfNMDADecayTNC "<<gmaxNMDADecTauMFtoNC<<endl;
	outSt<<"mfNMDADecayNC "<<gmaxNMDADecMFtoNC<<endl;
	outSt<<"mfAMPADecayTNC "<<gmaxAMPADecTauMFtoNC<<endl;
	outSt<<"mfAMPADecayNC "<<gmaxAMPADecMFtoNC<<endl;
	outSt<<"gMFNMDAIncNC "<<gNMDAIncMFtoNC<<endl;
	outSt<<"gMFAMPAIncNC "<<gAMPAIncMFtoNC<<endl;
	outSt<<"gPCScaleAvgNC "<<gIncAvgPCtoNC<<endl;
	outSt<<"gPCDecayTNC "<<gDecTauPCtoNC<<endl;
	outSt<<"gPCDecayNC "<<gDecPCtoNC<<endl;
	outSt<<"gLeakNC "<<gLeakNC<<endl;
	outSt<<"threshDecayTNC "<<threshDecTauNC<<endl;
	outSt<<"threshDecayNC "<<threshDecNC<<endl;
	outSt<<"threshMaxNC "<<threshMaxNC<<endl;
	outSt<<"threshBaseNC "<<threshRestNC<<endl;
	outSt<<"outIORelPDecTSNC "<<relPDecTSofNCtoIO<<endl;
	outSt<<"outIORelPDecTTNC "<<relPDecTTofNCtoIO<<endl;
	outSt<<"outIORelPDecT0NC "<<relPDecT0ofNCtoIO<<endl;
	outSt<<"outIORelPIncScaleNC "<<relPIncNCtoIO<<endl;
	outSt<<"outIORelPIncTNC "<<relPIncTauNCtoIO<<endl;
	outSt<<"mfSynWInitNC "<<initSynWofMFtoNC<<endl;
	outSt<<"mfNCLTDThreshNC "<<synLTDPCPopActThreshMFtoNC<<endl;
	outSt<<"mfNCLTPThreshNC "<<synLTPPCPopActThreshMFtoNC<<endl;
	outSt<<"mfNCLTDDecNC "<<synLTDStepSizeMFtoNC<<endl;
	outSt<<"mfNCLTPIncNC "<<synLTPStepSizeMFtoNC<<endl;
}

map<string, float> ActivityParams::getParamCopy()
{
	map<string, float> paramCopy;

	map<string, float>::iterator i;

	for(i=paramMap.begin(); i!=paramMap.end(); i++)
	{
		paramCopy[i->first]=i->second;
	}

	return paramCopy;
}

float ActivityParams::getParam(string paramName)
{
	return paramMap[paramName];
}

bool ActivityParams::setParam(string paramName, float value)
{
	if(paramMap.find(paramName)==paramMap.end())
	{
		return false;
	}
	paramMap[paramName]=value;

	updateParams();

	return true;
}

void ActivityParams::updateParams()
{
	msPerTimeStep=paramMap["msPerTimeStep"];

	msPerHistBinMF=paramMap["msPerHistBinMF"];
	numTSinMFHist=msPerHistBinMF/msPerTimeStep;

	//move elements from map to public variables
//	paramMap.
	if(paramMap.find("coupleRiRjRatioGO")==paramMap.end())
	{
		paramMap["coupleRiRjRatioGO"]=0;
	}

	if(paramMap.find("goGABAGOGOSynRecTau")==paramMap.end())
	{
		paramMap["goGABAGOGOSynRecTau"]=1;
	}

	if(paramMap.find("goGABAGOGOSynDepF")==paramMap.end())
	{
		paramMap["goGABAGOGOSynDepF"]=1;
	}

	eLeakGO=paramMap["eLeakGO"];
	eMGluRGO=paramMap["eMGluRGO"];
	eGABAGO=paramMap["eGABAGO"];
	threshMaxGO=paramMap["threshMaxGO"];
	threshRestGO=paramMap["threshBaseGO"];
	gIncMFtoGO=paramMap["gMFIncGO"];
	gIncGRtoGO=paramMap["gGRIncGO"];
	gGABAIncGOtoGO=paramMap["gGOIncGO"];
	coupleRiRjRatioGO=paramMap["coupleRiRjRatioGO"];

	gMGluRScaleGRtoGO=paramMap["gMGluRScaleGO"];
	gMGluRIncScaleGO=paramMap["gMGluRIncScaleGO"];
	mGluRScaleGO=paramMap["mGluRScaleGO"];
	gluScaleGO=paramMap["gluScaleGO"];
	gLeakGO=paramMap["rawGLeakGO"]/(6-msPerTimeStep);

	gDecTauMFtoGO=paramMap["gMFDecayTGO"];
	gDecMFtoGO=exp(-msPerTimeStep/gDecTauMFtoGO);

	gDecTauGRtoGO=paramMap["gGRDecayTGO"];
	gDecGRtoGO=exp(-msPerTimeStep/gDecTauGRtoGO);

	gGABADecTauGOtoGO=paramMap["gGODecayTGO"];
	gGABADecGOtoGO=exp(-msPerTimeStep/gGABADecTauGOtoGO);

	//synaptic depression test for GOGABAGO
	goGABAGOGOSynRecTau=paramMap["goGABAGOGOSynRecTau"];
	goGABAGOGOSynRec=1-exp(-msPerTimeStep/goGABAGOGOSynRecTau);
	goGABAGOGOSynDepF=paramMap["goGABAGOGOSynDepF"];

	mGluRDecayGO=paramMap["mGluRDecayGO"];
	gMGluRIncDecayGO=paramMap["gMGluRIncDecayGO"];
	gMGluRDecGRtoGO=paramMap["gMGluRDecayGO"];
	gluDecayGO=paramMap["gluDecayGO"];

	threshDecTauGO=paramMap["threshDecayTGO"];
	threshDecGO=1-exp(-msPerTimeStep/threshDecTauGO);


	eLeakGR=paramMap["eLeakGR"];
	eGOGR=paramMap["eGOGR"];
	eMFGR=paramMap["eMFGR"];
	threshMaxGR=paramMap["threshMaxGR"];
	threshRestGR=paramMap["threshBaseGR"];
	gIncMFtoGR=paramMap["gMFIncGR"];
	gIncGOtoGR=paramMap["gGOIncGR"];

	gDecTauMFtoGR=paramMap["gMFDecayTGR"];
	gDecMFtoGR=exp(-msPerTimeStep/gDecTauMFtoGR);

	gDecTauGOtoGR=paramMap["gGODecayTGR"];
	gDecGOtoGR=exp(-msPerTimeStep/gDecTauGOtoGR);

	threshDecTauGR=paramMap["threshDecayTGR"];
	threshDecGR=1-exp(-msPerTimeStep/threshDecTauGR);

	gLeakGR=paramMap["rawGLeakGR"]/(6-msPerTimeStep);

	msPerHistBinGR=paramMap["msPerHistBinGR"];
	tsPerHistBinGR=msPerHistBinGR/msPerTimeStep;

	eLeakSC=paramMap["eLeakSC"];
	gLeakSC=paramMap["rawGLeakSC"]/(6-msPerTimeStep);
	gDecTauGRtoSC=paramMap["gPFDecayTSC"];
	gDecGRtoSC=exp(-msPerTimeStep/gDecTauGRtoSC);
	threshMaxSC=paramMap["threshMaxSC"];
	threshRestSC=paramMap["threshBaseSC"];
	threshDecTauSC=paramMap["threshDecayTSC"];
	threshDecSC=1-exp(-msPerTimeStep/threshDecTauSC);
	gIncGRtoSC=paramMap["pfIncSC"];

	//**From mzone**
	eLeakBC=paramMap["eLeakBC"];
	ePCtoBC=paramMap["ePCBC"];
	gLeakBC=paramMap["rawGLeakBC"]/(6-msPerTimeStep);

	gDecTauGRtoBC=paramMap["gPFDecayTBC"];
	gDecGRtoBC=exp(-msPerTimeStep/gDecTauGRtoBC);

	gDecTauPCtoBC=paramMap["gPCDecayTBC"];
	gDecPCtoBC=exp(-msPerTimeStep/gDecTauPCtoBC);

	threshDecTauBC=paramMap["threshDecayTBC"];
	threshDecBC=1-exp(-msPerTimeStep/threshDecTauBC);

	threshRestBC=paramMap["threshBaseBC"];
	threshMaxBC=paramMap["threshMaxBC"];
	gIncGRtoBC=paramMap["pfIncConstBC"];
	gIncPCtoBC=paramMap["pcIncConstBC"];

	initSynWofGRtoPC=paramMap["pfSynWInitPC"];
	eLeakPC=paramMap["eLeakPC"];
	eBCtoPC=paramMap["eBCPC"];
	eSCtoPC=paramMap["eSCPC"];
	threshMaxPC=paramMap["threshMaxPC"];
	threshRestPC=paramMap["threshBasePC"];

	threshDecTauPC=paramMap["threshDecayTPC"];
	threshDecPC=1-exp(-msPerTimeStep/threshDecTauPC);

	gLeakPC=paramMap["rawGLeakPC"]/(6-msPerTimeStep);

	gDecTauGRtoPC=paramMap["gPFDecayTPC"];
	gDecGRtoPC=exp(-msPerTimeStep/gDecTauGRtoPC);

	gDecTauBCtoPC=paramMap["gBCDecayTPC"];
	gDecBCtoPC=exp(-msPerTimeStep/gDecTauBCtoPC);

	gDecTauSCtoPC=paramMap["gSCDecayTPC"];
	gDecSCtoPC=exp(-msPerTimeStep/gDecTauSCtoPC);

	gIncSCtoPC=paramMap["gSCIncConstPC"];
	gIncGRtoPC=paramMap["gPFScaleConstPC"];
	gIncBCtoPC=paramMap["gBCScaleConstPC"];

	tsPopHistPC=40/msPerTimeStep; //TODO: fixed for now
	tsPerPopHistBinPC=5/msPerTimeStep;
	numPopHistBinsPC=tsPopHistPC/tsPerPopHistBinPC;

	coupleRiRjRatioIO=paramMap["coupleScaleIO"];
	eLeakIO=paramMap["eLeakIO"];
	eNCtoIO=paramMap["eNCIO"];
	gLeakIO=paramMap["rawGLeakIO"]/(6-msPerTimeStep);
	gDecTSofNCtoIO=paramMap["gNCDecTSIO"];
	gDecTTofNCtoIO=paramMap["gNCDecTTIO"];
	gDecT0ofNCtoIO=paramMap["gNCDecT0IO"];
	gIncNCtoIO=paramMap["gNCIncScaleIO"];
	gIncTauNCtoIO=paramMap["gNCIncTIO"];
	threshRestIO=paramMap["threshBaseIO"];
	threshMaxIO=paramMap["threshMaxIO"];

	threshDecTauIO=paramMap["threshDecayTIO"];
	threshDecIO=1-exp(-msPerTimeStep/threshDecTauIO);

	tsLTDDurationIO=paramMap["msLTDDurationIO"]/msPerTimeStep;
	tsLTDStartAPIO=paramMap["msLTDStartAPIO"]/msPerTimeStep;
	tsLTPStartAPIO=paramMap["msLTPStartAPIO"]/msPerTimeStep;
	synLTPStepSizeGRtoPC=paramMap["grPCLTPIncIO"];
	synLTDStepSizeGRtoPC=paramMap["grPCLTDDecIO"];
	grPCHistCheckBinIO=abs(tsLTDStartAPIO/((int)tsPerHistBinGR));

	maxExtIncVIO=paramMap["maxErrDriveIO"];

	eLeakNC=paramMap["eLeakNC"];
	ePCtoNC=paramMap["ePCNC"];

	gmaxNMDADecTauMFtoNC=paramMap["mfNMDADecayTNC"];
	gmaxNMDADecMFtoNC=exp(-msPerTimeStep/gmaxNMDADecTauMFtoNC);

	gmaxAMPADecTauMFtoNC=paramMap["mfAMPADecayTNC"];
	gmaxAMPADecMFtoNC=exp(-msPerTimeStep/gmaxAMPADecTauMFtoNC);

	gNMDAIncMFtoNC=1-exp(-msPerTimeStep/paramMap["rawGMFNMDAIncNC"]);
	gAMPAIncMFtoNC=1-exp(-msPerTimeStep/paramMap["rawGMFAMPAIncNC"]);
	gIncAvgPCtoNC=paramMap["gPCScaleAvgNC"];

	gDecTauPCtoNC=paramMap["gPCDecayTNC"];
	gDecPCtoNC=exp(-msPerTimeStep/gDecTauPCtoNC);

	gLeakNC=paramMap["rawGLeakNC"]/(6-msPerTimeStep);

	threshDecTauNC=paramMap["threshDecayTNC"];
	threshDecNC=1-exp(-msPerTimeStep/threshDecTauNC);

	threshMaxNC=paramMap["threshMaxNC"];
	threshRestNC=paramMap["threshBaseNC"];
	relPDecTSofNCtoIO=paramMap["outIORelPDecTSNC"];
	relPDecTTofNCtoIO=paramMap["outIORelPDecTTNC"];
	relPDecT0ofNCtoIO=paramMap["outIORelPDecT0NC"];
	relPIncNCtoIO=paramMap["outIORelPIncScaleNC"];
	relPIncTauNCtoIO=paramMap["outIORelPIncTNC"];
	initSynWofMFtoNC=paramMap["mfSynWInitNC"];
	synLTDPCPopActThreshMFtoNC=paramMap["mfNCLTDThreshNC"];
	synLTPPCPopActThreshMFtoNC=paramMap["mfNCLTPThreshNC"];
	synLTDStepSizeMFtoNC=paramMap["mfNCLTDDecNC"];
	synLTPStepSizeMFtoNC=paramMap["mfNCLTPIncNC"];
}
