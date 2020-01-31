/*
 * innetconnectivitystate.cpp
 *
 *  Created on: Nov 6, 2012
 *      Author: consciousness
 */

#include "../../CBMStateInclude/state/innetconnectivitystate.h"

using namespace std;

InNetConnectivityState::InNetConnectivityState
	(ConnectivityParams *parameters, unsigned int msPerStep, int randSeed)
{
	CRandomSFMT *randGen;
	cp=parameters;

	randGen=new CRandomSFMT0(randSeed);

	cout<<"Input net state construction:"<<endl;
	cout<<"allocating memory"<<endl;
	allocateMemory();
	cout<<"initializing variables"<<endl;
	initializeVals();
	cout<<"connecting GR and GL"<<endl;
	connectGRGL(randGen);
	cout<<"connecting GO and GL"<<endl;
	connectGOGL(randGen);
	cout<<"connecting MF and GL"<<endl;
	connectMFGL(randGen);
	cout<<"translating MF GL"<<endl;
	translateMFGL();
	cout<<"translating GO and GL"<<endl;
	translateGOGL();
	cout<<"connecting GR to GO"<<endl;
	connectGRGO(randGen);
	cout<<"connecting GO to GO"<<endl;
	connectGOGO(randGen);
	cout<<"assigning GR delays"<<endl;
	assignGRDelays(msPerStep);
	cout<<"done"<<endl;

	delete randGen;
}

InNetConnectivityState::InNetConnectivityState
	(ConnectivityParams *parameters, fstream &infile)
{
	cp=parameters;

	allocateMemory();

	stateRW(true, infile);
}

InNetConnectivityState::InNetConnectivityState(const InNetConnectivityState &state)
{
	cp=state.cp;

	allocateMemory();

	arrayCopy<ct_uint8_t>(haspGLfromMFtoGL, state.haspGLfromMFtoGL, cp->numGL);
	arrayCopy<ct_uint32_t>(pGLfromMFtoGL, state.pGLfromMFtoGL, cp->numGL);

	arrayCopy<ct_int32_t>(numpGLfromGLtoGO, state.numpGLfromGLtoGO, cp->numGL);
	arrayCopy<ct_uint32_t>(pGLfromGLtoGO[0], state.pGLfromGLtoGO[0],
			cp->numGL*cp->maxnumpGLfromGLtoGO);

	arrayCopy<ct_int32_t>(numpGLfromGOtoGL, state.numpGLfromGOtoGL, cp->numGL);
	arrayCopy<ct_uint32_t>(pGLfromGOtoGL[0], state.pGLfromGOtoGL[0],
			cp->numGL*cp->maxnumpGLfromGOtoGL);

	arrayCopy<ct_int32_t>(numpGLfromGLtoGR, state.numpGLfromGLtoGR, cp->numGL);
	arrayCopy<ct_uint32_t>(pGLfromGLtoGR[0], state.pGLfromGLtoGR[0],
			cp->numGL*cp->maxnumpGLfromGOtoGL);


	arrayCopy<ct_int32_t>(numpMFfromMFtoGL, state.numpMFfromMFtoGL, cp->numMF);
	arrayCopy<ct_uint32_t>(pMFfromMFtoGL[0], state.pMFfromMFtoGL[0],
			cp->numMF*cp->numpMFfromMFtoGL);

	arrayCopy<ct_int32_t>(numpMFfromMFtoGR, state.numpMFfromMFtoGR, cp->numMF);
	arrayCopy<ct_uint32_t>(pMFfromMFtoGR[0], state.pMFfromMFtoGR[0],
			cp->numMF*cp->maxnumpMFfromMFtoGR);

	arrayCopy<ct_int32_t>(numpMFfromMFtoGO, state.numpMFfromMFtoGO, cp->numMF);
	arrayCopy<ct_uint32_t>(pMFfromMFtoGO[0], state.pMFfromMFtoGO[0],
			cp->numMF*cp->maxnumpMFfromMFtoGO);


	arrayCopy<ct_int32_t>(numpGOfromGLtoGO, state.numpGOfromGLtoGO, cp->numGO);
	arrayCopy<ct_uint32_t>(pGOfromGLtoGO[0], state.pGOfromGLtoGO[0],
			cp->numGO*cp->maxnumpGOfromGLtoGO);

	arrayCopy<ct_int32_t>(numpGOfromGOtoGL, state.numpGOfromGOtoGL, cp->numGO);
	arrayCopy<ct_uint32_t>(pGOfromGOtoGL[0], state.pGOfromGOtoGL[0],
			cp->numGO*cp->maxnumpGOfromGOtoGL);

	arrayCopy<ct_int32_t>(numpGOfromMFtoGO, state.numpGOfromMFtoGO, cp->numGO);
	arrayCopy<ct_uint32_t>(pGOfromMFtoGO[0], state.pGOfromMFtoGO[0],
			cp->numGO*cp->maxnumpGOfromMFtoGO);

	arrayCopy<ct_int32_t>(numpGOfromGOtoGR, state.numpGOfromGOtoGR, cp->numGO);
	arrayCopy<ct_uint32_t>(pGOfromGOtoGR[0], state.pGOfromGOtoGR[0],
			cp->numGO*cp->maxnumpGOfromGOtoGR);

	arrayCopy<ct_int32_t>(numpGOfromGRtoGO, state.numpGOfromGRtoGO, cp->numGO);
	arrayCopy<ct_uint32_t>(pGOfromGRtoGO[0], state.pGOfromGRtoGO[0],
			cp->numGO*cp->maxnumpGOfromGRtoGO);

	arrayCopy<ct_int32_t>(numpGOGABAInGOGO, state.numpGOGABAInGOGO, cp->numGO);
	arrayCopy<ct_uint32_t>(pGOGABAInGOGO[0], state.pGOGABAInGOGO[0],
			cp->numGO*cp->maxnumpGOGABAInGOGO);

	arrayCopy<ct_int32_t>(numpGOGABAOutGOGO, state.numpGOGABAOutGOGO, cp->numGO);
	arrayCopy<ct_uint32_t>(pGOGABAOutGOGO[0], state.pGOGABAOutGOGO[0],
			cp->numGO*cp->maxnumpGOGABAOutGOGO);

	arrayCopy<ct_int32_t>(numpGOCoupInGOGO, state.numpGOCoupInGOGO, cp->numGO);
	arrayCopy<ct_uint32_t>(pGOCoupInGOGO[0], state.pGOCoupInGOGO[0],
			cp->numGO*cp->maxnumpGOCoupInGOGO);

	arrayCopy<ct_int32_t>(numpGOCoupOutGOGO, state.numpGOCoupOutGOGO, cp->numGO);
	arrayCopy<ct_uint32_t>(pGOCoupOutGOGO[0], state.pGOCoupOutGOGO[0],
			cp->numGO*cp->maxnumpGOCoupOutGOGO);


	arrayCopy<ct_uint32_t>(pGRDelayMaskfromGRtoBSP, state.pGRDelayMaskfromGRtoBSP, cp->numGR);

	arrayCopy<ct_int32_t>(numpGRfromGLtoGR, state.numpGRfromGLtoGR, cp->numGR);
	arrayCopy<ct_uint32_t>(pGRfromGLtoGR[0], state.pGRfromGLtoGR[0],
			cp->numGR*cp->maxnumpGRfromGLtoGR);

	arrayCopy<ct_int32_t>(numpGRfromGRtoGO, state.numpGRfromGRtoGO, cp->numGR);
	arrayCopy<ct_uint32_t>(pGRfromGRtoGO[0], state.pGRfromGRtoGO[0],
			cp->numGR*cp->maxnumpGRfromGRtoGO);
	arrayCopy<ct_uint32_t>(pGRDelayMaskfromGRtoGO[0], state.pGRDelayMaskfromGRtoGO[0],
			cp->numGR*cp->maxnumpGRfromGRtoGO);

	arrayCopy<ct_int32_t>(numpGRfromGOtoGR, state.numpGRfromGOtoGR, cp->numGR);
	arrayCopy<ct_uint32_t>(pGRfromGOtoGR[0], state.pGRfromGOtoGR[0],
			cp->numGR*cp->maxnumpGRfromGOtoGR);

	arrayCopy<ct_int32_t>(numpGRfromMFtoGR, state.numpGRfromMFtoGR, cp->numGR);
	arrayCopy<ct_uint32_t>(pGRfromMFtoGR[0], state.pGRfromMFtoGR[0],
			cp->numGR*cp->maxnumpGRfromMFtoGR);
}

InNetConnectivityState::~InNetConnectivityState()
{
//	delete[] glomeruli;
	//glomeruli
	delete[] haspGLfromMFtoGL;
	delete[] pGLfromMFtoGL;

	delete[] numpGLfromGLtoGO;
	delete2DArray<ct_uint32_t>(pGLfromGLtoGO);

	delete[] numpGLfromGOtoGL;
	delete2DArray<ct_uint32_t>(pGLfromGOtoGL);

	delete[] numpGLfromGLtoGR;
	delete2DArray<ct_uint32_t>(pGLfromGLtoGR);


	//mossy fibers
	delete[] numpMFfromMFtoGL;
	delete2DArray<ct_uint32_t>(pMFfromMFtoGL);

	delete[] numpMFfromMFtoGR;
	delete2DArray<ct_uint32_t>(pMFfromMFtoGR);

	delete[] numpMFfromMFtoGO;
	delete2DArray<ct_uint32_t>(pMFfromMFtoGO);


	//golgi
	delete[] numpGOfromGLtoGO;
	delete2DArray<ct_uint32_t>(pGOfromGLtoGO);

	delete[] numpGOfromGOtoGL;
	delete2DArray<ct_uint32_t>(pGOfromGOtoGL);

	delete[] numpGOfromMFtoGO;
	delete2DArray<ct_uint32_t>(pGOfromMFtoGO);

	delete[] numpGOfromGOtoGR;
	delete2DArray<ct_uint32_t>(pGOfromGOtoGR);

	delete[] numpGOfromGRtoGO;
	delete2DArray<ct_uint32_t>(pGOfromGRtoGO);

	delete[] numpGOGABAInGOGO;
	delete2DArray<ct_uint32_t>(pGOGABAInGOGO);

	delete[] numpGOGABAOutGOGO;
	delete2DArray<ct_uint32_t>(pGOGABAOutGOGO);

	delete[] numpGOCoupInGOGO;
	delete2DArray<ct_uint32_t>(pGOCoupInGOGO);

	delete[] numpGOCoupOutGOGO;
	delete2DArray<ct_uint32_t>(pGOCoupOutGOGO);

	//granule
	delete[] pGRDelayMaskfromGRtoBSP;

	delete[] numpGRfromGLtoGR;
	delete2DArray<ct_uint32_t>(pGRfromGLtoGR);

	delete[] numpGRfromGRtoGO;
	delete2DArray<ct_uint32_t>(pGRfromGRtoGO);
	delete2DArray<ct_uint32_t>(pGRDelayMaskfromGRtoGO);

	delete[] numpGRfromGOtoGR;
	delete2DArray<ct_uint32_t>(pGRfromGOtoGR);

	delete[] numpGRfromMFtoGR;
	delete2DArray<ct_uint32_t>(pGRfromMFtoGR);
}

void InNetConnectivityState::writeState(fstream &outfile)
{
	cout<<"start innetOut"<<endl;
	stateRW(false, (fstream &)outfile);
	cout<<"end innetOut"<<endl;
}

bool InNetConnectivityState::equivalent(const InNetConnectivityState &compState)
{
	bool eq;

	eq=true;
	for(int i=0; i<cp->numGL; i++)
	{
		eq=eq && (haspGLfromMFtoGL[i]==compState.haspGLfromMFtoGL[i]);
	}

	for(int i=0; i<cp->numGO; i++)
	{
		eq=eq && (numpGOfromGOtoGR[i]==compState.numpGOfromGOtoGR[i]);
	}

	for(int i=0; i<cp->numGR; i++)
	{
		eq=eq && (numpGRfromGOtoGR[i]==compState.numpGRfromGOtoGR[i]);
	}
	return eq;
}

vector<ct_uint32_t> InNetConnectivityState::getpGOfromGOtoGLCon(int goN)
{
	return getConCommon(goN, numpGOfromGOtoGL, pGOfromGOtoGL);
}

vector<ct_uint32_t> InNetConnectivityState::getpGOfromGLtoGOCon(int goN)
{
	return getConCommon(goN, numpGOfromGLtoGO, pGOfromGLtoGO);
}

vector<ct_uint32_t> InNetConnectivityState::getpMFfromMFtoGLCon(int mfN)
{
	return getConCommon(mfN, numpMFfromMFtoGL, pMFfromMFtoGL);
}

vector<ct_uint32_t> InNetConnectivityState::getpGLfromGLtoGRCon(int glN)
{
	return getConCommon(glN, numpGLfromGLtoGR, pGLfromGLtoGR);
}


vector<ct_uint32_t> InNetConnectivityState::getpGRfromMFtoGR(int grN)
{
	return getConCommon(grN, numpGRfromMFtoGR, pGRfromMFtoGR);
}

vector<vector<ct_uint32_t> > InNetConnectivityState::getpGRPopfromMFtoGR()
{
	vector<vector<ct_uint32_t> > retVect;

	retVect.resize(cp->numGR);

	for(int i=0; i<cp->numGR; i++)
	{
		retVect[i]=getpGRfromMFtoGR(i);
	}

	return retVect;
}

vector<ct_uint32_t> InNetConnectivityState::getpGRfromGOtoGR(int grN)
{
	return getConCommon(grN, numpGRfromGOtoGR, pGRfromGOtoGR);
}

vector<vector<ct_uint32_t> > InNetConnectivityState::getpGRPopfromGOtoGR()
{
	vector<vector<ct_uint32_t> > retVect;

	retVect.resize(cp->numGR);

	for(int i=0; i<cp->numGR; i++)
	{
		retVect[i]=getpGRfromGOtoGR(i);
	}

	return retVect;
}

vector<ct_uint32_t> InNetConnectivityState::getpGOfromGRtoGOCon(int goN)
{
	return getConCommon(goN, numpGOfromGRtoGO, pGOfromGRtoGO);
}

vector<ct_uint32_t> InNetConnectivityState::getpGOfromGOtoGRCon(int goN)
{
	return getConCommon(goN, numpGOfromGOtoGR, pGOfromGOtoGR);
}

vector<ct_uint32_t> InNetConnectivityState::getpGOOutGOGOCon(int goN)
{
	return getConCommon(goN, numpGOGABAOutGOGO, pGOGABAOutGOGO);
}

vector<ct_uint32_t> InNetConnectivityState::getpGOInGOGOCon(int goN)
{
	return getConCommon(goN, numpGOGABAInGOGO, pGOGABAInGOGO);
}

vector<ct_uint32_t> InNetConnectivityState::getpMFfromMFtoGRCon(int mfN)
{
	return getConCommon(mfN, numpMFfromMFtoGR, pMFfromMFtoGR);
}

vector<ct_uint32_t> InNetConnectivityState::getpMFfromMFtoGOCon(int mfN)
{
	return getConCommon(mfN, numpMFfromMFtoGO, pMFfromMFtoGO);
}

vector<ct_uint32_t> InNetConnectivityState::getpGOfromMFtoGOCon(int goN)
{
	return getConCommon(goN, numpGOfromMFtoGO, pGOfromMFtoGO);
}
vector<vector<ct_uint32_t> > InNetConnectivityState::getpGOPopfromMFtoGOCon()
{
	vector<vector<ct_uint32_t> > con;

	con.resize(cp->numGO);
	for(int i=0; i<cp->numGO; i++)
	{
		con[i]=getpGOfromMFtoGOCon(i);
	}

	return con;
}

vector<ct_uint32_t> InNetConnectivityState::getConCommon(int cellN, ct_int32_t *numpCellCon, ct_uint32_t **pCellCon)
{
	vector<ct_uint32_t> inds;
	inds.resize(numpCellCon[cellN]);
	for(int i=0; i<numpCellCon[cellN]; i++)
	{
		inds[i]=pCellCon[cellN][i];
	}

	return inds;
}

vector<ct_uint32_t> InNetConnectivityState::getGOIncompIndfromGRtoGO()
{
	vector<ct_uint32_t> goInds;

	for(int i=0; i<cp->numGO; i++)
	{
		if(numpGOfromGRtoGO[i]<cp->maxnumpGOfromGRtoGO)
		{
			goInds.push_back(i);
		}
	}

	return goInds;
}

vector<ct_uint32_t> InNetConnectivityState::getGRIncompIndfromGRtoGO()
{
	vector<ct_uint32_t> grInds;

	for(int i=0; i<cp->numGR; i++)
	{
		if(numpGRfromGRtoGO[i]<cp->maxnumpGRfromGRtoGO)
		{
			grInds.push_back(i);
		}
	}

	return grInds;
}

bool InNetConnectivityState::deleteGOGOConPair(int srcGON, int destGON)
{
	bool hasCon;

	int conN;
	hasCon=false;
	for(int i=0; i<numpGOGABAOutGOGO[srcGON]; i++)
	{
		if(pGOGABAOutGOGO[srcGON][i]==destGON)
		{
			hasCon=true;
			conN=i;
			break;
		}
	}
	if(!hasCon)
	{
		return hasCon;
	}

	for(int i=conN; i<numpGOGABAOutGOGO[srcGON]-1; i++)
	{
		pGOGABAOutGOGO[srcGON][i]=pGOGABAOutGOGO[srcGON][i+1];
	}
	numpGOGABAOutGOGO[srcGON]--;

	for(int i=0; i<numpGOGABAInGOGO[destGON]; i++)
	{
		if(pGOGABAInGOGO[destGON][i]==srcGON)
		{
			conN=i;
		}
	}
	for(int i=conN; i<numpGOGABAInGOGO[destGON]-1; i++)
	{
		pGOGABAInGOGO[destGON][i]=pGOGABAInGOGO[destGON][i+1];
	}
	numpGOGABAInGOGO[destGON]--;

	return hasCon;
}

bool InNetConnectivityState::addGOGOConPair(int srcGON, int destGON)
{
	if(numpGOGABAOutGOGO[srcGON]>=cp->maxnumpGOGABAOutGOGO ||
			numpGOGABAInGOGO[destGON]>=cp->maxnumpGOGABAInGOGO)
	{
		return false;
	}

	pGOGABAOutGOGO[srcGON][numpGOGABAOutGOGO[srcGON]]=destGON;
	numpGOGABAOutGOGO[srcGON]++;

	pGOGABAInGOGO[destGON][numpGOGABAInGOGO[destGON]]=srcGON;
	numpGOGABAInGOGO[destGON]++;

	return true;
}

void InNetConnectivityState::allocateMemory()
{
//	glomeruli=new Glomerulus[p->numGL](p->maxNumGODenPerGL, p->maxNumGOAxPerGL, p->maxNumGRDenPerGL);
//	glomeruli.resize(p->numGL, Glomerulus(p->maxNumGODenPerGL, p->maxNumGOAxPerGL, p->maxNumGRDenPerGL));
	haspGLfromMFtoGL=new ct_uint8_t[cp->numGL];
	pGLfromMFtoGL=new ct_uint32_t[cp->numGL];

	numpGLfromGLtoGO=new ct_int32_t[cp->numGL];
	pGLfromGLtoGO=allocate2DArray<ct_uint32_t>(cp->numGL, cp->maxnumpGLfromGLtoGO);

	numpGLfromGOtoGL=new ct_int32_t[cp->numGL];
	pGLfromGOtoGL=allocate2DArray<ct_uint32_t>(cp->numGL, cp->maxnumpGLfromGOtoGL);

	numpGLfromGLtoGR=new ct_int32_t[cp->numGL];
	pGLfromGLtoGR=allocate2DArray<ct_uint32_t>(cp->numGL, cp->maxnumpGLfromGLtoGR);

	//mf
	numpMFfromMFtoGL=new ct_int32_t[cp->numMF];
	pMFfromMFtoGL=allocate2DArray<ct_uint32_t>(cp->numMF, cp->numpMFfromMFtoGL);

	numpMFfromMFtoGR=new ct_int32_t[cp->numMF];
	pMFfromMFtoGR=allocate2DArray<ct_uint32_t>(cp->numMF, cp->maxnumpMFfromMFtoGR);

	numpMFfromMFtoGO=new ct_int32_t[cp->numMF];
	pMFfromMFtoGO=allocate2DArray<ct_uint32_t>(cp->numMF, cp->maxnumpMFfromMFtoGO);

	//golgi
	numpGOfromGLtoGO=new ct_int32_t[cp->numGO];
	pGOfromGLtoGO=allocate2DArray<ct_uint32_t>(cp->numGO, cp->maxnumpGOfromGLtoGO);

	numpGOfromGOtoGL=new ct_int32_t[cp->numGO];
	pGOfromGOtoGL=allocate2DArray<ct_uint32_t>(cp->numGO, cp->maxnumpGOfromGOtoGL);

	numpGOfromMFtoGO=new ct_int32_t[cp->numGO];
	pGOfromMFtoGO=allocate2DArray<ct_uint32_t>(cp->numGO, cp->maxnumpGOfromMFtoGO);

	numpGOfromGOtoGR=new ct_int32_t[cp->numGO];
	pGOfromGOtoGR=allocate2DArray<ct_uint32_t>(cp->numGO, cp->maxnumpGOfromGOtoGR);

	numpGOfromGRtoGO=new ct_int32_t[cp->numGO];
	pGOfromGRtoGO=allocate2DArray<ct_uint32_t>(cp->numGO, cp->maxnumpGOfromGRtoGO);

	numpGOGABAInGOGO=new ct_int32_t[cp->numGO];
	pGOGABAInGOGO=allocate2DArray<ct_uint32_t>(cp->numGO, cp->maxnumpGOGABAInGOGO);

	numpGOGABAOutGOGO=new ct_int32_t[cp->numGO];
	pGOGABAOutGOGO=allocate2DArray<ct_uint32_t>(cp->numGO, cp->maxnumpGOGABAOutGOGO);

	numpGOCoupInGOGO=new ct_int32_t[cp->numGO];
	pGOCoupInGOGO=allocate2DArray<ct_uint32_t>(cp->numGO, cp->maxnumpGOCoupInGOGO);

	numpGOCoupOutGOGO=new ct_int32_t[cp->numGO];
	pGOCoupOutGOGO=allocate2DArray<ct_uint32_t>(cp->numGO, cp->maxnumpGOCoupOutGOGO);


	//granule
	pGRDelayMaskfromGRtoBSP=new ct_uint32_t[cp->numGR];

	numpGRfromGLtoGR=new ct_int32_t[cp->numGR];
	pGRfromGLtoGR=allocate2DArray<ct_uint32_t>(cp->numGR, cp->maxnumpGRfromGOtoGR);

	numpGRfromGRtoGO=new ct_int32_t[cp->numGR];
	pGRfromGRtoGO=allocate2DArray<ct_uint32_t>(cp->numGR, cp->maxnumpGRfromGRtoGO);
	pGRDelayMaskfromGRtoGO=allocate2DArray<ct_uint32_t>(cp->numGR, cp->maxnumpGRfromGRtoGO);

	numpGRfromGOtoGR=new ct_int32_t[cp->numGR];
	pGRfromGOtoGR=allocate2DArray<ct_uint32_t>(cp->numGR, cp->maxnumpGRfromGOtoGR);

	numpGRfromMFtoGR=new ct_int32_t[cp->numGR];
	pGRfromMFtoGR=allocate2DArray<ct_uint32_t>(cp->numGR, cp->maxnumpGRfromGOtoGR);
}

void InNetConnectivityState::stateRW(bool read, std::fstream &file)
{
	cout<<"glomerulus"<<endl;
	//glomerulus
	rawBytesRW((char *)haspGLfromMFtoGL, cp->numGL*sizeof(ct_uint8_t), read, file);
	cout<<"glomerulus 1.1"<<endl;
	rawBytesRW((char *)pGLfromMFtoGL, cp->numGL*sizeof(ct_int32_t), read, file);

	cout<<"glomerulus 2"<<endl;
	rawBytesRW((char *)numpGLfromGLtoGO, cp->numGL*sizeof(ct_int32_t), read, file);
	rawBytesRW((char *)pGLfromGLtoGO[0], cp->numGL*cp->maxnumpGLfromGLtoGO*sizeof(ct_uint32_t), read, file);

	cout<<"glomerulus 3"<<endl;
	rawBytesRW((char *)numpGLfromGOtoGL, cp->numGL*sizeof(ct_int32_t), read, file);
	rawBytesRW((char *)pGLfromGOtoGL[0], cp->numGL*cp->maxnumpGLfromGOtoGL*sizeof(ct_uint32_t), read, file);

	cout<<"glomerulus 4"<<endl;
	rawBytesRW((char *)numpGLfromGLtoGR, cp->numGL*sizeof(ct_int32_t), read, file);
	rawBytesRW((char *)pGLfromGLtoGR[0], cp->numGL*cp->maxnumpGLfromGLtoGR*sizeof(ct_uint32_t), read, file);

	cout<<"mf"<<endl;
	//mossy fibers
	rawBytesRW((char *)numpMFfromMFtoGL, cp->numMF*sizeof(ct_int32_t), read, file);
	rawBytesRW((char *)pMFfromMFtoGL[0], cp->numMF*cp->numpMFfromMFtoGL*sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)numpMFfromMFtoGR, cp->numMF*sizeof(ct_int32_t), read, file);
	rawBytesRW((char *)pMFfromMFtoGR[0], cp->numMF*cp->maxnumpMFfromMFtoGR*sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)numpMFfromMFtoGO, cp->numMF*sizeof(ct_int32_t), read, file);
	rawBytesRW((char *)pMFfromMFtoGO[0], cp->numMF*cp->maxnumpMFfromMFtoGO*sizeof(ct_uint32_t), read, file);

	cout<<"golgi"<<endl;
	//golgi
	rawBytesRW((char *)numpGOfromGLtoGO, cp->numGO*sizeof(ct_int32_t), read, file);
	rawBytesRW((char *)pGOfromGLtoGO[0], cp->numGO*cp->maxnumpGOfromGLtoGO*sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)numpGOfromGOtoGL, cp->numGO*sizeof(ct_int32_t), read, file);
	rawBytesRW((char *)pGOfromGOtoGL[0], cp->numGO*cp->maxnumpGOfromGOtoGL*sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)numpGOfromMFtoGO, cp->numGO*sizeof(ct_int32_t), read, file);
	rawBytesRW((char *)pGOfromMFtoGO[0], cp->numGO*cp->maxnumpGOfromMFtoGO*sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)numpGOfromGOtoGR, cp->numGO*sizeof(ct_int32_t), read, file);
	rawBytesRW((char *)pGOfromGOtoGR[0], cp->numGO*cp->maxnumpGOfromGOtoGR*sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)numpGOfromGRtoGO, cp->numGO*sizeof(ct_int32_t), read, file);
	rawBytesRW((char *)pGOfromGRtoGO[0], cp->numGO*cp->maxnumpGOfromGRtoGO*sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)numpGOGABAInGOGO, cp->numGO*sizeof(ct_int32_t), read, file);
	rawBytesRW((char *)pGOGABAInGOGO[0], cp->numGO*cp->maxnumpGOGABAInGOGO*sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)numpGOGABAOutGOGO, cp->numGO*sizeof(ct_int32_t), read, file);
	rawBytesRW((char *)pGOGABAOutGOGO[0], cp->numGO*cp->maxnumpGOGABAOutGOGO*sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)numpGOCoupInGOGO, cp->numGO*sizeof(ct_int32_t), read, file);
	rawBytesRW((char *)pGOCoupInGOGO[0], cp->numGO*cp->maxnumpGOCoupInGOGO*sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)numpGOCoupOutGOGO, cp->numGO*sizeof(ct_int32_t), read, file);
	rawBytesRW((char *)pGOCoupOutGOGO[0], cp->numGO*cp->maxnumpGOCoupOutGOGO*sizeof(ct_uint32_t), read, file);

	cout<<"granule"<<endl;
	//granule
	rawBytesRW((char *)pGRDelayMaskfromGRtoBSP, cp->numGR*sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)numpGRfromGLtoGR, cp->numGR*sizeof(ct_int32_t), read, file);
	rawBytesRW((char *)pGRfromGLtoGR[0], cp->numGR*cp->maxnumpGRfromGOtoGR*sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)numpGRfromGRtoGO, cp->numGR*sizeof(ct_int32_t), read, file);
	rawBytesRW((char *)pGRfromGRtoGO[0], cp->maxnumpGRfromGRtoGO*cp->numGR*sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pGRDelayMaskfromGRtoGO[0], cp->maxnumpGRfromGRtoGO*cp->numGR*sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)numpGRfromGOtoGR, cp->numGR*sizeof(ct_int32_t), read, file);
	rawBytesRW((char *)pGRfromGOtoGR[0], cp->maxnumpGRfromGOtoGR*cp->numGR*sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)numpGRfromMFtoGR, cp->numGR*sizeof(ct_int32_t), read, file);
	rawBytesRW((char *)pGRfromMFtoGR[0], cp->maxnumpGRfromGOtoGR*cp->numGR*sizeof(ct_uint32_t), read, file);
}

void InNetConnectivityState::initializeVals()
{
	//glomerulus
//	for(int i=0; i<cp->numGL; i++)
//	{
//		haspGLfromMFtoGL[i]=false;
//		numpGLfromGLtoGO[i]=0;
//		numpGLfromGOtoGL[i]=0;
//		numpGLfromGLtoGR[i]=0;
//	}
//
//	//mossy fiber
//	for(int i=0; i<cp->numMF; i++)
//	{
//		numpMFfromMFtoGL[i]=0;
//		numpMFfromMFtoGR[i]=0;
//		numpMFfromMFtoGO[i]=0;
//	}
//
//	//golgi cell
//	for(int i=0; i<cp->numGO; i++)
//	{
//		numpGOfromGLtoGO[i]=0;
//		numpGOfromGOtoGL[i]=0;
//		numpGOfromMFtoGO[i]=0;
//		numpGOfromGOtoGR[i]=0;
//		numpGOfromGRtoGO[i]=0;
//		numpGOInGOGO[i]=0;
//		numpGOOutGOGO[i]=0;
//	}
//
//	//granule cell
//	for(int i=0; i<cp->numGR; i++)
//	{
//		pGRDelayMaskfromGRtoBSP[i]=1;
//
//		numpGRfromGLtoGR[i]=0;
//		numpGRfromGRtoGO[i]=0;
//		numpGRfromGOtoGR[i]=0;
//		numpGRfromMFtoGR[i]=0;
//	}

	arrayInitialize<ct_uint8_t>(haspGLfromMFtoGL, 0, cp->numGL);
	arrayInitialize<ct_uint32_t>(pGLfromMFtoGL, UINT_MAX, cp->numGL);

	arrayInitialize<ct_int32_t>(numpGLfromGLtoGO, 0, cp->numGL);
	arrayInitialize<ct_uint32_t>(pGLfromGLtoGO[0], UINT_MAX, cp->numGL*cp->maxnumpGLfromGLtoGO);

	arrayInitialize<ct_int32_t>(numpGLfromGOtoGL, 0, cp->numGL);
	arrayInitialize<ct_uint32_t>(pGLfromGOtoGL[0], UINT_MAX, cp->numGL*cp->maxnumpGLfromGOtoGL);

	arrayInitialize<ct_int32_t>(numpGLfromGLtoGR, 0, cp->numGL);
	arrayInitialize<ct_uint32_t>(pGLfromGLtoGR[0], UINT_MAX, cp->numGL*cp->maxnumpGLfromGOtoGL);


	arrayInitialize<ct_int32_t>(numpMFfromMFtoGL, 0, cp->numMF);
	arrayInitialize<ct_uint32_t>(pMFfromMFtoGL[0], UINT_MAX, cp->numMF*cp->numpMFfromMFtoGL);

	arrayInitialize<ct_int32_t>(numpMFfromMFtoGR, 0, cp->numMF);
	arrayInitialize<ct_uint32_t>(pMFfromMFtoGR[0], UINT_MAX, cp->numMF*cp->maxnumpMFfromMFtoGR);

	arrayInitialize<ct_int32_t>(numpMFfromMFtoGO, 0, cp->numMF);
	arrayInitialize<ct_uint32_t>(pMFfromMFtoGO[0], UINT_MAX, cp->numMF*cp->maxnumpMFfromMFtoGO);


	arrayInitialize<ct_int32_t>(numpGOfromGLtoGO, 0, cp->numGO);
	arrayInitialize<ct_uint32_t>(pGOfromGLtoGO[0], UINT_MAX, cp->numGO*cp->maxnumpGOfromGLtoGO);

	arrayInitialize<ct_int32_t>(numpGOfromGOtoGL, 0, cp->numGO);
	arrayInitialize<ct_uint32_t>(pGOfromGOtoGL[0], UINT_MAX, cp->numGO*cp->maxnumpGOfromGOtoGL);

	arrayInitialize<ct_int32_t>(numpGOfromMFtoGO, 0, cp->numGO);
	arrayInitialize<ct_uint32_t>(pGOfromMFtoGO[0], UINT_MAX, cp->numGO*cp->maxnumpGOfromMFtoGO);

	arrayInitialize<ct_int32_t>(numpGOfromGOtoGR, 0, cp->numGO);
	arrayInitialize<ct_uint32_t>(pGOfromGOtoGR[0], UINT_MAX, cp->numGO*cp->maxnumpGOfromGOtoGR);

	arrayInitialize<ct_int32_t>(numpGOfromGRtoGO, 0, cp->numGO);
	arrayInitialize<ct_uint32_t>(pGOfromGRtoGO[0], UINT_MAX, cp->numGO*cp->maxnumpGOfromGRtoGO);

	arrayInitialize<ct_int32_t>(numpGOGABAInGOGO, 0, cp->numGO);
	arrayInitialize<ct_uint32_t>(pGOGABAInGOGO[0], UINT_MAX, cp->numGO*cp->maxnumpGOGABAInGOGO);

	arrayInitialize<ct_int32_t>(numpGOGABAOutGOGO, 0, cp->numGO);
	arrayInitialize<ct_uint32_t>(pGOGABAOutGOGO[0], UINT_MAX, cp->numGO*cp->maxnumpGOGABAOutGOGO);

	arrayInitialize<ct_int32_t>(numpGOCoupInGOGO, 0, cp->numGO);
	arrayInitialize<ct_uint32_t>(pGOCoupInGOGO[0], UINT_MAX, cp->numGO*cp->maxnumpGOCoupInGOGO);

	arrayInitialize<ct_int32_t>(numpGOCoupOutGOGO, 0, cp->numGO);
	arrayInitialize<ct_uint32_t>(pGOCoupOutGOGO[0], UINT_MAX, cp->numGO*cp->maxnumpGOCoupOutGOGO);

	arrayInitialize<ct_uint32_t>(pGRDelayMaskfromGRtoBSP, 0, cp->numGR);

	arrayInitialize<ct_int32_t>(numpGRfromGLtoGR, 0, cp->numGR);
	arrayInitialize<ct_uint32_t>(pGRfromGLtoGR[0], UINT_MAX, cp->numGR*cp->maxnumpGRfromGLtoGR);

	arrayInitialize<ct_int32_t>(numpGRfromGRtoGO, 0, cp->numGR);
	arrayInitialize<ct_uint32_t>(pGRfromGRtoGO[0], UINT_MAX, cp->numGR*cp->maxnumpGRfromGRtoGO);
	arrayInitialize<ct_uint32_t>(pGRDelayMaskfromGRtoGO[0], UINT_MAX, cp->numGR*cp->maxnumpGRfromGRtoGO);

	arrayInitialize<ct_int32_t>(numpGRfromGOtoGR, 0, cp->numGR);
	arrayInitialize<ct_uint32_t>(pGRfromGOtoGR[0], UINT_MAX, cp->numGR*cp->maxnumpGRfromGOtoGR);

	arrayInitialize<ct_int32_t>(numpGRfromMFtoGR, 0, cp->numGR);
	arrayInitialize<ct_uint32_t>(pGRfromMFtoGR[0], UINT_MAX, cp->numGR*cp->maxnumpGRfromMFtoGR);
}

void InNetConnectivityState::connectGRGL(CRandomSFMT *randGen)
{
	connectCommon(pGRfromGLtoGR, numpGRfromGLtoGR,
			pGLfromGLtoGR, numpGLfromGLtoGR,
			cp->maxnumpGRfromGLtoGR, cp->numGR,
			cp->maxnumpGLfromGLtoGR, cp->lownumpGLfromGLtoGR,
			cp->grX, cp->grY, cp->glX, cp->glY,
			cp->spanGRDenOnGLX, cp->spanGRDenOnGLY,
			20000, 50000, true,
			randGen);
#ifdef DEBUGOUT
	for(int i=0; i<10; i++)
	{
		cout<<"numpGLfromGLtoGR["<<i<<"]: "<<numpGLfromGLtoGR[i]<<endl;
	}
#endif
}

void InNetConnectivityState::connectGOGL(CRandomSFMT *randGen)
{
	connectCommon(pGOfromGLtoGO, numpGOfromGLtoGO,
			pGLfromGLtoGO, numpGLfromGLtoGO,
			cp->maxnumpGOfromGLtoGO, cp->numGO,
			cp->maxnumpGLfromGLtoGO, cp->maxnumpGLfromGLtoGO,
			cp->goX, cp->goY, cp->glX, cp->glY,
			cp->spanGODecDenOnGLX, cp->spanGODecDenOnGLY,
			20000, 50000, false,
			randGen);

	connectCommon(pGOfromGOtoGL, numpGOfromGOtoGL,
			pGLfromGOtoGL, numpGLfromGOtoGL,
			cp->maxnumpGOfromGOtoGL, cp->numGO,
			cp->maxnumpGLfromGOtoGL, cp->maxnumpGLfromGOtoGL,
			cp->goX, cp->goY, cp->glX, cp->glY,
			cp->spanGOAxonOnGLX, cp->spanGOAxonOnGLY,
			20000, 50000, false,
			randGen);
}

void InNetConnectivityState::connectMFGL(CRandomSFMT *randGen)
{
	int lastMFSynCount;

	for(int i=0; i<cp->numMF-1; i++)
	{
		for(int j=0; j<cp->numpMFfromMFtoGL; j++)
		{
			int glIndex;
			while(true)
			{
				glIndex=randGen->IRandom(0, cp->numGL-1);
				if(!haspGLfromMFtoGL[glIndex])
				{
					pGLfromMFtoGL[glIndex]=i;
					haspGLfromMFtoGL[glIndex]=true;
					pMFfromMFtoGL[i][j]=glIndex;
					numpMFfromMFtoGL[i]++;
					break;
				}
			}
		}
	}

	lastMFSynCount=0;
	for(int i=0; i<cp->numGL; i++)
	{
		if(!haspGLfromMFtoGL[i])
		{
			haspGLfromMFtoGL[i]=true;
			pGLfromMFtoGL[i]=cp->numMF-1;
			pMFfromMFtoGL[cp->numMF-1][lastMFSynCount]=i;
			lastMFSynCount++;
		}
	}
#ifdef DEBUGOUT
	for(int i=0; i<10; i++)
	{
		cout<<"numpMFfromMFtoGL["<<i<<"]: "<<numpMFfromMFtoGL[i]<<endl;
	}
#endif
}

void InNetConnectivityState::translateMFGL()
{
	translateCommon(pMFfromMFtoGL, numpMFfromMFtoGL,
			pGLfromGLtoGR, numpGLfromGLtoGR,
			pMFfromMFtoGR, numpMFfromMFtoGR,
			pGRfromMFtoGR, numpGRfromMFtoGR,
			cp->numMF);

#ifdef DEBUGOUT
	for(int i=0; i<10; i++)
	{
		cout<<"numpGRfromMFtoGR["<<i<<"]: "<<numpGRfromMFtoGR[i]<<endl;
		for(int j=0; j<numpGRfromMFtoGR[i]; j++)
		{
			cout<<pGRfromMFtoGR[i][j]<<" ";
		}
		cout<<endl;
	}
#endif

	translateCommon(pMFfromMFtoGL, numpMFfromMFtoGL,
			pGLfromGLtoGO, numpGLfromGLtoGO,
			pMFfromMFtoGO, numpMFfromMFtoGO,
			pGOfromMFtoGO, numpGOfromMFtoGO,
			cp->numMF);
#ifdef DEBUGOUT
	for(int i=0; i<10; i++)
	{
		cout<<"numpGOfromMFtoGO["<<i<<"]: "<<numpGOfromMFtoGO[i]<<endl;
	}
#endif
}

void InNetConnectivityState::translateGOGL()
{
	translateCommon(pGOfromGOtoGL, numpGOfromGOtoGL,
			pGLfromGLtoGR, numpGLfromGLtoGR,
			pGOfromGOtoGR, numpGOfromGOtoGR,
			pGRfromGOtoGR, numpGRfromGOtoGR,
			cp->numGO);
}

void InNetConnectivityState::connectGRGO(CRandomSFMT *randGen)
{
	connectCommon(pGOfromGRtoGO, numpGOfromGRtoGO,
			pGRfromGRtoGO, numpGRfromGRtoGO,
			cp->maxnumpGOfromGRtoGO, cp->numGO,
			cp->maxnumpGRfromGRtoGO, cp->maxnumpGRfromGRtoGO,
			cp->goX, cp->goY, cp->grX, cp->grY,
			cp->spanGOAscDenOnGRX, cp->spanGOAscDenOnGRY,
			20000, 50000, false,
			randGen);
}

void InNetConnectivityState::connectGOGO(CRandomSFMT *randGen)
{
	for(int i=0; i<cp->numGO; i++)
	{
		int currPosX;
		int currPosY;

		currPosX=i%cp->goX;
		currPosY=i/cp->goX;

		for(int j=0; j<cp->maxnumpGOGABAOutGOGO; j++)
		{
			int destPosX;
			int destPosY;
			ct_uint32_t destInd;

			if(randGen->Random()>=(1-cp->gogoGABALocalCon[j][2]))
			{
				destPosX=currPosX+cp->gogoGABALocalCon[j][0];
				destPosX=(destPosX%cp->goX+cp->goX)%cp->goX;

				destPosY=currPosY+cp->gogoGABALocalCon[j][1];
				destPosY=(destPosY%cp->goY+cp->goY)%cp->goY;

				destInd=cp->goX*destPosY+destPosX;
//				cout<<i<<" destx "<<destPosX<<" desty "<<destPosY<<" destInd "<<destInd;
//				cout.flush();
//				cout<<" numOut[i] "<<numpGOOutGOGO[i];
//				cout.flush();
//				cout<<" numIn[i] "<<numpGOInGOGO[destInd]<<endl;
				pGOGABAOutGOGO[i][numpGOGABAOutGOGO[i]]=destInd;
				numpGOGABAOutGOGO[i]++;

				pGOGABAInGOGO[destInd][numpGOGABAInGOGO[destInd]]=i;
				numpGOGABAInGOGO[destInd]++;
			}
		}

		for(int j=0; j<cp->maxnumpGOCoupOutGOGO; j++)
		{
			int destPosX;
			int destPosY;
			ct_uint32_t destInd;

			if(randGen->Random()>=(1-cp->gogoCoupLocalCon[j][2]))
			{
				destPosX=currPosX+cp->gogoCoupLocalCon[j][0];
				destPosX=(destPosX%cp->goX+cp->goX)%cp->goX;

				destPosY=currPosY+cp->gogoCoupLocalCon[j][1];
				destPosY=(destPosY%cp->goY+cp->goY)%cp->goY;

				destInd=cp->goX*destPosY+destPosX;
//				cout<<i<<" destx "<<destPosX<<" desty "<<destPosY<<" destInd "<<destInd;
//				cout.flush();
//				cout<<" numOut[i] "<<numpGOOutGOGO[i];
//				cout.flush();
//				cout<<" numIn[i] "<<numpGOInGOGO[destInd]<<endl;
				pGOCoupOutGOGO[i][numpGOCoupOutGOGO[i]]=destInd;
				numpGOCoupOutGOGO[i]++;

				pGOCoupInGOGO[destInd][numpGOCoupInGOGO[destInd]]=i;
				numpGOCoupInGOGO[destInd]++;
			}
		}
	}
}

void InNetConnectivityState::assignGRDelays(unsigned int msPerStep)
{
	for(int i=0; i<cp->numGR; i++)
	{
		int grPosX;
		int grBCPCSCDist;

		//calculate x coordinate of GR position
		grPosX=i%cp->grX;

		//calculate distance of GR to BC, PC, and SC, and assign time delay.
		grBCPCSCDist=abs((int)(cp->grX/2-grPosX));
		pGRDelayMaskfromGRtoBSP[i]=0x1<<
				(int)((grBCPCSCDist/cp->grPFVelInGRXPerTStep+
						cp->grAFDelayInTStep)/msPerStep);

		for(int j=0; j<numpGRfromGRtoGO[i]; j++)
		{
			int dfromGRtoGO;
			int goPosX;

			goPosX=(pGRfromGRtoGO[i][j]%cp->goX)*(((float)cp->grX)/cp->goX);

			dfromGRtoGO=abs(goPosX-grPosX);

			if(dfromGRtoGO > cp->grX/2)
			{
				if(goPosX<grPosX)
				{
					dfromGRtoGO=goPosX+cp->grX-grPosX;
				}
				else
				{
					dfromGRtoGO=grPosX+cp->grX-goPosX;
				}
			}

			pGRDelayMaskfromGRtoGO[i][j]=0x1<<
					(int)((dfromGRtoGO/cp->grPFVelInGRXPerTStep+
							cp->grAFDelayInTStep)/msPerStep);
		}
	}
}

void InNetConnectivityState::connectCommon(ct_uint32_t **srcConArr, ct_int32_t *srcNumCon,
		ct_uint32_t **destConArr, ct_int32_t *destNumCon,
		ct_uint32_t srcMaxNumCon, ct_uint32_t numSrcCells,
		ct_uint32_t destMaxNumCon, ct_uint32_t destNormNumCon,
		ct_uint32_t srcGridX, ct_uint32_t srcGridY, ct_uint32_t destGridX, ct_uint32_t destGridY,
		ct_uint32_t srcSpanOnDestGridX, ct_uint32_t srcSpanOnDestGridY,
		ct_uint32_t normConAttempts, ct_uint32_t maxConAttempts, bool needUnique,
		CRandomSFMT *randGen)
{
	bool *srcConnected;
	float gridXScaleStoD;
	float gridYScaleStoD;

	gridXScaleStoD=((float)srcGridX)/((float)destGridX);
	gridYScaleStoD=((float)srcGridY)/((float)destGridY);

	srcConnected=new bool[numSrcCells];

	cout<<"srcMaxNumCon "<<srcMaxNumCon<<" numSrcCells "<<numSrcCells<<endl;
	cout<<"destMaxNumCon "<<destMaxNumCon<<" destNormNumCon "<<destNormNumCon<<endl;
	cout<<"srcGridX "<<srcGridX<<" srcGridY "<<srcGridY<<" destGridX "<<destGridX<<" destGridY "<<destGridY<<endl;
	cout<<"srcSpanOnDestGridX "<<srcSpanOnDestGridX<<" srcSpanOnDestGridY "<<srcSpanOnDestGridY<<endl;
	cout<<"gridXScaleStoD "<<gridXScaleStoD<<" gridYScaleStoD "<<gridYScaleStoD<<endl;

	for(int i=0; i<srcMaxNumCon; i++)
	{
		int srcNumConnected;

		memset(srcConnected, false, numSrcCells*sizeof(bool));
		srcNumConnected=0;

//		cout<<"i "<<i<<endl;

		while(srcNumConnected<numSrcCells)
		{
			int srcInd;
			int srcPosX;
			int srcPosY;
			int attempts;
			int tempDestNumConLim;
			bool complete;

			srcInd=randGen->IRandom(0, numSrcCells-1);

			if(srcConnected[srcInd])
			{
				continue;
			}
//			cout<<"i "<<i<<" srcInd "<<srcInd<<" srcNumConnected "<<srcNumConnected<<endl;
			srcConnected[srcInd]=true;
			srcNumConnected++;

			srcPosX=srcInd%srcGridX;
			srcPosY=(int)(srcInd/srcGridX);

			tempDestNumConLim=destNormNumCon;

			for(attempts=0; attempts<maxConAttempts; attempts++)
			{
				int tempDestPosX;
				int tempDestPosY;
				int derivedDestInd;

				if(attempts==normConAttempts)
				{
					tempDestNumConLim=destMaxNumCon;
				}

				tempDestPosX=(int)round(srcPosX/gridXScaleStoD);
				tempDestPosY=(int)round(srcPosY/gridXScaleStoD);
//				cout<<"before rand: tempDestPosX "<<tempDestPosX<<" tempDestPosY "<<tempDestPosY<<endl;

				tempDestPosX+=round((randGen->Random()-0.5)*srcSpanOnDestGridX);//randGen->IRandom(-srcSpanOnDestGridX/2, srcSpanOnDestGridX/2);
				tempDestPosY+=round((randGen->Random()-0.5)*srcSpanOnDestGridY);//.randGen->IRandom(-srcSpanOnDestGridY/2, srcSpanOnDestGridY/2);
//				cout<<"after  rand: tempDestPosX "<<tempDestPosX<<" tempDestPosY "<<tempDestPosY<<endl;

				tempDestPosX=((tempDestPosX%destGridX+destGridX)%destGridX);
				tempDestPosY=((tempDestPosY%destGridY+destGridY)%destGridY);
//				cout<<"after mod: tempDestPosX "<<tempDestPosX<<" tempDestPosY "<<tempDestPosY<<endl;

				derivedDestInd=tempDestPosY*destGridX+tempDestPosX;
//				cout<<"derivedDestInd "<<derivedDestInd<<endl;

				if(needUnique)
				{
					bool unique=true;

					for(int j=0; j<i; j++)
					{
						if(derivedDestInd==srcConArr[srcInd][j])
						{
							unique=false;
							break;
						}
					}
					if(!unique)
					{
						continue;
					}
				}

				if(destNumCon[derivedDestInd]<tempDestNumConLim)
				{
					destConArr[derivedDestInd][destNumCon[derivedDestInd]]=srcInd;
					destNumCon[derivedDestInd]++;
					srcConArr[srcInd][i]=derivedDestInd;
					srcNumCon[srcInd]++;

					break;
				}
			}
			if(attempts==maxConAttempts)
			{
//				cout<<"incomplete connection for cell #"<<srcInd<<endl;
			}
		}
	}

	delete[] srcConnected;
}

void InNetConnectivityState::translateCommon(ct_uint32_t **pPreGLConArr, ct_int32_t *numpPreGLCon,
		ct_uint32_t **pGLPostGLConArr, ct_int32_t *numpGLPostGLCon,
		ct_uint32_t **pPreConArr, ct_int32_t *numpPreCon,
		ct_uint32_t **pPostConArr, ct_int32_t *numpPostCon,
		ct_uint32_t numPre)
{
//	cout<<"numPre "<<endl;
	for(int i=0; i<numPre; i++)
	{
		numpPreCon[i]=0;

		for(int j=0; j<numpPreGLCon[i]; j++)
		{
			ct_uint32_t glInd;

			glInd=pPreGLConArr[i][j];

			for(int k=0; k<numpGLPostGLCon[glInd]; k++)
			{
				ct_uint32_t postInd;

				postInd=pGLPostGLConArr[glInd][k];
//				cout<<"i "<<i<<" j "<<j<<" k "<<k<<" numpPreCon "<<numpPreCon[i]<<" glInd "<<glInd<<" postInd "<<postInd;
//				cout.flush();

				pPreConArr[i][numpPreCon[i]]=postInd;
				numpPreCon[i]++;

				pPostConArr[postInd][numpPostCon[postInd]]=i;
				numpPostCon[postInd]++;

//				cout<<" "<<numpGLPostGLCon[glInd]<<" "<<numpPreGLCon[i]<<" "<<numPre<<" "<<" done "<<endl;
			}
//			cout<<"k done"<<endl;
		}
	}
//	cout<<"i done"<<endl;
}

