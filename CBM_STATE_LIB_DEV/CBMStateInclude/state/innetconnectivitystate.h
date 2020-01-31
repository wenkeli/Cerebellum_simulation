/*
 * innetconnectivitystate.h
 *
 *  Created on: Nov 6, 2012
 *      Author: consciousness
 */

#ifndef INNETCONNECTIVITYSTATE_H_
#define INNETCONNECTIVITYSTATE_H_


#include <fstream>
#include <sstream>
#include <string.h>
#include <math.h>
#include <vector>
#include <limits.h>

#include <CXXToolsInclude/memoryMgmt/dynamic2darray.h>
#include <CXXToolsInclude/memoryMgmt/arrayinitalize.h>
#include <CXXToolsInclude/memoryMgmt/arraycopy.h>
#include <CXXToolsInclude/fileIO/rawbytesrw.h>
#include <CXXToolsInclude/stdDefinitions/pstdint.h>
#include <CXXToolsInclude/randGenerators/sfmt.h>
//#include <cstdint>

#include "../params/connectivityparams.h"
#include "../interfaces/iinnetconstate.h"

class InNetConnectivityState : public virtual IInNetConState
{
public:
	InNetConnectivityState(ConnectivityParams *parameters, unsigned int msPerStep, int randSeed);
	InNetConnectivityState(ConnectivityParams *parameters, std::fstream &infile);
	InNetConnectivityState(const InNetConnectivityState &state);

	virtual ~InNetConnectivityState();

	virtual void writeState(std::fstream &outfile);

	virtual bool equivalent(const InNetConnectivityState &compState);

	virtual std::vector<ct_uint32_t> getpGOfromGOtoGLCon(int goN);
	virtual std::vector<ct_uint32_t> getpGOfromGLtoGOCon(int goN);
	virtual std::vector<ct_uint32_t> getpMFfromMFtoGLCon(int mfN);
	virtual std::vector<ct_uint32_t> getpGLfromGLtoGRCon(int glN);

	virtual std::vector<ct_uint32_t> getpGRfromMFtoGR(int grN);
	virtual std::vector<std::vector<ct_uint32_t> > getpGRPopfromMFtoGR();

	virtual std::vector<ct_uint32_t> getpGRfromGOtoGR(int grN);
	virtual std::vector<std::vector<ct_uint32_t> > getpGRPopfromGOtoGR();

	virtual std::vector<ct_uint32_t> getpGOfromGRtoGOCon(int goN);
	virtual std::vector<ct_uint32_t> getpGOfromGOtoGRCon(int goN);

	virtual std::vector<ct_uint32_t> getpMFfromMFtoGRCon(int mfN);

	virtual std::vector<ct_uint32_t> getpMFfromMFtoGOCon(int mfN);
	virtual std::vector<ct_uint32_t> getpGOfromMFtoGOCon(int goN);
	virtual std::vector<std::vector<ct_uint32_t> > getpGOPopfromMFtoGOCon();

	virtual std::vector<ct_uint32_t> getpGOOutGOGOCon(int goN);
	virtual std::vector<ct_uint32_t> getpGOInGOGOCon(int goN);

	virtual std::vector<ct_uint32_t> getGOIncompIndfromGRtoGO();
	virtual std::vector<ct_uint32_t> getGRIncompIndfromGRtoGO();

	virtual bool deleteGOGOConPair(int srcGON, int destGON);
	virtual bool addGOGOConPair(int srcGON, int destGON);

	//glomerulus
//	std::vector<Glomerulus> glomeruli;

	ct_uint8_t *haspGLfromMFtoGL;
	ct_uint32_t *pGLfromMFtoGL;

	ct_int32_t *numpGLfromGLtoGO;
	ct_uint32_t **pGLfromGLtoGO;

	ct_int32_t *numpGLfromGOtoGL;
	ct_uint32_t **pGLfromGOtoGL;

	ct_int32_t *numpGLfromGLtoGR;
	ct_uint32_t **pGLfromGLtoGR;


	//mossy fiber
	ct_int32_t *numpMFfromMFtoGL;
	ct_uint32_t **pMFfromMFtoGL;//[numMF][numGLOutPerMF];

	ct_int32_t *numpMFfromMFtoGR;
	ct_uint32_t **pMFfromMFtoGR;//[numMF][maxNumGROutPerMF];

	ct_int32_t *numpMFfromMFtoGO;
	ct_uint32_t **pMFfromMFtoGO;//[numMF][maxNumGOOutPerMF];

	//golgi
	ct_int32_t *numpGOfromGLtoGO;
	ct_uint32_t **pGOfromGLtoGO;//[numGO][maxNumGLInPerGO];

	ct_int32_t *numpGOfromGOtoGL;
	ct_uint32_t **pGOfromGOtoGL;//[numGO][maxNumGLOutPerGO];

	ct_int32_t *numpGOfromMFtoGO;
	ct_uint32_t **pGOfromMFtoGO;//[numGO][maxNumMFInPerGO];

	ct_int32_t *numpGOfromGOtoGR;//[numGO];
	ct_uint32_t **pGOfromGOtoGR;//[numGO][maxNumGROutPerGO];

	ct_int32_t *numpGOfromGRtoGO;
	ct_uint32_t **pGOfromGRtoGO;

	ct_int32_t *numpGOGABAInGOGO;
	ct_uint32_t **pGOGABAInGOGO;

	ct_int32_t *numpGOGABAOutGOGO;
	ct_uint32_t **pGOGABAOutGOGO;

	ct_int32_t *numpGOCoupInGOGO;
	ct_uint32_t **pGOCoupInGOGO;

	ct_int32_t *numpGOCoupOutGOGO;
	ct_uint32_t **pGOCoupOutGOGO;
	//[numGO][maxNumGOOutPerGO];//TODO: special
	//ct_int32_t goConGOOutLocal;//[2][maxNumGOOutPerGO];//TODO: how to define gogo local connectivity in the parameters?


	//granule
	ct_uint32_t *pGRDelayMaskfromGRtoBSP;//[numGR]; //TODO: add in parameters delay stuff

	ct_int32_t *numpGRfromGLtoGR;
	ct_uint32_t **pGRfromGLtoGR;//[numGR][maxNumInPerGR];

	ct_int32_t *numpGRfromGRtoGO;
	ct_uint32_t **pGRfromGRtoGO;//[maxNumGOOutPerGR][numGR];
	ct_uint32_t **pGRDelayMaskfromGRtoGO;//[maxNumGOOutPerGR][numGR];

	ct_int32_t *numpGRfromGOtoGR;
	ct_uint32_t **pGRfromGOtoGR;//[maxNumInPerGR][numGR];

	ct_int32_t *numpGRfromMFtoGR;//[numGR];
	ct_uint32_t **pGRfromMFtoGR;//[maxNumInPerGR][numGR];

protected:
	ConnectivityParams *cp;

	virtual std::vector<ct_uint32_t> getConCommon(int cellN, ct_int32_t *numpCellCon, ct_uint32_t **pCellCon);

	virtual void allocateMemory();

	virtual void stateRW(bool read, std::fstream &file);

	virtual void initializeVals();

	virtual void connectGRGL(CRandomSFMT *randGen);
	virtual void connectGOGL(CRandomSFMT *randGen);
	virtual void connectMFGL(CRandomSFMT *randGen);
	virtual void translateMFGL();
	virtual void translateGOGL();
	virtual void connectGRGO(CRandomSFMT *randGen);
	virtual void connectGOGO(CRandomSFMT *randGen);
	virtual void assignGRDelays(unsigned int msPerStep);

	virtual void connectCommon(ct_uint32_t **srcConArr, int32_t *srcNumCon,
			ct_uint32_t **destConArr, ct_int32_t *destNumCon,
			ct_uint32_t srcMaxNumCon, ct_uint32_t numSrcCells,
			ct_uint32_t destMaxNumCon, ct_uint32_t destNormNumCon,
			ct_uint32_t srcGridX, ct_uint32_t srcGridY, ct_uint32_t destGridX, ct_uint32_t destGridY,
			ct_uint32_t srcSpanOnDestGridX, ct_uint32_t srcSpanOnDestGridY,
			ct_uint32_t normConAttempts, ct_uint32_t maxConAttempts, bool needUnique,
			CRandomSFMT *randGen);

	virtual void translateCommon(ct_uint32_t **pPreGLConArr, ct_int32_t *numpPreGLCon,
			ct_uint32_t **pGLPostGLConArr, ct_int32_t *numpGLPostGLCon,
			ct_uint32_t **pPreConArr, ct_int32_t *numpPreCon,
			ct_uint32_t **pPostConArr, ct_int32_t *numpPostCon,
			ct_uint32_t numPre);

private:
	InNetConnectivityState();
};


#endif /* INNETCONNECTIVITYSTATE_H_ */
