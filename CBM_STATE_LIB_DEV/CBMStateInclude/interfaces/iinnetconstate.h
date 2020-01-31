/*
 * iinnetconstate.h
 *
 *  Created on: Dec 14, 2012
 *      Author: consciousness
 */

#ifndef IINNETCONSTATE_H_
#define IINNETCONSTATE_H_

#include <vector>
#include <CXXToolsInclude/stdDefinitions/pstdint.h>

class IInNetConState
{
public:
	virtual ~IInNetConState();
	virtual std::vector<ct_uint32_t> getpGOfromGOtoGLCon(int goN)=0;
	virtual std::vector<ct_uint32_t> getpGOfromGLtoGOCon(int goN)=0;
	virtual std::vector<ct_uint32_t> getpMFfromMFtoGLCon(int mfN)=0;
	virtual std::vector<ct_uint32_t> getpGLfromGLtoGRCon(int glN)=0;

	virtual std::vector<ct_uint32_t> getpGRfromMFtoGR(int grN)=0;
	virtual std::vector<std::vector<ct_uint32_t> > getpGRPopfromMFtoGR()=0;

	virtual std::vector<ct_uint32_t> getpGRfromGOtoGR(int grN)=0;
	virtual std::vector<std::vector<ct_uint32_t> > getpGRPopfromGOtoGR()=0;

	virtual std::vector<ct_uint32_t> getpGOfromGRtoGOCon(int goN)=0;
	virtual std::vector<ct_uint32_t> getpGOfromGOtoGRCon(int goN)=0;

	virtual std::vector<ct_uint32_t> getpMFfromMFtoGRCon(int mfN)=0;

	virtual std::vector<ct_uint32_t> getpMFfromMFtoGOCon(int mfN)=0;
	virtual std::vector<ct_uint32_t> getpGOfromMFtoGOCon(int goN)=0;
	virtual std::vector<std::vector<ct_uint32_t> > getpGOPopfromMFtoGOCon()=0;

	virtual std::vector<ct_uint32_t> getpGOOutGOGOCon(int goN)=0;
	virtual std::vector<ct_uint32_t> getpGOInGOGOCon(int goN)=0;

	virtual std::vector<ct_uint32_t> getGOIncompIndfromGRtoGO()=0;
	virtual std::vector<ct_uint32_t> getGRIncompIndfromGRtoGO()=0;

	virtual bool deleteGOGOConPair(int srcGON, int destGON)=0;
	virtual bool addGOGOConPair(int srcGON, int destGON)=0;
};

#endif /* IINNETCONSTATE_H_ */
