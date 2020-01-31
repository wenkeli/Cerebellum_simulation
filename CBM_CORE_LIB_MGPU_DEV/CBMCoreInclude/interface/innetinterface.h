/*
 * innetinterface.h
 *
 *  Created on: Oct 1, 2012
 *      Author: consciousness
 */

#ifndef INNETINTERFACE_H_
#define INNETINTERFACE_H_

#include <CXXToolsInclude/stdDefinitions/pstdint.h>

class InNetInterface
{
public:
	virtual ~InNetInterface();

	virtual void setGIncGRtoGO(float inc)=0;
	virtual void resetGIncGRtoGO()=0;

	virtual const ct_uint8_t* exportAPSC()=0;
	virtual const ct_uint8_t* exportAPGO()=0;
	virtual const ct_uint8_t* exportAPGR()=0;

	virtual const ct_uint8_t* exportHistMF()=0;

	virtual const ct_uint32_t* exportAPBufMF()=0;
	virtual const ct_uint32_t* exportAPBufGR()=0;
	virtual const ct_uint32_t* exportAPBufGO()=0;
	virtual const ct_uint32_t* exportAPBufSC()=0;

	virtual const float* exportVmGR()=0;
	virtual const float* exportVmGO()=0;
	virtual const float* exportVmSC()=0;

	virtual const float* exportGESumGR()=0;
	virtual const float* exportGISumGR()=0;

	virtual const ct_uint32_t* exportSumGRInputGO()=0;
	virtual const float* exportSumGOInputGO()=0;
	virtual const float* exportGOOutSynScaleGOGO()=0;
};


#endif /* INNETINTERFACE_H_ */
