/*
 * imzoneactstate.h
 *
 *  Created on: Mar 4, 2013
 *      Author: consciousness
 */

#ifndef IMZONEACTSTATE_H_
#define IMZONEACTSTATE_H_

#include <vector>
#include <CXXToolsInclude/stdDefinitions/pstdint.h>

class IMZoneActState
{
public:
	virtual ~IMZoneActState();
	virtual std::vector<float> getGRPCSynWeightLinear()=0;
	virtual void resetGRPCSynWeight()=0;
//	virtual std::vector<std::vector<float> > getGRPCSynWeight()=0;
};


#endif /* IMZONEACTSTATE_H_ */
