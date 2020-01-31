/*
 * globaltypes.h
 *
 *  Created on: Oct 29, 2010
 *      Author: consciousness
 */

#ifndef GLOBALTYPES_H_
#define GLOBALTYPES_H_
#include "parameters.h"

struct SCBCPCActs
{
	bool apSC[NUMSC];
	bool apBC[NUMBC];
	bool apPC[NUMPC];

	float vPC[NUMPC];
};

struct IONCPCActs
{
	bool apIO[NUMIO];
	bool apNC[NUMNC];
	bool apPC[NUMPC];

	float vIO[NUMIO];
	float vNC[NUMNC];
	float vPC[NUMPC];
};

//struct GranuleCell
//{
//	float v;
//	float gKCa;
//	float gE[NUMINPERGR];
//	float gEInc[NUMINPERGR];
//	float gI[NUMINPERGR];
//	float thresh;
//	float threshBase;
//};

#endif /* GLOBALTYPES_H_ */
