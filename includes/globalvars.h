/*
 * globalvars.h
 *
 *  Created on: Feb 2, 2009
 *      Author: wen
 */

#ifndef GLOBALVARS_H_
#define GLOBALVARS_H_
#include "parameters.h"
#include <vector>

//***********
//connectivity matrices
//Mossy fiber to granule cells
//last index: [0]=gr cell number, [1]=gr dendrite number
extern int conMFtoGR[NUMMF][MFGRSYNPERMF][2];

//mossy fiber to golgi cells
extern int conMFtoGO[NUMMF][MFGOSYNPERMF][2];

//granule cells to golgi cells
extern int conGRtoGO[NUMGR][GRGOSYNPERGR][2];

//golgi cells to granule cells
extern int conGOtoGR[NUMGO][GOGRSYNPERGO][2];

//***********
//dendritic/synaptic conductance matrices
//excitatory dendritic conductance of granule cells
extern float gEGR[NUMGR][DENPERGR];
//inhibitory dendritic conductance of granule cells
extern float gIGR[NUMGR][DENPERGR];

//excitatory synaptic conductance of golgi cells input from mossy fibers
extern float gEMFGO[NUMGO][MFDENPERGO];
//excitatory synaptic conductance of golgi cells input from granule cells
extern float gEGRGO[NUMGO][GRGOSYNPERGO];

//***********
//membrane potentials arrays
//granule cells
extern float vmGR[NUMGR];

//golgi cells
extern float vmGO[NUMGO];

//***********
//cell activity (AP) arrays
//mossy fibers
extern bool actMF[NUMMF];

//granule cells
extern bool actGR[NUMGR];

//golgi cells
extern bool actGO[NUMGO];

//***************
//status variables
extern bool connsMade;

//*********************
//unknown variables*********************
extern int GrV[NUMGR];
extern bool simRunning;
extern float r[NUMGR];

//*****************
//debug variables
extern std::vector<int> incompGRs;


#endif /* GLOBALVARS_H_ */
