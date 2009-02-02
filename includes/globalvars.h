/*
 * globalvars.h
 *
 *  Created on: Feb 2, 2009
 *      Author: wen
 */

#ifndef GLOBALVARS_H_
#define GLOBALVARS_H_

//***********
//connectivity matrices
//Mossy fiber to granule cells
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

//*********************
//unknown variables*********************
extern int GrV[NUMGR];
extern bool simRunning;
extern float r[NUMGR];

#endif /* GLOBALVARS_H_ */
