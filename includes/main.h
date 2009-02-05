/*
 * main.h
 *
 *  Created on: Feb 2, 2009
 *      Author: wen
 */

#ifndef MAIN_H_
#define MAIN_H_

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <Qt/qapplication.h>

#include "randomc.h"
#include "parameters.h"
//#include "globalvars.h"
//#include "common.h"
#include "synapsegenesis.h"
#include "genesismw.h"

//***********
//connectivity matrices
//Mossy fiber to granule cells
int conMFtoGR[NUMMF][MFGRSYNPERMF][2];

//mossy fiber to golgi cells
int conMFtoGO[NUMMF][MFGOSYNPERMF][2];

//granule cells to golgi cells
int conGRtoGO[NUMGR][GRGOSYNPERGR][2];

//golgi cells to granule cells
int conGOtoGR[NUMGO][GOGRSYNPERGO][2];

//***********
//dendritic/synaptic conductance matrices
//excitatory dendritic conductance of granule cells
float gEGR[NUMGR][DENPERGR];
//inhibitory dendritic conductance of granule cells
float gIGR[NUMGR][DENPERGR];

//excitatory synaptic conductance of golgi cells input from mossy fibers
float gEMFGO[NUMGO][MFDENPERGO];
//excitatory synaptic conductance of golgi cells input from granule cells
float gEGRGO[NUMGO][GRGOSYNPERGO];

//***********
//membrane potentials arrays
//granule cells
float vmGR[NUMGR];

//golgi cells
float vmGO[NUMGO];

//***********
//cell activity (AP) arrays
//mossy fibers
bool actMF[NUMMF];

//granule cells
bool actGR[NUMGR];

//golgi cells
bool actGO[NUMGO];

//*********************
//unknown variables*********************
int GrV[NUMGR];
bool simRunning;
float r[NUMGR];

int main(int, char **);

#endif /* MAIN_H_ */
