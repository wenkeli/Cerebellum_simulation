/*
 * common.h
 *
 *  Created on: May 4, 2009
 *      Author: wen
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#define NULL 0

#define NUMBINS 200
#define NUMMF 1024
#define NUMGO 1024
#define NUMGR 1048576

#define ALLVIEWPH 1024
#define ALLVIEWPW 1000

#define SINGLEVIEWPH 500
#define SINGLEVIEWPW 1000


using namespace std;

extern unsigned short pshMF[NUMBINS][NUMMF];
extern unsigned short pshMFMax;

extern unsigned short pshGO[NUMBINS][NUMGO];
extern unsigned short pshGOMax;

extern unsigned short pshGR[NUMBINS][NUMGR];
extern unsigned short pshGRMax;


#endif /* COMMON_H_ */
