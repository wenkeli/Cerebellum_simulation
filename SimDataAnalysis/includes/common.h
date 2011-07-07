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
#include <math.h>
#include <ctime>

#ifndef NULL
#define NULL 0
#endif

#ifdef EYELID
#define NUMMZONES 1
#endif

#ifdef CARTPOLE
#define NUMMZONES 2
#endif

//end simulation state parameters

#endif /* COMMON_H_ */
