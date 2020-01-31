/*
 * common.h
 *
 *  Created on: Oct 27, 2009
 *      Author: wen
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <string>
#include <string.h>

//use the intel math library if using intel compiler
#ifdef INTELCC
#include <mathimf.h>
#else //otherwise use standard math library
#include <math.h>
#endif

#include <vector>

#include "randomc.h"
#include "sfmt.h"
#include "parameters.h"
#include "globaltypes.h"

#ifndef NULL
#define NULL 0
#endif

using namespace std;

#endif /* COMMON_H_ */
