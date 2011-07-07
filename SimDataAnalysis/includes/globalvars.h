/*
 * globalvars.h
 *
 *  Created on: Jul 7, 2011
 *      Author: consciousness
 */

#ifndef GLOBALVARS_H_
#define GLOBALVARS_H_
#include "datamodules/psh.h"
#include "common.h"

extern PSHData *mfPSH;
extern PSHData *goPSH;
extern PSHData *grPSH;
extern PSHData *scPSH;

extern PSHData *BCPSH[NUMMZONES];
extern PSHData *PCPSH[NUMMZONES];
extern PSHData *IOPSH[NUMMZONES];
extern PSHData *NCPSH[NUMMZONES];

#endif /* GLOBALVARS_H_ */
