/*
 * main.h
 *
 *  Created on: Jan 5, 2010
 *      Author: wen
 */

#ifndef MAIN_H_
#define MAIN_H_

#include <QtGui/QApplication>

#include "common.h"
#include "mainw.h"
#include "datamodules/psh.h"

PSHData *mfPSH;
PSHData *goPSH;
PSHData *grPSH;
PSHData *scPSH;

PSHData *bcPSH[NUMMZONES];
PSHData *pcPSH[NUMMZONES];
PSHData *ioPSH[NUMMZONES];
PSHData *ncPSH[NUMMZONES];


int main(int argc, char **argv);

#endif /* MAIN_H_ */
