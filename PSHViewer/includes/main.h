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

unsigned short pshMF[NUMBINS][NUMMF];
unsigned short pshMFMax;

unsigned short pshGO[NUMBINS][NUMGO];
unsigned short pshGOMax;

unsigned short pshGR[NUMBINS][NUMGR];
unsigned short pshGRMax;

int main(int argc, char **argv);

#endif /* MAIN_H_ */
