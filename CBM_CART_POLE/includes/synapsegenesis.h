/*
 * synapsegenesis.h
 *
 * this declares the function that generate the connections
 *
 *  Created on: Jan 26, 2009
 *      Author: Wen
 */

#ifndef SYNAPSEGENESIS_H_
#define SYNAPSEGENESIS_H_

#include <QtGui/QTextBrowser>
#include <QtCore/QString>
#include "common.h"
#include "globalvars.h"


//populate the connectivity matrices
//QTextBrowser is the object to output all status message to
void assignGRDelays(stringstream &);

void genesisGUI(QTextBrowser *);

void genesisCLI();

//void newGenesis(QTextBrowser *);

#endif /* SYNAPSEGENESIS_H_ */
