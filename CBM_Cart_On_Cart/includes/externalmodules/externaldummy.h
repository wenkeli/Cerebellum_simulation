/*
 * externaldummy.h
 *
 *  Created on: Jun 9, 2011
 *      Author: consciousness
 */

#ifndef EXTERNALDUMMY_H_
#define EXTERNALDUMMY_H_

#include "externalbase.h"
#include "../common.h"

class DummyExternal : public BaseExternal
{
public:
	DummyExternal(float ts, float tsus, BaseErrorInput **ems, BaseOutput **oms, unsigned int numMods);
	DummyExternal(ifstream &infile, BaseErrorInput **ems, BaseOutput **oms, unsigned int numMods);
	void run();

	void exportState(ofstream &outfile);
private:
	DummyExternal();
};

#endif /* EXTERNALDUMMY_H_ */
