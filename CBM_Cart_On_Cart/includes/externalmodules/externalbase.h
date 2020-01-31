/*
 * externalbase.h
 *
 *  Created on: May 25, 2011
 *      Author: consciousness
 */

#ifndef EXTERNALBASE_H_
#define EXTERNALBASE_H_

#include "../common.h"
#include "../errorinputmodules/errorinputbase.h"
#include "../outputmodules/outputbase.h"

class BaseExternal
{
public:
	BaseExternal(float ts, float tsus, BaseErrorInput **ems, BaseOutput **oms, unsigned int numMods);
	BaseExternal(ifstream &infile, BaseErrorInput **ems, BaseOutput **oms, unsigned int numMods);
	virtual ~BaseExternal();

	virtual void run()=0;

	virtual void exportState(ofstream &outfile);
protected:
	float timeStepSize;
	float tsUnitInS;

	BaseErrorInput **errorModules;
	BaseOutput **outputModules;
	unsigned int numModules;
private:
	BaseExternal();
};


#endif /* EXTERNALBASE_H_ */
