/*
 * simexternalec.h
 *
 *  Created on: Aug 15, 2011
 *      Author: consciousness
 */

#ifndef SIMEXTERNALEC_H_
#define SIMEXTERNALEC_H_

#include <fstream>

using namespace std;

class SimExternalEC
{
public:
	SimExternalEC(ifstream &infile);
private:
	SimExternalEC();

	float timeStepSize;
	float tsUnitInS;

};

#endif /* SIMEXTERNALEC_H_ */
