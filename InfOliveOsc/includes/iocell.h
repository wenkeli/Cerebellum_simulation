/*
 * iocell.h
 *
 *  Created on: Apr 8, 2009
 *      Author: wen
 */

#ifndef IOCELL_H_
#define IOCELL_H_

#include "common.h"

class IOCell
{
	public:
		void calcActivity();
	private:
		double calcIhMax(double v){return 1/(1+exp((v+75)/8));};
		double calcIhTau(double v){return exp(0.033*(v+70))/(0.011*(1+exp(0.083*(v+70))));};
		double calcIhDelta(double v){return (calcIhMax(vm)-Ih)/calcIhTau(vm);};


		double Ih;
		double vm;
};

#endif /* IOCELL_H_ */
