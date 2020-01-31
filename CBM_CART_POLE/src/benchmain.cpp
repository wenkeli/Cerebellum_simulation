/*
 * benchmain.cpp
 *
 *  Created on: Feb 19, 2010
 *      Author: wen
 */

#include "../includes/main.h"
#include "cartpole.cpp"

int main(int argc, char **argv)
{
	int trialTime;

	CRandomSFMT0 randGen(time(NULL));
	genesisCLI();
	initSim();
	initCUDA();
	Cartpole cpole;

	for(int i=0; i<100; i++)
	{
		trialTime=time(NULL);
		cout<<i<<": ";
		for(int j=0; j<5000; j++)
		{
			float theta = cpole.theta;
			float theta_dot = cpole.theta_dot;

			calcCellActivities(j, randGen);
		}
		cout<<time(NULL)-trialTime<<" seconds"<<endl;
	}
}
