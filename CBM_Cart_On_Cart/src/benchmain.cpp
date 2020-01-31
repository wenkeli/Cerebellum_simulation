/*
 * benchmain.cpp
 *
 *  Created on: Feb 19, 2010
 *      Author: wen
 */

#include "../includes/main.h"

int main(int argc, char **argv)
{
    randGen = new CRandomSFMT0(time(NULL));
    int trialTime;

    newSim();
    cout<<"initializing CUDA..."<<endl;

    cout<<"starting run"<<endl;
    for(int i=0; i<1; i++)
    {
        trialTime=time(NULL);
        cout<<i<<": ";
        for(int j=0; j<5000; j++)
        {
            calcCellActivities(j, randGen);
//			cout<<j<<endl;
        }
        cout<<time(NULL)-trialTime<<" seconds"<<endl;
    }

    cleanSim();
}
