/*
 * exprmain.cpp
 *
 *  Created on: 1-11-12
 *      Author: matthew
 */

#include "../includes/main.h"
#include <getopt.h>

int main(int argc, char **argv)
{
    randGen = new CRandomSFMT0(time(NULL));
    int c;
    int digit_optind = 0;
    char *copt = 0, *dopt = 0;
    static struct option long_options[] = {
        {"num_trials", 1, 0, 0},
        {"trial_length", 1, 0, 0},
        {"difficulty", 1, 0, 0},
        {"seed", 1, 0, 0},
        {NULL, 0, NULL, 0}
    };
    int option_index = 0;
    while ((c = getopt_long(argc, argv, "bl:", long_options, &option_index)) != -1) {
        int this_option_optind = optind ? optind : 1;
        switch (c) {
        case 0:
            if (optarg) {
                switch(option_index) {
                case 0: // num_trials
                    num_trials = strtol(optarg,NULL,10);
                    printf("Run will end after %d trial(s)\n",num_trials);
                    break;
                case 1: // trial_length
                    max_trial_length = strtol(optarg,NULL,10);
                    printf("Using max trial length: %ld\n",max_trial_length);
                    break;
                case 2: // max lower cart force
                    lower_cart_difficulty = atof(optarg);
                    printf("Lower cart difficulty: %f\n",lower_cart_difficulty);
                    break;
                case 3:
                    delete randGen;
                    int seed = atoi(optarg);
                    printf("Using random seed %d\n",seed);
                    randGen = new CRandomSFMT0(seed);
                    break;
                }
            }
            break;
        case '0':
        case '1':
        case '2':
            if (digit_optind != 0 && digit_optind != this_option_optind)
                printf ("digits occur in two different argv-elements.\n");
        digit_optind = this_option_optind;
        printf ("option %c\n", c);
        break;
        case 'l':
            cp_logfile.assign(optarg);
            break;
        case '?':
            break;
        default:
            printf ("?? getopt returned character code 0%o ??\n", c);
        }
    }
    if (optind < argc) {
        printf ("non-option ARGV-elements: ");
        while (optind < argc)
            printf ("%s ", argv[optind++]);
        printf ("\n");
    }

    newSim();
    cudaSetDevice(0);
    cout<<"initializing CUDA..."<<endl;

    int trialRunTime;
    int nRuns=0;

    cout<<"pre-run to stabilize network"<<endl;
    trialRunTime=time(NULL);
    for(int i=5000; i<10000; i++) {
        calcCellActivities(i, *randGen);
        if(i%500==0) {
            cout<<(i/500-10)*10<<" % complete"<<endl;
        }
    }
    cout<<"pre-run completed in "<<time(NULL)-trialRunTime<<" seconds"<<endl;

    while(true) {
        trialRunTime=time(NULL);
        nRuns++;

        for(short i=0; i<TRIALTIME; i++) {
            if(i%(TRIALTIME/5)==0) {
                cout<<i<<"ms ";
                cout.flush();
            }

            simPauseLock.lock();
            calcCellActivities(i, *randGen);
            simPauseLock.unlock();
        }
        pfSynWeightPCLock.lock();
        for(int i=0; i<NUMMZONES; i++) {
            zones[i]->cpyPFPCSynWCUDA();
        }
        pfSynWeightPCLock.unlock();

        cout<<endl<<"trial #"<<nRuns<<" "<<"trial run time: "<<time(NULL)-trialRunTime<<endl;
    }
    cleanSim();
}
