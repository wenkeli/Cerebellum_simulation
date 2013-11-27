#include "../../includes/environments/disjunction.hpp"
#include <boost/filesystem.hpp>

using namespace std;
using namespace boost::filesystem;
namespace po = boost::program_options;

po::options_description Disjunction::getOptions() {
    po::options_description desc("Disjunction Environment Options");    
    desc.add_options()
        ("logfile", po::value<string>()->default_value("disjunction.log"),"log file")
        ;
    return desc;
}

Disjunction::Disjunction(CRandomSFMT0 *randGen, int argc, char **argv)
    : Environment(randGen),
      mz_0("0", 0, 1, 1, .95),
      sv_highFreq("highFreqMFs", HIGH_FREQ, .03),
      sv_manual("manual", MANUAL, .5),
      phase(resting), lastPhase(resting), phaseTransitionTime(0)
{
    po::options_description desc = getOptions();
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), vm);
    po::notify(vm);
    
    logfile.open(vm["logfile"].as<string>().c_str());

    assert(stateVariables.empty());
    stateVariables.push_back((StateVariable<Environment>*) (&sv_highFreq));
    stateVariables.push_back((StateVariable<Environment>*) (&sv_manual));

    assert(microzones.empty());
    microzones.push_back(&mz_0);

    for (int i=0; i<3000; i++) {
        mzOutputs[i] = 0.0;
    }
}

Disjunction::~Disjunction() {
    logfile.close();
}

void Disjunction::setupMossyFibers(CBMState *simState) {
    Environment::setupMossyFibers(simState);
    Environment::setupStateVariables(randomizeMFs, logfile);

    sv_manual.initializeManual(this, &Disjunction::getManualMF);
}

float* Disjunction::getManualMF() {
    for (int i=0; i<sv_manual.getNumMF(); i++)
        manMFs[i] = 0;

    if (phase == resting) {
        ;
    } else if (phase == justA) {
        A();
    } else if (phase == justB) {
        B();
    } else if (phase == AB) {
        A();
        B();
    } else {
        cout << "Got unexpected phase: " << phase << endl;
        assert(false);
    }
    return manMFs;
}

float* Disjunction::getState() {
    for (int i=0; i<numMF; i++)
        mfFreq[i] = mfFreqRelaxed[i];

    for (uint i=0; i<stateVariables.size(); i++)
        stateVariables[i]->update();

    return &mfFreq[0];
}

void Disjunction::step(CBMSimCore *simCore) {
    Environment::step(simCore);

    mzOutputs[timestep%3000] += mz_0.getMovingAverage();

    if (timestep % 10 == 0) {
        logfile << timestep%3000 << " mz0MovingAvg " << mz_0.getMovingAverage() << endl;
    }

    if (phase == resting) {
        if (timestep - phaseTransitionTime >= restTimeMSec) {
            phase = static_cast<state>(lastPhase + 1);
            if (phase > justB) phase = justA;
            logfile << timestep << " Starting Phase " << phase << endl;
            lastPhase = resting;
            phaseTransitionTime = timestep;
        }
    } else if (phase == AB) {
        // Single error at the end of the "real" phase
        if (timestep - phaseTransitionTime == phaseDuration) {
            mz_0.smartDeliverError();
            logfile << timestep << " EndABMovingAvg " << mz_0.getMovingAverage() << endl;    
        }

        if (timestep - phaseTransitionTime >= phaseDuration) {
            lastPhase = phase;
            phase = resting;
            logfile << timestep << " Starting Phase resting" << endl;
            phaseTransitionTime = timestep;
        } 
    } else if (phase == justA) {
        if (timestep - phaseTransitionTime == phaseDuration) {
            logfile << timestep << " EndAMovingAvg " << mz_0.getMovingAverage() << endl;    
        }

        if (timestep - phaseTransitionTime >= phaseDuration) {
            mz_0.smartDeliverError();
            lastPhase = phase;
            phase = resting;
            logfile << timestep << " Starting Phase resting" << endl;
            phaseTransitionTime = timestep;
        }
    } else if (phase == justB) {
        if (timestep - phaseTransitionTime == phaseDuration) {
            logfile << timestep << " EndBMovingAvg " << mz_0.getMovingAverage() << endl;    
        }

        if (timestep - phaseTransitionTime >= phaseDuration) {
            mz_0.smartDeliverError();
            lastPhase = phase;
            phase = resting;
            logfile << timestep << " Starting Phase resting" << endl;
            phaseTransitionTime = timestep;
        }
    } else {
        cout << "Got unexpected phase: " << phase << endl;        
        assert(false);
    }
}

bool Disjunction::terminated() {
    // return timestep >= 1000000;

    if (timestep >= 10 * 3000) {
        printf("MZOutput: [");
        for (int i=0; i<3000; i++) {
            printf("%lf, ", mzOutputs[i]/10.0);
        }
        printf("\n");
        return true;
    }
    return false;
}
