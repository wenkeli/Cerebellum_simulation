#include "../../includes/environments/subtraction.hpp"
#include <boost/filesystem.hpp>

using namespace std;
using namespace boost::filesystem;
namespace po = boost::program_options;

po::options_description Subtraction::getOptions() {
    po::options_description desc("Subtraction Environment Options");    
    desc.add_options()
        ("logfile", po::value<string>()->default_value("subtraction.log"),"log file")
        ;
    return desc;
}

Subtraction::Subtraction(CRandomSFMT0 *randGen, int argc, char **argv)
    : Environment(randGen),
      mz_0("0", 0, 1, 1, .95),
      sv_highFreq("highFreqMFs", HIGH_FREQ, .03),
      sv_manual("manual", MANUAL, .5),
      phase(resting), phaseTransitionTime(0)
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
}

Subtraction::~Subtraction() {
    logfile.close();
}

void Subtraction::setupMossyFibers(CBMState *simState) {
    Environment::setupMossyFibers(simState);
    Environment::setupStateVariables(randomizeMFs, logfile);

    sv_manual.initializeManual(this, &Subtraction::getManualMF);
}

float* Subtraction::getManualMF() {
    for (int i=0; i<sv_manual.getNumMF(); i++)
        manMFs[i] = 0;

    if (phase == resting) {
        ;
    } else if (phase == fake) {
        toneA();
        toneB();
    } else if (phase == real) {
        toneA();
        if (timestep - phaseTransitionTime < 3000)
            toneB();
    } else {
        assert(false);
    }
    return manMFs;
}

float* Subtraction::getState() {
    for (int i=0; i<numMF; i++)
        mfFreq[i] = mfFreqRelaxed[i];

    for (uint i=0; i<stateVariables.size(); i++)
        stateVariables[i]->update();

    return &mfFreq[0];
}

void Subtraction::step(CBMSimCore *simCore) {
    Environment::step(simCore);

    if (phase == real || phase == fake) {
        if (timestep % 100 == 0)
            logfile << timestep << " mz0MovingAvg " << mz_0.getMovingAverage() << endl;                
    }

    if (phase == resting) {
        if (timestep - phaseTransitionTime >= restTimeMSec) {
            // if (lastPhase == real) { 
            //     phase = fake;
            //     logfile << timestep << " Phase fake" << endl;
            // } else {
            //     phase = real;
            //     logfile << timestep << " Phase real" << endl;
            // }
            phase = real;
            logfile << timestep << " Phase real" << endl;

            lastPhase = resting;
            phaseTransitionTime = timestep;
        }
    } else if (phase == real) {
        // Single error at the end of the "real" phase
        if (timestep - phaseTransitionTime == phaseDuration) {
            mz_0.smartDeliverError();
            logfile << timestep << " EndRealMovingAvg " << mz_0.getMovingAverage() << endl;    
        }

        if (timestep - phaseTransitionTime >= phaseDuration) {
            phase = resting;
            logfile << timestep << " Phase resting" << endl;
            lastPhase = real;
            phaseTransitionTime = timestep;
        } 
    } else if (phase == fake) {
        if (timestep - phaseTransitionTime == phaseDuration) {
            logfile << timestep << " EndFakeMovingAvg " << mz_0.getMovingAverage() << endl;    
        }

        if (timestep - phaseTransitionTime >= phaseDuration) {
            phase = resting;
            logfile << timestep << " Phase resting" << endl;
            lastPhase = fake;
            phaseTransitionTime = timestep;
        }
    } else
        assert(false);
}

bool Subtraction::terminated() {
    return timestep >= 9250000; // About 500 trials
}
