#include "../../includes/environments/temporalsequence.hpp"
#include <boost/filesystem.hpp>

using namespace std;
using namespace boost::filesystem;
namespace po = boost::program_options;

po::options_description TemporalSequence::getOptions() {
    po::options_description desc("Sequence Environment Options:");
    desc.add_options()
        ("logfile", po::value<string>()->default_value("sequence.log"),"log file")
        ("duration", po::value<int>()->default_value(1500), "Duration of the sequence in ms. E.g. 1500 means each of A,B,C would be played for 500ms.")
        ("iti", po::value<int>()->default_value(2000), "Duration of inter-trial-interval in ms.")
        ;
    return desc;
}

TemporalSequence::TemporalSequence(CRandomSFMT0 *randGen, int argc, char **argv)
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

    phaseDuration = vm["duration"].as<int>();
    restTimeMSec = vm["iti"].as<int>();
    trialLen = 2 * (phaseDuration + restTimeMSec);

    assert(stateVariables.empty());
    stateVariables.push_back((StateVariable<Environment>*) (&sv_highFreq));
    stateVariables.push_back((StateVariable<Environment>*) (&sv_manual));

    assert(microzones.empty());
    microzones.push_back(&mz_0);

    for (int i=0; i<trialLen; i++) {
        mzOutputs.push_back(0.0);
    }
}

TemporalSequence::~TemporalSequence() {
    logfile.close();
}

void TemporalSequence::setupMossyFibers(CBMState *simState) {
    Environment::setupMossyFibers(simState);
    Environment::setupStateVariables(randomizeMFs, logfile);

    sv_manual.initializeManual(this, &TemporalSequence::getManualMF);
}

float* TemporalSequence::getManualMF() {
    for (int i=0; i<sv_manual.getNumMF(); i++)
        manMFs[i] = 0;

    if (phase == resting) {
        ;
    } else if (phase == fake) {
        if (timestep - phaseTransitionTime < phaseDuration / 3.0) {
            toneA();
        } else if (timestep - phaseTransitionTime < 2.0/3.0 * phaseDuration) {
            toneB();
        } else {
            toneC();
        }
    } else if (phase == real) {
        if (timestep - phaseTransitionTime < phaseDuration / 3.0) {
            toneB();
        } else if (timestep - phaseTransitionTime < 2.0/3.0 * phaseDuration) {
            toneA();
        } else {
            toneC();
        }
    } else {
        assert(false);
    }
    return manMFs;
}

float* TemporalSequence::getState() {
    for (int i=0; i<numMF; i++)
        mfFreq[i] = mfFreqRelaxed[i];

    for (uint i=0; i<stateVariables.size(); i++)
        stateVariables[i]->update();

    return &mfFreq[0];
}

void TemporalSequence::step(CBMSimCore *simCore) {
    Environment::step(simCore);

    if (timestep >= nTrials * trialLen)
        mzOutputs[timestep%trialLen] += mz_0.getMovingAverage();

    if (phase == real || phase == fake) {
        if (timestep % 100 == 0)
            logfile << timestep << " mz0MovingAvg " << mz_0.getMovingAverage() << endl;
    }

    if (phase == resting) {
        if (timestep - phaseTransitionTime >= restTimeMSec) {
            if (lastPhase == real) { 
                phase = fake;
                logfile << timestep << " Phase fake" << endl;
            } else {
                phase = real;
                logfile << timestep << " Phase real" << endl;
            }
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

bool TemporalSequence::terminated() {
    if (timestep >= nTrials * trialLen + nAdditionalTrials * trialLen) {
        printf("MZOutput: ");
        for (int i=0; i<trialLen; i++) {
            printf("%.3f, ", mzOutputs[i]/float(nAdditionalTrials));
        }
        printf("\n");
        return true;
    }
    return false;
}
