#include "../../includes/environments/negation.hpp"
#include <boost/filesystem.hpp>

using namespace std;
using namespace boost::filesystem;
namespace po = boost::program_options;

po::options_description Negation::getOptions() {
    po::options_description desc("Negation Environment Options");    
    desc.add_options()
        ("logfile", po::value<string>()->default_value("negation.log"),"log file")
        ;
    return desc;
}

Negation::Negation(CRandomSFMT0 *randGen, int argc, char **argv)
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

    for (int i=0; i<trialLen; i++) {
        mzOutputs[i] = 0.0;
    }
}

Negation::~Negation() {
    logfile.close();
}

void Negation::setupMossyFibers(CBMState *simState) {
    Environment::setupMossyFibers(simState);
    Environment::setupStateVariables(randomizeMFs, logfile);

    sv_manual.initializeManual(this, &Negation::getManualMF);
}

float* Negation::getManualMF() {
    for (int i=0; i<sv_manual.getNumMF(); i++)
        manMFs[i] = 0;

    if (phase == resting) {
        A();
    } else if (phase == notA) {
        ;
    } else {
        cout << "Got unexpected phase: " << phase << endl;
        assert(false);
    }
    return manMFs;
}

float* Negation::getState() {
    for (int i=0; i<numMF; i++)
        mfFreq[i] = mfFreqRelaxed[i];

    for (uint i=0; i<stateVariables.size(); i++)
        stateVariables[i]->update();

    return &mfFreq[0];
}

void Negation::step(CBMSimCore *simCore) {
    Environment::step(simCore);

    if (timestep >= nTrials * trialLen)
        mzOutputs[timestep%trialLen] += mz_0.getMovingAverage();

    if (phase == resting) {
        if (timestep - phaseTransitionTime >= restTimeMSec) {
            phase = notA;
            logfile << timestep << " Starting Phase " << phase << endl;
            lastPhase = resting;
            phaseTransitionTime = timestep;
        }
    } else if (phase == notA) {
        if (timestep % 500 == 0)
            mz_0.smartDeliverError();

        if (timestep - phaseTransitionTime == phaseDuration) {
            logfile << timestep << " EndRealMovingAvg " << mz_0.getMovingAverage() << endl;    
        }

        if (timestep - phaseTransitionTime >= phaseDuration) {
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

bool Negation::terminated() {
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
