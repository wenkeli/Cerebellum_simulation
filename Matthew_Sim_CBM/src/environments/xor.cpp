#include "../../includes/environments/xor.hpp"
#include <boost/filesystem.hpp>

using namespace std;
using namespace boost::filesystem;
namespace po = boost::program_options;

po::options_description Xor::getOptions() {
    po::options_description desc("Xor Environment Options");    
    desc.add_options()
        ("logfile", po::value<string>()->default_value("xor.log"),"log file")
        ;
    return desc;
}

Xor::Xor(CRandomSFMT0 *randGen, int argc, char **argv)
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

    for (int i=0; i<mzOutputLen; i++) {
        mzOutputs[i] = 0.0;
    }
}

Xor::~Xor() {
    logfile.close();
}

void Xor::setupMossyFibers(CBMState *simState) {
    Environment::setupMossyFibers(simState);
    Environment::setupStateVariables(randomizeMFs, logfile);

    sv_manual.initializeManual(this, &Xor::getManualMF);
}

float* Xor::getManualMF() {
    for (int i=0; i<sv_manual.getNumMF(); i++)
        manMFs[i] = 0;

    if (phase == resting) {
        ;
    } else if (phase == AB) {
        // A();
        // B();
        if (timestep % 2 == 0) 
            A();
        else 
            B();
    } else if (phase == AnotB) {
        // A();
        // notB();
        if (timestep % 2 == 0) 
            A();
        else
            notB();
    } else if (phase == notAnotB) {
        // notA();
        // notB();
        if (timestep % 2 == 0) 
            notA();
        else
            notB();
    } else if (phase == notAB) {
        // notA();
        // B();
        if (timestep % 2 == 0) 
            notA();
        else
            B();
    } else {
        assert(false);
    }
    return manMFs;
}

float* Xor::getState() {
    for (int i=0; i<numMF; i++)
        mfFreq[i] = mfFreqRelaxed[i];

    for (uint i=0; i<stateVariables.size(); i++)
        stateVariables[i]->update();

    return &mfFreq[0];
}

void Xor::step(CBMSimCore *simCore) {
    Environment::step(simCore);

    mzOutputs[timestep%mzOutputLen] += mz_0.getMovingAverage();

    if (phase != resting) {
        if (timestep % 100 == 0)
            logfile << timestep << " mz0MovingAvg " << mz_0.getMovingAverage() << endl;                
    }

    if (phase == resting) {
        if (timestep - phaseTransitionTime >= restTimeMSec) {
            phase = static_cast<state>(lastPhase + 1);
            if (phase > notAB) phase = AB;
            logfile << timestep << " Phase " << phase << endl;
            lastPhase = resting;
            phaseTransitionTime = timestep;
        }
    } else if (phase == AB || phase == notAnotB) {
        // Single error at the end of the "real" phase
        if (timestep - phaseTransitionTime == phaseDuration) {
            mz_0.smartDeliverError();
            logfile << timestep << " EndRealMovingAvg " << mz_0.getMovingAverage() << endl;    
        }

        if (timestep - phaseTransitionTime >= phaseDuration) {
            lastPhase = phase;
            phase = resting;
            logfile << timestep << " Phase resting" << endl;
            phaseTransitionTime = timestep;
        } 
    } else if (phase == AnotB || phase == notAB) {
        if (timestep - phaseTransitionTime == phaseDuration) {
            logfile << timestep << " EndFakeMovingAvg " << mz_0.getMovingAverage() << endl;    
        }

        if (timestep - phaseTransitionTime >= phaseDuration) {
            lastPhase = phase;
            phase = resting;
            logfile << timestep << " Phase resting" << endl;
            phaseTransitionTime = timestep;
        }
    } else
        assert(false);
}

bool Xor::terminated() {
    // return timestep >= 1000000;

    if (timestep >= 10 * mzOutputLen) {
        printf("MZOutput: [");
        for (int i=0; i<mzOutputLen; i++) {
            printf("%lf, ", mzOutputs[i]/10.0);
        }
        printf("\n");
        return true;
    }
    return false;
}
