#include "../../includes/environments/identity.hpp"
#include <boost/filesystem.hpp>

using namespace std;
using namespace boost::filesystem;
namespace po = boost::program_options;

po::options_description Identity::getOptions() {
    po::options_description desc("Identity Environment Options");    
    desc.add_options()
        ("logfile", po::value<string>()->default_value("identity.log"),"log file")
        ;
    return desc;
}

Identity::Identity(CRandomSFMT0 *randGen, int argc, char **argv)
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

    for (int i=0; i<2000; i++) {
        mzOutputs[i] = 0.0;
    }
}

Identity::~Identity() {
    logfile.close();
}

void Identity::setupMossyFibers(CBMState *simState) {
    Environment::setupMossyFibers(simState);
    Environment::setupStateVariables(randomizeMFs, logfile);

    sv_manual.initializeManual(this, &Identity::getManualMF);
}

float* Identity::getManualMF() {
    for (int i=0; i<sv_manual.getNumMF(); i++)
        manMFs[i] = 0;

    if (phase == resting) {
        ;
    } else if (phase == real) {
        A();
    } else if (phase == fake) {
        B();
    } else {
        assert(false);
    }
    return manMFs;
}

float* Identity::getState() {
    for (int i=0; i<numMF; i++)
        mfFreq[i] = mfFreqRelaxed[i];

    for (uint i=0; i<stateVariables.size(); i++)
        stateVariables[i]->update();

    return &mfFreq[0];
}

void Identity::step(CBMSimCore *simCore) {
    Environment::step(simCore);

    mzOutputs[timestep%2000] += mz_0.getMovingAverage();

    if (timestep % 10 == 0) {
        logfile << timestep%2000 << " mz0MovingAvg " << mz_0.getMovingAverage() << endl;
    }

    if (phase == resting) {
        if (timestep - phaseTransitionTime >= restTimeMSec) {
            phase = static_cast<state>(lastPhase + 1);
            if (phase > fake) phase = real;
            logfile << timestep << " Starting Phase " << phase << endl;
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
            lastPhase = phase;
            phase = resting;
            logfile << timestep << " Starting Phase resting" << endl;
            phaseTransitionTime = timestep;
        } 
    } else if (phase == fake) {
        if (timestep - phaseTransitionTime == phaseDuration) {
            logfile << timestep << " EndFakeMovingAvg " << mz_0.getMovingAverage() << endl;    
        }

        if (timestep - phaseTransitionTime >= phaseDuration) {
            lastPhase = phase;
            phase = resting;
            logfile << timestep << " Starting Phase resting" << endl;
            phaseTransitionTime = timestep;
        }
    } else
        assert(false);
}

bool Identity::terminated() {
    if (timestep >= 20000) {
        printf("MZOutput: [");
        for (int i=0; i<2000; i++) {
            printf("%lf, ", mzOutputs[i]/10.0);
        }
        printf("\n");
        return true;
    }
    return false;
}
