#include "../../includes/environments/robocup.hpp"

#include "headers.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

using namespace std;
namespace po = boost::program_options;

po::options_description Robocup::getOptions() {
    po::options_description desc("Robocup Options");
    desc.add_options()
        ("logfile", po::value<string>()->default_value("robocup.log"),"log file")
        ("host", po::value<string>()->default_value("127.0.0.1"), "IP of the server")
        ("gPort", po::value<int>()->default_value(3100), "Server port")
        ("mPort", po::value<int>()->default_value(3200), "Monitor port")
        ("uNum", po::value<int>()->default_value(2), "Uniform number of the player")
        ("paramFile", po::value<string>()->default_value("/home/matthew/projects/3Dsim/agents/nao-agent/paramfiles/defaultParams.txt"), "Parameter file for the agent")
        ("rsg", po::value<string>()->default_value("rsg/agent/nao"),//"/usr/local/share/rcssserver3d/rsg/agent/nao"),
         "Folder for the nao model")
        ("behavior", po::value<string>()->default_value("cerebellumAgent"), "Agent behavior")
        ("maxNumTrials", po::value<int>()->default_value(250),
         "Maximum number of trials.")
        ("simStateDir", po::value<string>()->default_value("./"), "Directory to save sim state files.")
        ;
    return desc;
}

Robocup::Robocup(CRandomSFMT0 *randGen, int argc, char **argv)
    : Environment(randGen),
      mz_hipPitchForwards("HipPitchForwards", 0, forceScale, forcePow, forceDecay), 
      //mz_hipPitchBack("HipPitchBack", 1, forceScale, forcePow, forceDecay),
      sv_highFreq("highFreqMFs", HIGH_FREQ, .03),
      sv_impactTimer("impactMFs", GAUSSIAN, .15),
      //sv_hipPitch("hipPitchMFs", GAUSSIAN, .06),
      //sv_gyrox("gyroX", GAUSSIAN, .06),      
      hpFF(0), hpBF(0), avgHipPitchForce(0)
{
    po::options_description desc = getOptions();
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), vm);
    po::notify(vm);

    robosim.PrintGreeting();
    robosim.mHost      = vm["host"].as<string>();
    robosim.gPort      = vm["gPort"].as<int>(); // agent-port: The port for the server
    robosim.mPort      = vm["mPort"].as<int>(); // server-port: The port for monitor
    robosim.uNum       = vm["uNum"].as<int>();
    robosim.rsgdir     = vm["rsg"].as<string>();
    robosim.agentType  = vm["behavior"].as<string>();
    robosim.LoadParams(vm["paramFile"].as<string>());
    maxNumTrials = vm["maxNumTrials"].as<int>();

    logfile.open(vm["logfile"].as<string>().c_str());
    saveStateDir = boost::filesystem::path(vm["simStateDir"].as<string>());
    assert(exists(saveStateDir) && is_directory(saveStateDir));

    assert(stateVariables.empty());
    stateVariables.push_back((StateVariable<Environment>*) (&sv_highFreq));
    stateVariables.push_back((StateVariable<Environment>*) (&sv_impactTimer));
    //stateVariables.push_back((StateVariable<Environment>*) (&sv_hipPitch));
    //stateVariables.push_back((StateVariable<Environment>*) (&sv_gyrox));    
    assert(microzones.empty());
    microzones.push_back(&mz_hipPitchForwards);
    //microzones.push_back(&mz_hipPitchBack);
}

Robocup::~Robocup() {
    robosim.Done();
    logfile.close();
}

void Robocup::setupMossyFibers(CBMState *simState) {
    Environment::setupMossyFibers(simState);
    Environment::setupStateVariables(randomizeMFs, logfile);

    // Connect to the server and initialize the behavior.
    // Odd things happen if this is done in the constructor,
    // so we do it here to appease the whimsical computer gods.
    assert(robosim.Init() == true);
    robosim.initializeBehavior();
    if (robosim.agentType == "cerebellumAgent") {
        behavior = (OptimizationBehaviorBalance *) robosim.behavior;
        behavior->setMaxNumShots(maxNumTrials);
    }

    bodyModel = robosim.behavior->getBodyModel();
    worldModel = robosim.behavior->getWorldModel();
    walkEngine = robosim.behavior->getWalkEngine();

    sv_impactTimer.initializeGaussian(minImpactTimerVal, maxImpactTimerVal, this, &Robocup::getTimeToImpact, 12);
    //sv_hipPitch.initializeGaussian(minHipPitch, maxHipPitch, this, &Robocup::getHipPitch);
    //sv_gyrox.initializeGaussian(minGX, maxGX, this, &Robocup::getGyroX);    
}

float* Robocup::getState() {
    for (int i=0; i<numMF; i++)
        mfFreq[i] = mfFreqRelaxed[i];

    // Update high freq state variables at all times. Others only when not in reset.
    for (uint i=0; i<stateVariables.size(); i++)
        if (stateVariables[i]->type == HIGH_FREQ || behavior->getShotPhase() != OptimizationBehaviorBalance::reset)
            stateVariables[i]->update();

    return &mfFreq[0];
}

void Robocup::step(CBMSimCore *simCore) {
    Environment::step(simCore);

    // Setup the MZs
    if (!mz_hipPitchForwards.initialized()) mz_hipPitchForwards.initialize(simCore, numNC);
    //if (!mz_hipPitchBack.initialized())     mz_hipPitchBack.initialize(simCore, numNC); 

    calcForce();

    if (timestep % cbm_steps_to_robosim_steps == 0) {
        float avgHipPitchForwardForce = hpFF / float(cbm_steps_to_robosim_steps);
        float avgHipPitchBackForce = hpBF / float(cbm_steps_to_robosim_steps);
        avgHipPitchForce = avgHipPitchForwardForce; // - avgHipPitchBackForce;
        hpFF = 0; hpBF = 0;

        // Only take actions when not in reset phase
        if (behavior->getShotPhase() != OptimizationBehaviorBalance::reset) {
            robosim.drawHipForces(avgHipPitchForwardForce, avgHipPitchBackForce);
            walkEngine->changeHips(-avgHipPitchForce, -avgHipPitchForce);

            // logfile << timestep << " TSTS " << getTimeSinceTrialStart() << " HPFF " << avgHipPitchForce << endl;

            // This is the 4 line solution strategy to this problem!!!
            // if (getTimeToImpact() > -.5 && getTimeToImpact() < .5)
            //     walkEngine->changeHips(-20, -20);
            // else 
            //     walkEngine->changeHips(0, 0);
        }

        // Let the robosim do its thing
        robosim.runStep();

        // Grab any messages from the behavior
        vector<string>* messages = behavior->getMessages();
        for (uint i=0; i<messages->size(); i++) {
            string message = (*messages)[i];
            cout << message << endl;
            logfile << timestep << " " << message << endl;
        }
        messages->clear();
    }

    if (behavior->getShotPhase() == OptimizationBehaviorBalance::prep ||
        behavior->getShotPhase() == OptimizationBehaviorBalance::recovery) {
        deliverErrors();

    }
}

void Robocup::deliverErrors() {
    float maxErrProb = .01;

    // Err based on gyro x
    float expectedVal = 0;
    float posErrorVal = 134;
    float errScale = 1.0 * maxErrProb / (posErrorVal - expectedVal);
    float value = getGyroX();
    // if (getTimeToImpact() > 0) {
    //     if (value > 0 && avgHipPitchForce < 0)
    //         value = max(value + 2.0f * avgHipPitchForce, 0.0f);
    //     if (value < 0 && avgHipPitchForce > 0) 
    //         value = min(value + 2.0f * avgHipPitchForce, 0.0f);
    // }
    float errProb = min(errScale*fabsf(value), maxErrProb);
    if (timestep % cbm_steps_to_robosim_steps == 0)
        logfile << "TTI: " << getTimeToImpact() << " ErrProb: " << errProb << endl;
    if (getTimeToImpact() < 0 && value > 0 && randGen->Random() < errProb)
        mz_hipPitchForwards.deliverError();
    // else if (getTimeToImpact() < 0 && value < 0 && randGen->Random() < errProb)
    //     mz_hipPitchBack.deliverError();

    // if (timestep % cbm_steps_to_robosim_steps == 0)
    //     printf("%5.2lf gyrox %5.2lf avgHipForce %5.2lf Value %5.2lf ErrProb %5.2lf\n", getTimeToImpact(), getGyroX(), avgHipPitchForce, value, errProb);

    // {
    //     static int numShots = 0;
    //     double tts = behavior->getTimeToShot();
    //     if (tts == 0 && behavior->getNumberShots() >= numShots) {
    //         mz_hipPitchForwards.deliverError();
    //         numShots++;
    //     }
    // }

    // {   // Error based on solution
    //     double tts = behavior->getTimeToShot();
    //     if (tts < .5 && tts > -.5) {
    //         float hfTarget = 20;
    //         float errProb = min(.001f*fabsf(hfTarget-avgHipPitchForce), maxErrProb);
    //         if (timestep % cbm_steps_to_robosim_steps == 0)
    //             logfile << "TTI: " << getTimeToImpact() << " ErrProb: " << errProb << endl;

    //         if (hfTarget - avgHipPitchForce > 0 && randGen->Random() < errProb) {
    //             // logfile << timestep << " TSTS " << getTimeSinceTrialStart() << " ERR " << endl;
    //             mz_hipPitchForwards.deliverError();
    //         }
    //     } else if (timestep % cbm_steps_to_robosim_steps == 0)
    //         logfile << "TTI: " << getTimeToImpact() << " ErrProb: 0" << endl;
    //     // else if (tts < -.5) {
    //     //     float hfTarget = 0;
    //     //     float errProb = min(.0001f*fabsf(hfTarget-avgHipPitchForce), maxErrProb);
    //     //     if (hfTarget - avgHipPitchForce < 0 && randGen->Random() < errProb)
    //     //         mz_hipPitchBack.deliverError();
    //     // }
    // }
    
    // // COM based error
    // cout << getCOMX() << endl;
    // float comx = getCOMX() - 9;
    // float errProb = min(.002f * fabsf(comx), maxErrProb);
    // // if (comx > 0 && timestep % cbm_steps_to_robosim_steps == 0)
    // //     logfile << "TTI: " << getTimeToImpact() << " ErrProb: " << errProb << endl;
    // if (randGen->Random() < errProb) {
    //     if (comx > 0)
    //         mz_hipPitchForwards.deliverError();
    // }
    //     else
    //         mz_hipPitchBack.deliverError();
    // }

    // Error based on acceleration
    // VecPosition accel = bodyModel->getAccelRates();
    // //cout << accel.getX() << endl;
    // float errorProbability = min(.005f * fabsf(accel.getX()), maxErrProb);
    // if (timestep % cbm_steps_to_robosim_steps == 0)
    //     logfile << "TTI: " << getTimeToImpact() << " ErrProb: " << errorProbability << endl;
    // // Accel X < 0 Indicates a backwards lean
    // if (accel.getX() < 0 && randGen->Random() < errorProbability)
    //     mz_hipPitchForwards.deliverError();
    // Accel X > 0 Indicates a forwards lean
    // errorProbability = min(.001f * fabsf(accel.getX()), maxErrProb);
    // if (accel.getX() > 0 && randGen->Random() < errorProbability)
    //     mz_hipPitchBack.deliverError();

        // Error based on deviation from upright
    // {   // Standing angle = ~.5; Fallen front or back = ~.06
    //     VecPosition accel = bodyModel->getAccelRates();
    //     float uprightAngle = float(worldModel->getMyPositionGroundTruth().getZ());
    //     float errorProbability = min(.05f * fabsf(uprightAngle - .5f), maxErrProb);
    //     if (accel.getX() < 0 && randGen->Random() < errorProbability)
    //         mz_hipPitchForwards.deliverError();
    //     // if (accel.getX() > 0 && randGen->Random() < errorProbability)
    //     //     mz_hipPitchBack.deliverError();
    // }
}

void Robocup::calcForce() {
    float hipPitchForwardsForce = mz_hipPitchForwards.getForce();

    //float hipPitchBackForce = mz_hipPitchBack.getForce();
    hpFF += hipPitchForwardsForce;
    //hpBF += hipPitchBackForce;
}

bool Robocup::terminated() {
    return robosim.behavior->finished(); 
}

