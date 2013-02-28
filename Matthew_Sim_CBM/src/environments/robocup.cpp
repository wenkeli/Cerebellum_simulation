#include "../../includes/environments/robocup.hpp"

#include "headers.h"

using namespace std;


void Robocup::addOptions(boost::program_options::options_description &desc) {
    namespace po = boost::program_options;
    desc.add_options()
        ("host", po::value<string>()->default_value("127.0.0.1"), "Robocup: IP of the server")
        ("gPort", po::value<int>()->default_value(3100), "Robocup: Server port")
        ("mPort", po::value<int>()->default_value(3200), "Robocup: Monitor port")
        ("uNum", po::value<int>()->default_value(2), "Robocup: Uniform number of the player")
        ("paramFile", po::value<string>()->default_value("/home/matthew/projects/3Dsim/agents/nao-agent/paramfiles/defaultParams.txt"), "Robocup: Path to the parameter files for the agent.")
        ("rsg", po::value<string>()->default_value("/usr/local/share/rcssserver3d/rsg/agent/nao"),
         "Robocup: Folder for the nao model")
        ("behavior", po::value<string>()->default_value("omniWalkAgent"), "Robocup: Agent behavior")
        ;
}

Robocup::Robocup(CRandomSFMT0 *randGen, boost::program_options::variables_map &vm) : Environment(randGen) {
    robosim.PrintGreeting();
    robosim.mHost      = vm["host"].as<string>();
    robosim.gPort      = vm["gPort"].as<int>(); // agent-port: The port for the server
    robosim.mPort      = vm["mPort"].as<int>(); // server-port: The port for monitor
    robosim.uNum       = vm["uNum"].as<int>();
    robosim.rsgdir     = vm["rsg"].as<string>();
    robosim.outputFile = "/dev/null"; // File where robot fitness is written
    robosim.agentType  = vm["behavior"].as<string>();
    robosim.LoadParams(vm["paramFile"].as<string>());
    assert(robosim.Init() == true);
    robosim.initializeBehavior();
}

Robocup::~Robocup() {
    robosim.Done();
}

void Robocup::setupMossyFibers(CBMState *simState) {
    Environment::setupMossyFibers(simState);
    bodyModel = robosim.behavior->getBodyModel();
    worldModel = robosim.behavior->getWorldModel();
    walkEngine = robosim.behavior->getWalkEngine();

    int numHighFreqMF = highFreqMFProportion * numMF;
    int numGyroXMF    = gyroXMFProportion * numMF;
    int numGyroYMF    = gyroYMFProportion * numMF;
    int numGyroZMF    = gyroZMFProportion * numMF;    
    int numAccelXMF   = accelXMFProportion * numMF;
    int numAccelYMF   = accelYMFProportion * numMF;
    int numAccelZMF   = accelZMFProportion * numMF;

    if (randomizeMFs) {
        vector<int> unassigned;
        for (int i=0; i<numMF; i++)
            unassigned.push_back(i);
        assignRandomMFs(unassigned, numHighFreqMF, highFreqMFs);
        assignRandomMFs(unassigned, numGyroXMF, gyroXMFs);
        assignRandomMFs(unassigned, numGyroYMF, gyroYMFs);
        assignRandomMFs(unassigned, numGyroZMF, gyroZMFs);    
        assignRandomMFs(unassigned, numAccelXMF, accelXMFs);
        assignRandomMFs(unassigned, numAccelYMF, accelYMFs);
        assignRandomMFs(unassigned, numAccelZMF, accelZMFs);
    } else {
        int m = 100;
        for (int i=0; i < numHighFreqMF; i++) highFreqMFs.push_back(m++);
        for (int i=0; i < numGyroXMF; i++) gyroXMFs.push_back(m++);
        for (int i=0; i < numGyroYMF; i++) gyroYMFs.push_back(m++);
        for (int i=0; i < numGyroZMF; i++) gyroZMFs.push_back(m++);        
        for (int i=0; i < numAccelXMF; i++) accelXMFs.push_back(m++);
        for (int i=0; i < numAccelYMF; i++) accelYMFs.push_back(m++);
        for (int i=0; i < numAccelZMF; i++) accelZMFs.push_back(m++);        
    }
}

float* Robocup::getState() {
    for (int i=0; i<numMF; i++)
        mfFreq[i] = mfFreqRelaxed[i];
    for (vector<int>::iterator it=highFreqMFs.begin(); it != highFreqMFs.end(); it++)
        mfFreq[*it] = mfFreqExcited[*it];

    VecPosition gyros = bodyModel->getGyroRates();
    VecPosition accel = bodyModel->getAccelRates();

    gaussMFAct(minGX, maxGX, gyros.getX(), gyroXMFs);
    gaussMFAct(minGY, maxGY, gyros.getY(), gyroYMFs);
    gaussMFAct(minGZ, maxGZ, gyros.getZ(), gyroZMFs);    
    gaussMFAct(minAX, maxAX, accel.getX(), accelXMFs);
    gaussMFAct(minAY, maxAY, accel.getY(), accelYMFs);
    gaussMFAct(minAZ, maxAZ, accel.getZ(), accelZMFs);    

    // // Get information about the joints
    // for (int i=0; i<HJ_NUM; i++) {
    //     bodyModel->getJointAngle(i);
    // }

    // // Get information about the effectors
    // for (int i=0; i<EFF_NUM; i++) {
    //     bodyModel->getCurrentAngle(i);
    //     bodyModel->getTargetAngle(i);
    // }

    return &mfFreq[0];
}

void Robocup::step(CBMSimCore *simCore) {
    // Setup the MZs 
    if (!shoulderPitchForward.initialized()) {
        shoulderPitchForward = Microzone(0, numNC, forceScale, forcePow, forceDecay, simCore);
        shoulderPitchBack = Microzone(1, numNC, forceScale, forcePow, forceDecay, simCore);
    }

    robosim.runStep();
    calcForce(simCore);
    deliverErrors(simCore);
}

void Robocup::deliverErrors(CBMSimCore *simCore) {
    // Standing angle = ~.5; Fallen front or back = ~.06
    float uprightAngle = float(worldModel->getMyPositionGroundTruth().getZ());
    float maxErrProb = .01;
    // Error associated with falling
    float errorProbability = min(.05f * fabsf(uprightAngle - .5f), maxErrProb);
    if (randGen->Random() < errorProbability) {
        shoulderPitchForward.deliverError();
        shoulderPitchBack.deliverError();
    }
}

void Robocup::calcForce(CBMSimCore *simCore) {
    float netShoulderPitchForce = shoulderPitchForward.getForce() - shoulderPitchBack.getForce();
    // Pitch brings the arms up/down in front of the robot (rotator cuff). [-120,120]; 0 = arms straight out in front
    float lShoulderPitch = netShoulderPitchForce - 80;
    float rShoulderPitch = netShoulderPitchForce - 80;
    // Roll brings the arms up/down to the sides of the robot (deltoids). [-1,95]; 95 = full lateral raise.
    float lShoulderRoll = 0;
    float rShoulderRoll = 0;
    walkEngine->setArms(lShoulderPitch, rShoulderPitch, lShoulderRoll, rShoulderRoll);
}

bool Robocup::terminated() {
    return robosim.behavior->finished(); 
}


