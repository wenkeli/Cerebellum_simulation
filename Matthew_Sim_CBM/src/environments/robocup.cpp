#include "../../includes/environments/robocup.hpp"

#include "headers.h"

using namespace std;


void Robocup::addOptions(boost::program_options::options_description &desc) {
    namespace po = boost::program_options;
    desc.add_options()
        // ("host", po::value<string>()->default_value("127.0.0.1"), "Robocup: IP of the server")
        // ("port", po::value<int>()->default_value(20), "Robocup: Server port")
        // ("rsg", po::value<string>()->default_value("127.0.0.1"), "Robocup: Folder for the nao model")
        // ("team", po::value<string>()->default_value("127.0.0.1"), "Robocup: Name of team")
        ;
}


Robocup::Robocup(CRandomSFMT0 *randGen, boost::program_options::variables_map &vm) : Environment(randGen) {
    robosim.PrintGreeting();
    robosim.gPort = 3100; // agent-port: The port for the server
    robosim.mPort = 3200; // server-port: The port for monitor
    robosim.uNum = 2;
    robosim.LoadParams("/home/matthew/projects/3Dsim/agents/nao-agent/paramfiles/defaultParams.txt");
    robosim.rsgdir = "/usr/local/share/rcssserver3d/rsg/agent/nao";
    robosim.outputFile = "/tmp/out.txt";
    assert(robosim.Init() == true);
    robosim.agentType = "omniWalkAgent";
    robosim.initializeBehavior();
}

Robocup::~Robocup() {
    robosim.Done();
}

void Robocup::setupMossyFibers(CBMState *simState) {
    Environment::setupMossyFibers(simState);
    bodyModel = robosim.behavior->getBodyModel();
}

float* Robocup::getState() {
    VecPosition gyros = bodyModel->getGyroRates();
    cout << "Gyro: " << gyros.getX() << " " << gyros.getY() << " " << gyros.getZ() << endl;
    VecPosition accel = bodyModel->getAccelRates();
    cout << "Accel: " << accel.getX() << " " << accel.getY() << " " << accel.getZ() << endl;

    // Get information about the joints
    for (int i=0; i<HJ_NUM; i++) {
        bodyModel->getJointAngle(i);
    }

    // Get information about the effectors
    for (int i=0; i<EFF_NUM; i++) {
        bodyModel->getCurrentAngle(i);
        bodyModel->getTargetAngle(i);
    }

    return &mfFreq[0];
}

void Robocup::step(CBMSimCore *simCore) {
    // TODO: Take the outputs from the cerebellum and add them to torques or otherwise
    // TODO: Deliver errors to the cerebellum
    robosim.runStep();
}

bool Robocup::terminated() {
    return false; //robosim.gLoop; 
}


