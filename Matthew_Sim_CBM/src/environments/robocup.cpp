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
        //("outfile", po::value<string>()->default_value("/tmp/robosim.log"), "Robocup: output file")
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
}

float* Robocup::getState() {
    VecPosition gyros = bodyModel->getGyroRates();
    //cout << "Gyro: " << gyros.getX() << " " << gyros.getY() << " " << gyros.getZ() << endl;
    VecPosition accel = bodyModel->getAccelRates();
    //cout << "Accel: " << accel.getX() << " " << accel.getY() << " " << accel.getZ() << endl;

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
    return robosim.behavior->finished(); 
}


