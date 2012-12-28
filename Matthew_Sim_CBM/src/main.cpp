#include <boost/program_options.hpp>
#include <fstream>

#include <CBMStateInclude/interfaces/cbmstate.h>
#include <CBMCoreInclude/interface/cbmsimcore.h>

#include "../includes/main.h"

using namespace std;
namespace po = boost::program_options;

int main(int argc, char **argv)
{
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("conPF", po::value<string>()->default_value("conParams.txt"),
         "Connectivity Parameter File")
        ("actPF", po::value<string>()->default_value("actParams.txt"),
         "Activity Parameter File")
        ("seed", po::value<int>()->default_value(0), "Random Seed")
        // ("numT", po::value<int>()->default_value(50000), "Number of Trials")
        // ("iti", po::value<int>()->default_value(5000),"Inter-Trial Interval")
        ("numMZ", po::value<int>()->default_value(1),"Number of Microzones")
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }

    int numMZ     = vm["numMZ"].as<int>();
    int randSeed  = vm["seed"].as<int>();
    assert(numMZ == 1); // TODO: Random Seeds for multiple MZs

    // Load the parameter files
    ifstream conPF(vm["conPF"].as<string>().c_str());
    ifstream actPF(vm["actPF"].as<string>().c_str());

    // Create the simulation
    CBMState simState(actPF, conPF, numMZ, randSeed, &randSeed, &randSeed);
    CBMSimCore simCore(&simState, &randSeed);

    conPF.close();
    actPF.close();

    // calcSimActivity
    // simulation->updateMFInput(apMF);
    // simulation->calcActivity();
}
