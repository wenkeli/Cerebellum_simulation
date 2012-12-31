#include <boost/program_options.hpp>
#include <fstream>

#include <QtGui/qapplication.h>

#include <CBMStateInclude/interfaces/cbmstate.h>
#include <CBMCoreInclude/interface/cbmsimcore.h>
#include <CBMToolsInclude/poissonregencells.h>
#include <CBMVisualInclude/acttemporalview.h>

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
        ("seed", po::value<int>(), "Random Seed")
        ("numMZ", po::value<int>()->default_value(1),"Number of Microzones")
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }

    int numMZ = vm["numMZ"].as<int>();
    int randSeed;
    vector<int> mzoneCRSeeds; // Seed for each MZ's connectivity
    vector<int> mzoneARSeeds; // Seed for each MZ's activity
    if (vm.count("seed")) {
        randSeed  = vm["seed"].as<int>();
        cout << "Using random seed: " << randSeed << endl;
        srand(randSeed);
    } else {
        randSeed = time(NULL);
        cout << "No seed specified. Seeding to time. Seed: " << randSeed << endl;
        srand(randSeed);
    }

    // Create MZ seeds from the main random seed
    for (int i=0; i<numMZ; i++) {
        mzoneCRSeeds.push_back(rand());
        mzoneARSeeds.push_back(rand());
    }
    cout << "mzoneCRSeeds: ";
    for (int i=0; i<numMZ; i++)
        cout << mzoneCRSeeds[i] << " ";
    cout << endl;
    cout << "mzoneARSeeds: ";
    for (int i=0; i<numMZ; i++)
        cout << mzoneARSeeds[i] << " ";
    cout << endl;

    // Load the parameter files
    ifstream conPF(vm["conPF"].as<string>().c_str());
    ifstream actPF(vm["actPF"].as<string>().c_str());

    // Create the simulation
    CBMState simState(actPF, conPF, numMZ, randSeed,
                      &mzoneCRSeeds[0], &mzoneARSeeds[0]);
    CBMSimCore simCore(&simState, &randSeed);

    conPF.close();
    actPF.close();

    // Setup the Mossy Fibers
    int numMF = simState.getConnectivityParams()->getNumMF();
    float threshDecayTau = 4.0f; // Rate of decay = 1-exp(-msPerTS/threshDecayTau)
    float msPerTimeStep = 1.0f;
    PoissonRegenCells mfs(numMF, randSeed, threshDecayTau, msPerTimeStep);
    float mfFreq[numMF]; // MF Firings Frequencies

    // Create the input visualization
    QApplication app(argc, argv);
    int numGO = simState.getConnectivityParams()->getNumGO();
    int windowWidth  = 800;
    int windowHeight = numGO;
    ActTemporalView inputNetTView(numGO, 1, windowWidth, windowWidth, windowHeight, Qt::white, "inputNet");

    // app.setQuitOnLastWindowClosed(true);
    // app.setActiveWindow(&inputNetTView);
    // inputNetTView.show();
    // inputNetTView.update();    
    //app.exec();

    CRandomSFMT0 randGen(randSeed);

    for (int simStep=0; simStep<1000000; simStep++) {
        if (simStep % 1000 == 0)
            cout << "." << flush;
        if (simStep % 10000 == 0)
            cout << endl;

        // Calculate MF Activity
        for (int i=0; i<numMF; i++) {
            float contextFreqMin = 30;
            float contextFreqMax = 60;
            mfFreq[i] = randGen.Random()*(contextFreqMax-contextFreqMin)+contextFreqMin;            
        }
        const ct_uint8_t *apMF = mfs.calcActivity(&mfFreq[0]);
        
        // Calculate Sim Activity
        simCore.updateMFInput(apMF);
        simCore.calcActivity();

        // Display the activity
        const ct_uint8_t* apGO = simCore.getInputNet()->exportAPGO();
        vector<ct_uint8_t> tmp;
        for (int i=0; i<numGO; i++) {
            tmp.push_back(apGO[i]);
        }
        inputNetTView.drawRaster(tmp, simStep);
	inputNetTView.show();
	inputNetTView.update();
    }
}
