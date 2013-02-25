#include <QtGui/qapplication.h>

#include <boost/program_options.hpp>

#include "../includes/main.hpp"
#include "../includes/mainw.hpp"
#include "../includes/simthread.hpp"
#include "../includes/environments/environment.hpp"
#include "../includes/environments/eyelid.hpp"
#include "../includes/environments/cartpole.hpp"
#include "../includes/environments/robocup.hpp"

using namespace std;
namespace po = boost::program_options;

int main(int argc, char **argv)
{
    // Declare the supported options.
    po::options_description desc("Program Usage");
    desc.add_options()
        ("help,h", "produce help message")
        ("environment,e", po::value<string>()->required(),
         "Experimental Environment. Choices: default, eyelid, cartpole, robocup")
        ("conPF", po::value<string>()->default_value("../CBM_Params/conParams.txt"),
         "Connectivity Parameter File")
        ("actPF", po::value<string>()->default_value("../CBM_Params/actParams1.txt"),
         "Activity Parameter File")
        ("seed", po::value<int>(), "Random Seed")
        ("nogui", "Run without a gui")
        ;
    // Allow the environments to add command line args
    // TODO: Consider allowing each different environment to have its own options set
    Eyelid::addOptions(desc);
    Cartpole::addOptions(desc);
    Robocup::addOptions(desc);

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help") || !vm.count("environment")) {
        cout << desc << "\n";
        return 1;
    }

    po::notify(vm);

    int randSeed  = vm.count("seed") ? vm["seed"].as<int>() : -1;
    if (randSeed >= 0) {
        cout << "Using random seed: " << randSeed << endl;
        srand(randSeed);
    } else {
        randSeed = time(NULL);
        cout << "No seed specified. Seeding to time. Seed: " << randSeed << endl;
        srand(randSeed);
    }

    CRandomSFMT0 randGen(randSeed);

    Environment *env = NULL;
    string envStr = vm["environment"].as<string>();
    if (envStr == "default")
        env = new Environment(&randGen);
    else if (envStr == "eyelid")
        env = new Eyelid(&randGen, vm);
    else if (envStr == "cartpole")
        env = new Cartpole(&randGen, vm);
    else if (envStr == "robocup")
        env = new Robocup(&randGen, vm);

    int numMZ     = env->numRequiredMZ();
    string conPF  = vm["conPF"].as<string>();
    string actPF  = vm["actPF"].as<string>();

    if (vm.count("nogui")) {
        SimThread t(NULL, numMZ, randSeed, conPF, actPF, env);
        t.start();
        t.wait();
    } else {
        QApplication app(argc, argv);
        MainW *mainW = new MainW(NULL, numMZ, randSeed, conPF, actPF, env);
        app.setActiveWindow(mainW);
        mainW->show();

        app.connect(mainW, SIGNAL(destroyed()), &app, SLOT(quit()));
        return app.exec();
    }

    delete env;
}


