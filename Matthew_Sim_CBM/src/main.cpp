#include <QtGui/qapplication.h>

#include <boost/program_options.hpp>

#include "../includes/main.hpp"
#include "../includes/mainw.hpp"
#include "../includes/simthread.hpp"
#include "../includes/environments/environment.hpp"
#include "../includes/environments/eyelid.hpp"
#include "../includes/environments/cartpole.hpp"
#include "../includes/environments/robocup.hpp"

#include "../includes/analyze.hpp"

using namespace std;
namespace po = boost::program_options;

int main(int argc, char **argv)
{
    // Declare the supported options.
    po::options_description desc("Usage: " + string(argv[0]) + " -e[environment] <OPTIONS>");
    desc.add_options()
        ("help,h", "produce help message")
        ("environment,e", po::value<string>()->required(),
         "Experimental Environment. Choices: default, eyelid, cartpole, robocup, analysis")
        ("conPF", po::value<string>()->default_value("../CBM_Params/conParams.txt"),
         "Connectivity Parameter File")
        ("actPF", po::value<string>()->default_value("../CBM_Params/actParams1.txt"),
         "Activity Parameter File")
        ("seed", po::value<int>(), "Random Seed")
        ("nogui", "Run without a gui")
        ;

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), vm);

    if (vm.count("help") || argc == 1 || !vm.count("environment")) {
        cout << desc << endl;
        cout << WeightAnalyzer::getOptions() << endl;
        cout << Cartpole::getOptions() << endl;
        cout << Robocup::getOptions() << endl;
        return 0;
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
        env = new Eyelid(&randGen);
    else if (envStr == "cartpole")
        env = new Cartpole(&randGen, argc, argv);
    else if (envStr == "robocup")
        env = new Robocup(&randGen, argc, argv);
    else if (envStr == "analysis") {
        WeightAnalyzer a(argc, argv);
        return 1;
    }

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


