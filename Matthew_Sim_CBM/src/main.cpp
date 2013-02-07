#include <QtGui/qapplication.h>

#include <boost/program_options.hpp>

#include "../includes/main.hpp"
#include "../includes/mainw.hpp"
#include "../includes/simthread.hpp"
#include "../includes/environments/environment.hpp"
#include "../includes/environments/eyelid.hpp"
#include "../includes/environments/cartpole.hpp"

using namespace std;
namespace po = boost::program_options;

int main(int argc, char **argv)
{
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("conPF", po::value<string>()->default_value("../CBM_Params/conParams.txt"),
         "Connectivity Parameter File")
        ("actPF", po::value<string>()->default_value("../CBM_Params/actParams1.txt"),
         "Activity Parameter File")
        ("seed", po::value<int>(), "Random Seed")
        ("nogui", "Run without a gui")
        ("environment", po::value<string>()->default_value("default"),
         "Experimental Environment. Choices: default, eyelid, cartpole")
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }

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
        env = new Cartpole(&randGen);

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


