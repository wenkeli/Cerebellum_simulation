#include <QtGui/qapplication.h>

#include <boost/program_options.hpp>

#include "../includes/main.hpp"
#include "../includes/mainw.hpp"
#include "../includes/simthread.hpp"
#include "../includes/environments/environment.hpp"
#include "../includes/environments/eyelid.hpp"
#include "../includes/environments/cartpole.hpp"
#include "../includes/environments/robocup.hpp"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#ifdef BUILD_ANALYSIS
#include "../includes/analyze.hpp"
#endif

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
        ("load", po::value<string>(), "Load saved simulator state file")
        ("freeze", "Freeze the current policy by disabling synaptic plasticity")
        ;

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), vm);

    if (vm.count("help") || argc == 1 || !vm.count("environment")) {
        cout << desc << endl;
#ifdef BUILD_ANALYSIS
        cout << WeightAnalyzer::getOptions() << endl;
#endif
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

    auto_ptr<Environment> env;
    string envStr = vm["environment"].as<string>();
    if (envStr == "default")
        env.reset(new Environment(&randGen));
    else if (envStr == "eyelid")
        env.reset(new Eyelid(&randGen));
    else if (envStr == "cartpole")
        env.reset(new Cartpole(&randGen, argc, argv));
    else if (envStr == "robocup")
        env.reset(new Robocup(&randGen, argc, argv));
#ifdef BUILD_ANALYSIS
    else if (envStr == "analysis") {
        WeightAnalyzer a(argc, argv);
        return 0;
    }
#endif
    else {
        cout << "Unrecognized Environment " << envStr << endl;
        return 1;
    }

    int numMZ     = env->numRequiredMZ();
    string conPF  = vm["conPF"].as<string>();
    string actPF  = vm["actPF"].as<string>();

    // Create the simulation thread
    auto_ptr<SimThread> t;
    if (vm.count("load")) {
        std::ifstream ifs(vm["load"].as<string>().c_str());
        boost::archive::text_iarchive ia(ifs);
        ia >> (*env);
        cout << "Loaded boost archive of environment." << endl;
        t.reset(new SimThread(NULL, numMZ, randSeed, vm["load"].as<string>(), env.get()));
    } else
        t.reset(new SimThread(NULL, numMZ, randSeed, conPF, actPF, env.get()));

    if (vm.count("freeze"))
        t->disablePlasticity();

    // Create the main window if needed
    if (vm.count("nogui")) {
        t->start();
        t->wait();
    } else {
        QApplication app(argc, argv);
        MainW *mainW = new MainW(NULL, t.get(), env.get());        
        app.setActiveWindow(mainW);
        mainW->show();
        app.connect(mainW, SIGNAL(destroyed()), &app, SLOT(quit()));
        return app.exec();
    }
}


