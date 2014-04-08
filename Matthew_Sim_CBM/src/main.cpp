#include <QtGui/qapplication.h>

#include <boost/program_options.hpp>

#include "../includes/main.hpp"
#include "../includes/mainw.hpp"
#include "../includes/simthread.hpp"
#include "../includes/environments/environment.hpp"
#include "../includes/environments/eyelid.hpp"
#include "../includes/environments/cartpole.hpp"
#include "../includes/environments/robocup.hpp"
#include "../includes/environments/audio.hpp"
#include "../includes/environments/test.hpp"
#include "../includes/environments/pid.hpp"
#include "../includes/environments/xor.hpp"
#include "../includes/environments/subtraction.hpp"
#include "../includes/environments/identity.hpp"
#include "../includes/environments/conjunction.hpp"
#include "../includes/environments/disjunction.hpp"
#include "../includes/environments/negation.hpp"
#include "../includes/environments/nand.hpp"

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
         "Experimental Environment. Choices: default, eyelid, cartpole, robocup, audio, test, xor, subtraction, pid")
        ("conPF", po::value<string>()->default_value("../CBM_Params/conParams.txt"),
         "Connectivity Parameter File")
        ("actPF", po::value<string>()->default_value("../CBM_Params/actParams1.txt"),
         "Activity Parameter File")
        ("seed", po::value<int>(), "Random Seed")
        ("nogui", "Run without a gui")
        ("load", po::value<string>(), "Load saved simulator state file")
        ("freeze", "Freeze the current policy by disabling synaptic plasticity")
        ("analyze", "Run analysis mode")
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
        cout << Audio::getOptions() << endl;
        cout << Test::getOptions() << endl;        
        cout << Xor::getOptions() << endl;
        cout << Subtraction::getOptions() << endl;
        cout << PID::getOptions() << endl;        
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

    string conPF  = vm["conPF"].as<string>();
    string actPF  = vm["actPF"].as<string>();

    CRandomSFMT0 randGen(randSeed);

    Environment *env;
    string envStr = vm["environment"].as<string>();
    if (envStr == "default")
        env = new Environment(&randGen);
    else if (envStr == "eyelid")
        env = new Eyelid(&randGen);
    else if (envStr == "cartpole")
        env = new Cartpole(&randGen, argc, argv);
    else if (envStr == "robocup")
        env = new Robocup(&randGen, argc, argv);
    else if (envStr == "audio")
        env = new Audio(&randGen, argc, argv);
    else if (envStr == "test")
        env = new Test(&randGen, argc, argv);
    else if (envStr == "xor")
        env = new Xor(&randGen, argc, argv);
    else if (envStr == "subtraction") {
        env = new Subtraction(&randGen, argc, argv);
        // conPF = "../CBM_Params/sim_GOHGO/fig4/conParams_GOGOI.txt";
        // actPF = "../CBM_Params/sim_GOHGO/fig4/actParams_GOGOI_DEP_Matt.txt";
    } else if (envStr == "PID")
        env = new PID(&randGen, argc, argv);
    else if (envStr == "Identity")
        env = new Identity(&randGen, argc, argv);
    else if (envStr == "Conjunction")
        env = new Conjunction(&randGen, argc, argv);
    else if (envStr == "Disjunction")
        env = new Disjunction(&randGen, argc, argv);
    else if (envStr == "Negation")
        env = new Negation(&randGen, argc, argv);
    else if (envStr == "Nand")
        env = new Nand(&randGen, argc, argv);
    else {
        cout << "Unrecognized Environment " << envStr << endl;
        return 1;
    }

#ifdef BUILD_ANALYSIS
    if (vm.count("analyze")) {
        cout << "Entering Analysis Mode" << endl;
        WeightAnalyzer a(env, argc, argv);
        return 0;
    }
#endif

    int numMZ     = env->numRequiredMZ();
    cout << "Using connectivity param file: " << conPF << endl;
    cout << "Using activity param file: " << actPF << endl;

    // Create the simulation thread
    auto_ptr<SimThread> t;
    if (vm.count("load")) {
        CBMState *simState = loadSim(vm["load"].as<string>().c_str(), *env);
        t.reset(new SimThread(NULL, numMZ, randSeed, simState, env));
    } else
        t.reset(new SimThread(NULL, numMZ, randSeed, conPF, actPF, env));

    if (vm.count("freeze"))
        t->disablePlasticity();

    // Create the main window if needed
    if (vm.count("nogui")) {
        t->start();
        t->wait();
    } else {
        QApplication app(argc, argv);
        MainW *mainW = new MainW(NULL, t.get(), env);
        app.setActiveWindow(mainW);
        mainW->show();
        app.connect(mainW, SIGNAL(destroyed()), &app, SLOT(quit()));
        return app.exec();
    }

    delete env;
}


