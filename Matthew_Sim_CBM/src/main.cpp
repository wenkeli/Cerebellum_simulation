#include <QtGui/qapplication.h>

#include <boost/program_options.hpp>

#include "../includes/main.h"

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
    int randSeed  = vm.count("seed") ? vm["seed"].as<int>() : -1;
    string conPF  = vm["conPF"].as<string>();
    string actPF  = vm["actPF"].as<string>();

    QApplication app(argc, argv);
    MainW *mainW = new MainW(&app, NULL, numMZ, randSeed, conPF, actPF);
    app.setActiveWindow(mainW);
    mainW->show();

    app.connect(mainW, SIGNAL(destroyed()), &app, SLOT(quit()));
    return app.exec();
}


