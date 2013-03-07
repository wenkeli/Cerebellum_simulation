#include "../includes/analyze.hpp"
#include <fstream>

using namespace std;
namespace po = boost::program_options;

po::options_description WeightAnalyzer::getOptions() {
    po::options_description desc("Analysis Options");
    desc.add_options()
        ("file,f", po::value<string>()->required(),
         "Saved simulator state file to analyze")
        ;
    return desc;
}

WeightAnalyzer::WeightAnalyzer(int argc, char **argv) {
    po::options_description desc = getOptions();
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), vm);
    po::notify(vm);
    cout << "Requested File: " << vm["file"].as<string>() << endl;
    // fstream inStream(savedSimState, fstream::in);
    // CBMState(inStream);
}
