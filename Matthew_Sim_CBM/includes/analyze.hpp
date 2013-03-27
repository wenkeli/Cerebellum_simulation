#ifdef BUILD_ANALYSIS
#ifndef ANALYZE_HPP_
#define ANALYZE_HPP_

#include <string>
#include <vector>
#include <assert.h>

#include <QtCore/QThread>

#include <CBMStateInclude/interfaces/cbmstate.h>
#include <CBMCoreInclude/interface/cbmsimcore.h>
#include <CBMToolsInclude/poissonregencells.h>
#include <CBMVisualInclude/acttemporalview.h>

#include "environments/environment.hpp"

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <RInside.h>


class WeightAnalyzer
{
public:
    WeightAnalyzer(int argc, char **argv);
    ~WeightAnalyzer() {};

    static boost::program_options::options_description getOptions();

    // Analyzes the weights in a single file
    void analyzeFile(std::string fname);

    // Looks for differences in the weights between the two files
    void analyzeFiles(std::string fname1, std::string fname2);

    // Creates a histogram of the GR-PC weights for each microzone
    void grPCWeightHist(std::string fname);

    void plotMFChange(std::string vName, int *mfInds, int numMFInds,
                      std::vector<std::vector<float> >&mfWeightDiffSums,
                      std::vector<std::vector<float> >&mfWeightDiffPercents,
                      std::vector<std::vector<float> >&mfWeightSums,                      
                      int numMZ);

protected:
    RInside R;

    boost::filesystem::path plot_dir; // directory to store the plots
};
#endif /* ANALYZE_HPP_ */
#endif /* BUILD_ANALYSIS */
