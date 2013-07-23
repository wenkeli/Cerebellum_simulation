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
    WeightAnalyzer(Environment *env, int argc, char **argv);
    ~WeightAnalyzer() {};

    static boost::program_options::options_description getOptions();

    void AnalyzeRobocupLogFile(boost::filesystem::path logpath);
    void AnalyzeAudioLogFile(boost::filesystem::path logpath);
    
    // Analyzes the weights in a single file
    void analyzeFile(boost::filesystem::path p);

    // Looks for differences in the weights between the two files
    void analyzeFiles(boost::filesystem::path p1, boost::filesystem::path p2);

    // Creates a histogram of the GR-PC weights for each microzone
    void grPCWeightHist(boost::filesystem::path p);

    // Plots the MF weights for each state variable
    void plotMFWeights(boost::filesystem::path p);

    void plotMFWeights(std::string vName, std::vector<int>& mfInds,
                       std::vector<std::vector<float> >& mfWeightSums,
                       int numMZ, std::vector<std::string>& mzNames);

    void plotMFChange(std::string vName, std::vector<std::string> mzNames,
                      std::vector<int> &mfInds,
                      std::vector<std::vector<float> >&mfWeightDiffSums,
                      std::vector<std::vector<float> >&mfWeightDiffPercents,
                      std::vector<std::vector<float> >&mfWeightSums,                      
                      int numMZ);

protected:
    Environment *env;

    RInside R;

    boost::filesystem::path plot_dir; // directory to store the plots
    boost::filesystem::path logfile; // logfile to read from (optional)
};
#endif /* ANALYZE_HPP_ */
#endif /* BUILD_ANALYSIS */
