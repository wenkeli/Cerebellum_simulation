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

#include <boost/program_options.hpp>


class WeightAnalyzer
{
public:
    WeightAnalyzer(int argc, char **argv);
    ~WeightAnalyzer() {};

    static boost::program_options::options_description getOptions();
};

#endif /* ANALYZE_HPP_ */
