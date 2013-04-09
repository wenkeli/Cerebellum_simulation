#ifdef BUILD_ANALYSIS
#include "../includes/analyze.hpp"
#include <fstream>
#include <map>
#include <iterator>

using namespace std;
using namespace boost::filesystem;
namespace po = boost::program_options;

po::options_description WeightAnalyzer::getOptions() {
    po::options_description desc("Analysis Options");
    desc.add_options()
        ("simfile,s", po::value<vector<string> >()->required()->multitoken(),
         "Saved simulator state files to compare")
        ("logfile,l", po::value<string>(),
         "Logfile of run. Necessary for Ordered MF analysis.")
        ;
    return desc;
}

WeightAnalyzer::WeightAnalyzer(int argc, char **argv) : R(argc, argv), plot_dir("./") {
    po::options_description desc = getOptions();
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), vm);
    po::notify(vm);

    // Read the granule-PC weights off into a vector
    vector<string> simFiles = vm["simfile"].as<vector<string> >();

    // Convert the file names into boost paths
    vector<path> simPaths;
    for (uint i=0; i<simFiles.size(); i++) {
        path p(simFiles[i]);
        assert(exists(p) && is_regular_file(p));
        simPaths.push_back(p);
    }

    if (vm.count("logfile")) {
        logfile = path(vm["logfile"].as<string>());
        assert(exists(logfile) && is_regular_file(logfile));
    }

    if (simPaths.size() == 1)
        analyzeFile(simPaths[0]);

    // Analyze all pairs of files
    for (uint i=0; i<simPaths.size(); i++) {
        for (uint j=i; j<simPaths.size(); j++) {
            if (i==j) continue;
            analyzeFiles(simPaths[i], simPaths[j]);
        }
    }
}

void WeightAnalyzer::analyzeFiles(path p1, path p2) {
    string fname1 = p1.leaf().native();
    string fname2 = p2.leaf().native();
    cout << "Analzying files: " << fname1 << ", " << fname2 << endl;
    plot_dir /= "plots_" + fname1 + "-" + fname2;
    create_directory(plot_dir);

    fstream savedSimFile1(p1.c_str(), fstream::in);
    CBMState state1(savedSimFile1);
    savedSimFile1.close();
    fstream savedSimFile2(p2.c_str(), fstream::in);
    CBMState state2(savedSimFile2);
    savedSimFile2.close();
    
    assert(state1.getConnectivityParams()->getNumGR() == state2.getConnectivityParams()->getNumGR());
    assert(state1.getNumZones() == state2.getNumZones());
    assert(state1.getConnectivityParams()->getNumMF() == state2.getConnectivityParams()->getNumMF());
    
    int numMZ = state1.getNumZones();
    int numGR = state1.getConnectivityParams()->getNumGR();
    int numMF = state1.getConnectivityParams()->getNumMF();

    vector<vector<float> > origWeights; // Original gr->pc Weights [mz][weight]
    vector<vector<float> > weightDiff; // [mz][weight]
    for (int i=0; i<numMZ; i++) {
        MZoneActivityState *mzActState1 = state1.getMZoneActStateInternal(i);
        vector<float> w1 = mzActState1->getGRPCSynWeightLinear();
        MZoneActivityState *mzActState2 = state2.getMZoneActStateInternal(i);
        vector<float> w2 = mzActState2->getGRPCSynWeightLinear();
        vector<float> diff;
        origWeights.push_back(w1);
        for (int j=0; j<numGR; j++) {
            diff.push_back(w2[j] - w1[j]);
        }
        weightDiff.push_back(diff);
    }

    // Trace these differences back to the MFs who generated them
    vector<vector<int> > numConnectedGRs; // [mz][mfNum] - How many granule cells are connected to each mf
    vector<vector<float> > mfWeightSums; // [mz][mfNum] - Sum of granule weights connected to each mf
    vector<vector<float> > mfWeightDiffSums; // [mz][mfNum] - Diff of connected granule weights
    vector<vector<float> > mfWeightDiffPercents; // [mz][mfNum] - Percentage weight change of connected granule weights
    for (int mz=0; mz<numMZ; mz++) {
        vector<int> numConnectedGR;
        vector<float> mfWeight;
        vector<float> mfWeightDiff;
        vector<float> mfWeightDiffPercent;
        for (int mf=0; mf<numMF; mf++) {
            // Make sure connectivity is the same
            assert(state1.getInnetConState()->getpMFfromMFtoGRCon(mf) ==
                   state2.getInnetConState()->getpMFfromMFtoGRCon(mf));
            float weightSum = 0;
            float weightDiffSum = 0;
            // Get the vector of granule cells connected to the mf in question
            vector<ct_uint32_t> connectedGRs = state1.getInnetConState()->getpMFfromMFtoGRCon(mf);
            for (uint j=0; j<connectedGRs.size(); j++) {
                weightSum += origWeights[mz][connectedGRs[j]];
                weightDiffSum += weightDiff[mz][connectedGRs[j]];
            }
            numConnectedGR.push_back(connectedGRs.size());
            mfWeight.push_back(weightSum);
            mfWeightDiff.push_back(weightDiffSum);
            mfWeightDiffPercent.push_back(100.0 * weightDiffSum / weightSum);
        }
        numConnectedGRs.push_back(numConnectedGR);
        mfWeightSums.push_back(mfWeight);
        mfWeightDiffSums.push_back(mfWeightDiff);
        mfWeightDiffPercents.push_back(mfWeightDiffPercent);
    }

    // Plot the MZ Percentage weight changes over all MFs
    for (int mz=0; mz<numMZ; mz++) {
        stringstream ss;
        ss << mz;
        plot_dir /= "MZ" + ss.str() + "_weight_diff_percent.pdf";
        R["weightsvec"] = mfWeightDiffPercents[mz];
        string txt =
            "library(ggplot2); "
            "data=data.frame(w=weightsvec); "
            "plot=ggplot(data=data, aes(x=1:nrow(data), y=w)) + geom_bar(stat=\"identity\") + xlab(\"MF Number\") + ylab(\"Connected Granule Percentage Weight Changes\") + labs(title = expression(\"MZ"+ss.str()+" " + fname1+" -> "+fname2+" MF Weight Changes\"));"
            "ggsave(plot,file=\""+plot_dir.c_str()+"\"); ";
        R.parseEvalQ(txt);
        plot_dir.remove_leaf();
    }

    // Parse log file for the mf indexes associated with each state variable
    ifstream ifs(logfile.c_str(), ifstream::in);
    if (ifs.good()) {
        cout << "Doing log specific analysis on logfile " << logfile.c_str() << endl;
        CRandomSFMT0 randGen(rand());
        Environment env(&randGen);
        vector<string> variableNames;
        vector<vector<int> > mfInds;
        env.readMFInds(ifs, variableNames, mfInds);

        for (uint i=0; i<variableNames.size(); i++) {
            plotMFChange(variableNames[i], mfInds[i], mfWeightDiffSums, mfWeightDiffPercents, mfWeightSums, numMZ);
        }
    }

    plot_dir.remove_leaf();
}

void WeightAnalyzer::plotMFChange(string vName, vector<int>& mfInds, vector<vector<float> >&mfWeightDiffSums,
                                  vector<vector<float> >&mfWeightDiffPercents, vector<vector<float> >&mfWeightSums,
                                  int numMZ) {
    for (int mz=0; mz<numMZ; mz++) {
        vector<float> weights;
        vector<float> wDiff;
        vector<float> wDiffPerc;
        for (uint i=0; i<mfInds.size(); i++) {
            weights.push_back(mfWeightSums[mz][mfInds[i]]);
            wDiff.push_back(mfWeightDiffSums[mz][mfInds[i]]);
            wDiffPerc.push_back(mfWeightDiffPercents[mz][mfInds[i]]);
        }

        {
            // Plot this re-ordered weight diff
            stringstream ss;
            ss << mz;
            plot_dir /= vName + "_MZ" + ss.str() + "_ordered_weight_diff_percent.pdf";
            R["weightsvec"] = wDiffPerc;
            string txt =
                "library(ggplot2); "
                "data=data.frame(w=weightsvec); "
                "plot=ggplot(data=data, aes(x=1:nrow(data), y=w)) + geom_bar(stat=\"identity\") + xlab(\"MF Number\") + ylab(\"Connected Granule Percent Weight Change\") + labs(title = expression(\"MZ"+ss.str()+" " + vName + " Ordered MF Percent Weight Changes\"));"
                "ggsave(plot,file=\""+plot_dir.c_str()+"\"); ";
            R.parseEvalQ(txt);
            plot_dir.remove_leaf();
        }

        {
            // Plot this re-ordered weights colored by diff
            stringstream ss;
            ss << mz;
            plot_dir /= vName + "_MZ" + ss.str() + "_ordered_weights.pdf";
            R["weightsvec"] = weights;
            R["diffvec"] = wDiff;
            string txt =
                "library(ggplot2); "
                "data=data.frame(w=weightsvec,WeightDiff=diffvec); "
                "plot=ggplot(data=data, aes(x=1:nrow(data), y=w, fill=WeightDiff)) + geom_bar(stat=\"identity\") + scale_fill_gradient2(low=\"darkred\",high=\"darkblue\", mid=\"white\", midpoint=0) + theme(panel.background = element_rect(fill='gray')) + xlab(\"MF Number\") + ylab(\"Sum of Connected Granule Weights\") + labs(title = expression(\"MZ"+ss.str()+" " + vName + " Ordered MF Weights\"));"
                "ggsave(plot,file=\""+plot_dir.c_str()+"\"); ";
            // theme(panel.background = element_rect(fill='black'), panel.grid.major = element_line(colour = 'green')) + 
            R.parseEvalQ(txt);
            plot_dir.remove_leaf();
        }
    }
}

void WeightAnalyzer::analyzeFile(path p) {
    cout << "Analyzing file " << p.c_str() << endl;
    plot_dir /= "plots_" + p.leaf().native() + "/";
    create_directory(plot_dir);
    grPCWeightHist(p);
    plotMFWeights(p);
    plot_dir.remove_leaf();
}

void WeightAnalyzer::grPCWeightHist(path p) {
    cout << "Creating GR-PC weight histogram for file " << p.c_str() << endl;
    fstream savedSimFile(p.c_str(), fstream::in);
    CBMState state(savedSimFile);
    savedSimFile.close();

    int numMZ = state.getNumZones();

    vector<vector<float> > grPCWeights; // [mz#][weight#]
    for (uint i=0; i<state.getNumZones(); i++) {
        MZoneActivityState *mzActState = state.getMZoneActStateInternal(i);
        vector<float> w = mzActState->getGRPCSynWeightLinear();
        grPCWeights.push_back(w);
    }

    // Weight Distributions
    for (int i=0; i<numMZ; i++) {
        stringstream ss;
        ss << i;
        plot_dir /= "MZ" + ss.str() + "_GRPC_weight_hist.pdf";
        R["weightsvec"] = grPCWeights[i];
        string txt =
            "library(ggplot2); "
            "data=data.frame(x=weightsvec); "
            "plot=qplot(x, data=data, geom=\"histogram\", binwidth=.01, xlab=\"Granule PC Weight\"); " 
            "ggsave(plot,file=\""+plot_dir.native()+"\"); ";
        R.parseEvalQ(txt);
        plot_dir.remove_leaf();
    }
}

void WeightAnalyzer::plotMFWeights(path p) {
    fstream savedSimFile(p.c_str(), fstream::in);
    CBMState state(savedSimFile);
    savedSimFile.close();
    
    int numMZ = state.getNumZones();
    int numGR = state.getConnectivityParams()->getNumGR();
    int numMF = state.getConnectivityParams()->getNumMF();

    vector<vector<float> > origWeights; // Original gr->pc Weights [mz][weight]
    for (int i=0; i<numMZ; i++) {
        MZoneActivityState *mzActState = state.getMZoneActStateInternal(i);
        vector<float> w = mzActState->getGRPCSynWeightLinear();
        origWeights.push_back(w);
    }

    // Trace these differences back to the MFs who generated them
    vector<vector<int> > numConnectedGRs; // [mz][mfNum] - How many granule cells are connected to each mf
    vector<vector<float> > mfWeightSums; // [mz][mfNum] - Sum of granule weights connected to each mf

    for (int mz=0; mz<numMZ; mz++) {
        vector<int> numConnectedGR;
        vector<float> mfWeight;
        for (int mf=0; mf<numMF; mf++) {
            float weightSum = 0;
            // Get the vector of granule cells connected to the mf in question
            vector<ct_uint32_t> connectedGRs = state.getInnetConState()->getpMFfromMFtoGRCon(mf);
            for (uint j=0; j<connectedGRs.size(); j++) {
                weightSum += origWeights[mz][connectedGRs[j]];
            }
            numConnectedGR.push_back(connectedGRs.size());
            mfWeight.push_back(weightSum);
        }
        numConnectedGRs.push_back(numConnectedGR);
        mfWeightSums.push_back(mfWeight);
    }

    CRandomSFMT0 randGen(rand());
    Environment env(&randGen);
    
    vector<string> mzNames;
    for (int i=0; i<numMZ; i++)
        mzNames.push_back("MZ" + boost::lexical_cast<string>(i));

    { // Read the better MZ names if present
        ifstream ifs(logfile.c_str(), ifstream::in);
        if (ifs.good()) {
            vector<int> mzNums;
            vector<string> names;
            env.readMZ(ifs, mzNums, names);
            for (uint i=0; i<mzNums.size(); i++) {
                mzNames[mzNums[i]] = names[i];
            }
        }
    }

    { // Plot the MF weights associated with each state variable
        ifstream ifs(logfile.c_str(), ifstream::in);
        if (ifs.good()) {
            vector<string> variableNames;
            vector<vector<int> > mfInds;
            env.readMFInds(ifs, variableNames, mfInds);

            for (uint i=0; i<variableNames.size(); i++) {
                plotMFWeights(variableNames[i], mfInds[i], mfWeightSums, numMZ, mzNames);
            }
        }
    }

    { // Plot the maximally responsive state variable values for each MF
        ifstream ifs(logfile.c_str(), ifstream::in);
        if (ifs.good()) {
            vector<string> variableNames;
            vector<vector<float> > mfResp;
            env.readMFResponses(ifs, variableNames, mfResp);

            for (uint i=0; i<variableNames.size(); i++) {
                plot_dir /= variableNames[i] + "_maximal_responses.pdf";
                R["weightsvec"] = mfResp[i];
                string txt =
                    "library(ggplot2); "
                    "data=data.frame(w=weightsvec); "
                    "plot=ggplot(data=data, aes(x=1:nrow(data), y=w)) + geom_line() + xlab(\"MF Number\") + ylab(\"Maximal Responsive Value\") + labs(title = expression(\"" + variableNames[i] + " Maximal Responses\"));"
                    "ggsave(plot,file=\""+plot_dir.c_str()+"\"); ";
                R.parseEvalQ(txt);
                plot_dir.remove_leaf();
            }
        }
    }
}

void WeightAnalyzer::plotMFWeights(string vName, vector<int>& mfInds, vector<vector<float> >& mfWeightSums,
                                   int numMZ, vector<string>& mzNames) {
    for (int mz=0; mz<numMZ; mz++) {
        vector<float> weights;
        for (uint i=0; i<mfInds.size(); i++) {
            weights.push_back(mfWeightSums[mz][mfInds[i]]);
        }

        // Plot this re-ordered weights
        stringstream ss;
        ss << mz;
        plot_dir /= vName + "_" + mzNames[mz] + "_ordered_weights.pdf";
        R["weightsvec"] = weights;
        string txt =
            "library(ggplot2); "
            "data=data.frame(w=weightsvec); "
            "plot=ggplot(data=data, aes(x=1:nrow(data), y=w)) + geom_bar(stat=\"identity\") + xlab(\"MF Number\") + ylab(\"Sum of Connected Granule Weights\") + labs(title = expression(\"" + mzNames[mz] + " " + vName + " Ordered MF Weights\"));"
            "ggsave(plot,file=\""+plot_dir.c_str()+"\"); ";
        R.parseEvalQ(txt);
        plot_dir.remove_leaf();
    }
}
#endif /* BUILD_ANALYSIS */
