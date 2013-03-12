#include "../includes/analyze.hpp"
#include <fstream>
#include <map>
#include <iterator>

using namespace std;
namespace po = boost::program_options;

po::options_description WeightAnalyzer::getOptions() {
    po::options_description desc("Analysis Options");
    desc.add_options()
        ("file,f", po::value<vector<string> >()->required()->multitoken(),
         "Saved simulator state files to compare")
        ;
    return desc;
}

WeightAnalyzer::WeightAnalyzer(int argc, char **argv) : R(argc, argv), plot_dir("plots/") {
    po::options_description desc = getOptions();
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), vm);
    po::notify(vm);

    // Read the granule-PC weights off into a vector
    vector<string> simFiles = vm["file"].as<vector<string> >();
    if (simFiles.size() == 1)
        analyzeFile(simFiles[0]);

    // Analyze all pairs of files
    for (int i=0; i<simFiles.size(); i++) {
        for (int j=i; j<simFiles.size(); j++) {
            if (i==j) continue;
            analyzeFiles(simFiles[i], simFiles[j]);
        }
    }

}

void WeightAnalyzer::analyzeFiles(string fname1, string fname2) {
    cout << "Analzying files: " << fname1 << ", " << fname2 << endl;

    fstream savedSimFile1(fname1.c_str(), fstream::in);
    CBMState state1(savedSimFile1);
    savedSimFile1.close();
    fstream savedSimFile2(fname2.c_str(), fstream::in);
    CBMState state2(savedSimFile2);
    savedSimFile2.close();
    
    assert(state1.getConnectivityParams()->getNumGR() == state2.getConnectivityParams()->getNumGR());
    assert(state1.getNumZones() == state2.getNumZones());
    assert(state1.getConnectivityParams()->getNumMF() == state2.getConnectivityParams()->getNumMF());
    
    int numMZ = state1.getNumZones();
    int numGR = state1.getConnectivityParams()->getNumGR();
    int numMF = state1.getConnectivityParams()->getNumMF();

    vector<vector<float> > weightDiff; // [mz][weight]
    for (int i=0; i<numMZ; i++) {
        MZoneActivityState *mzActState1 = state1.getMZoneActStateInternal(i);
        vector<float> w1 = mzActState1->getGRPCSynWeightLinear();
        MZoneActivityState *mzActState2 = state2.getMZoneActStateInternal(i);
        vector<float> w2 = mzActState2->getGRPCSynWeightLinear();
        vector<float> diff;
        for (int j=0; j<numGR; j++) {
            diff.push_back(w1[j] - w2[j]);
        }
        weightDiff.push_back(diff);
    }

    // Plot the granule weight diffs
    for (int mz=0; mz<numMZ; mz++) {
        stringstream ss;
        ss << mz;
        string filename = plot_dir + fname1 +"-"+ fname2 + "_MZ" + ss.str() + "_gr_weight_diff_hist.pdf";
        const vector<float> w(weightDiff[mz]);
        R["weightsvec"] = w;
        string txt =
            "library(ggplot2); "
            "data=data.frame(x=weightsvec); "
            "plot=qplot(x, data=data, geom=\"histogram\", binwidth=.01, xlab=\"Granule PC Weight Diff\") + labs(title = expression(\"MZ"+ss.str()+" " + fname1+" -> "+fname2+" GR-PC Weight Diff Hist\")); " 
            "ggsave(plot,file=\""+filename+"\"); ";
        R.parseEvalQ(txt);
    }    

    // Trace these differences back to the MFs who generated them
    vector<vector<float> > mfWeightSums; // [mz][mfNum]
    for (int mz=0; mz<numMZ; mz++) {
        vector<float> mfWeightDiff;
        for (int mf=0; mf<numMF; mf++) {
            float weightSum = 0;
            // Make sure connectivity is the same
            assert(state1.getInnetConState()->getpMFfromMFtoGRCon(mf) ==
                   state2.getInnetConState()->getpMFfromMFtoGRCon(mf));
            // Get the vector of granule cells connected to the mf in question
            vector<ct_uint32_t> connectedGRs = state1.getInnetConState()->getpMFfromMFtoGRCon(mf);
            for (int j=0; j<connectedGRs.size(); j++) {
                weightSum += weightDiff[mz][connectedGRs[j]];
            }
            mfWeightDiff.push_back(weightSum);
        }
        mfWeightSums.push_back(mfWeightDiff);
    }

    // Plots the MZ weight changes
    for (int mz=0; mz<numMZ; mz++) {
        stringstream ss;
        ss << mz;
        string filename = plot_dir + fname1 +"-"+ fname2 + "_MZ" + ss.str() + "_weight_diff.pdf";
        const vector<float> w(mfWeightSums[mz]);
        R["weightsvec"] = w;
        string txt =
            "library(ggplot2); "
            "data=data.frame(w=weightsvec); "
            "plot=ggplot(data=data, aes(x=1:nrow(data), y=w, fill=1:nrow(data))) + geom_bar(stat=\"identity\") + xlab(\"MF Number\") + ylab(\"Sum of Connected Granule Weight Changes\") + labs(title = expression(\"MZ"+ss.str()+" " + fname1+" -> "+fname2+" MF Weight Changes\"));"
            "ggsave(plot,file=\""+filename+"\"); ";
        R.parseEvalQ(txt);
    }
}

void WeightAnalyzer::analyzeFile(string fname) {
    cout << "Analyzing file " << fname << endl;
    grPCWeightHist(fname);
}

void WeightAnalyzer::grPCWeightHist(string fname) {
    cout << "Creating GR-PC weight histogram for file " << fname << endl;
    fstream savedSimFile(fname.c_str(), fstream::in);
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
        string filename = plot_dir + fname + "_MZ" + ss.str() + "_GRPC_weight_hist.pdf";
        cout << "Saving weight histogram to file " << filename << endl;
        const vector<float> w(grPCWeights[i]);
        R["weightsvec"] = w;
        string txt =
            "library(ggplot2); "
            "data=data.frame(x=weightsvec); "
            "plot=qplot(x, data=data, geom=\"histogram\", binwidth=.01, xlab=\"Granule PC Weight\"); " 
            "ggsave(plot,file=\""+filename+"\"); ";
        R.parseEvalQ(txt);
    }
}
