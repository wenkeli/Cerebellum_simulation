#ifdef BUILD_ANALYSIS
#include "../includes/analyze.hpp"
#include <fstream>
#include <map>
#include <iterator>
#include <sys/stat.h>
#include <sys/types.h>

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
    plot_dir = "plots_" + fname1 + "-" + fname2 + "/";
    mkdir(plot_dir.c_str(),S_IRWXU|S_IRGRP|S_IXGRP);

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

    vector<vector<float> > origWeights; // Original gr->pc Weights [mz][weight]
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
        origWeights.push_back(w1);
    }

    // Plot the granule weight diffs
    for (int mz=0; mz<numMZ; mz++) {
        stringstream ss;
        ss << mz;
        string filename = plot_dir + "MZ" + ss.str() + "_gr_weight_diff_hist.pdf";
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
    vector<vector<float> > mfWeightDiffSums; // [mz][mfNum]
    vector<vector<float> > mfWeightDiffPercents; // [mz][mfNum]    
    for (int mz=0; mz<numMZ; mz++) {
        vector<float> mfWeight;
        vector<float> mfWeightDiff;
        vector<float> mfWeightDiffPercent;
        for (int mf=0; mf<numMF; mf++) {
            float weightSum = 0;
            float weightDiffSum = 0;
            // Make sure connectivity is the same
            assert(state1.getInnetConState()->getpMFfromMFtoGRCon(mf) ==
                   state2.getInnetConState()->getpMFfromMFtoGRCon(mf));
            // Get the vector of granule cells connected to the mf in question
            vector<ct_uint32_t> connectedGRs = state1.getInnetConState()->getpMFfromMFtoGRCon(mf);
            for (int j=0; j<connectedGRs.size(); j++) {
                weightSum += origWeights[mz][connectedGRs[j]];
                weightDiffSum += weightDiff[mz][connectedGRs[j]];
            }
            mfWeight.push_back(weightSum);
            mfWeightDiff.push_back(weightDiffSum);
            mfWeightDiffPercent.push_back(100.0 * weightDiffSum / weightSum);
        }
        mfWeightSums.push_back(mfWeight);
        mfWeightDiffSums.push_back(mfWeightDiff);
        mfWeightDiffPercents.push_back(mfWeightDiffPercent);
    }

    // Plots the MZ weight changes
    for (int mz=0; mz<numMZ; mz++) {
        stringstream ss;
        ss << mz;
        string filename = plot_dir + "MZ" + ss.str() + "_weight_diff_percent.pdf";
        const vector<float> w(mfWeightDiffPercents[mz]);
        R["weightsvec"] = w;
        string txt =
            "library(ggplot2); "
            "data=data.frame(w=weightsvec); "
            "plot=ggplot(data=data, aes(x=1:nrow(data), y=w)) + geom_bar(stat=\"identity\") + xlab(\"MF Number\") + ylab(\"Connected Granule Percentage Weight Changes\") + labs(title = expression(\"MZ"+ss.str()+" " + fname1+" -> "+fname2+" MF Weight Changes\"));"
            "ggsave(plot,file=\""+filename+"\"); ";
        R.parseEvalQ(txt);
    }

    // Plot original gr-pc weights colored by weight change
    // for (int mz=0; mz<numMZ; mz++) {
    //     stringstream ss;
    //     ss << mz;
    //     string filename = plot_dir + fname1 +"-"+ fname2 + "_MZ" + ss.str() + "_weight.pdf";
    //     const vector<float> w(mfWeightSums[mz]);
    //     const vector<float> x(mfWeightDiffSums[mz]);
    //     R["weightsvec"] = w;
    //     R["diffvec"] = x;
    //     string txt =
    //         "library(ggplot2); "
    //         "data=data.frame(w=weightsvec,WeightChange=diffvec); "
    //         "plot=ggplot(data=data, aes(x=1:nrow(data), y=w, fill=WeightChange)) + geom_bar(stat=\"identity\") + scale_fill_gradient(low=\"darkred\",high=\"darkblue\") + xlab(\"MF Number\") + ylab(\"Sum of Connected Granule Weight Changes\") + labs(title = expression(\"MZ"+ss.str()+" " + fname1+" -> "+fname2+" MF Weights\"));"
    //         "ggsave(plot,file=\""+filename+"\"); ";
    //     R.parseEvalQ(txt);
    // }

    // Plot the MZ weight changes associated with each state variable
    int highFreqMFInds[] = { 992, 845, 925, 1667, 270, 1753, 933, 1803, 1112, 585, 271, 1657, 1341, 1143, 30, 1623, 702, 1025, 1860, 795, 1477, 130, 1118, 1757, 21, 898, 1501, 1271, 120, 570, 1394, 885, 760, 1045, 708, 1244, 1337, 1761, 745, 438, 1448, 283, 616, 1586, 1331, 1961, 98, 23, 1194, 174, 1102, 958, 1409, 906, 1705, 1853, 1869, 1521, 296, 596, 1800 };
    int poleVelMFInds[] = { 1979, 547, 1163, 228, 260, 1942, 606, 136, 1348, 1127, 214, 2002, 1909, 231, 1567, 1904, 647, 1290, 2043, 342, 1940, 1198, 161, 1990, 344, 1491, 288, 768, 128, 297, 424, 343, 735, 1203, 443, 71, 729, 155, 907, 1410, 1956, 1005, 705, 341, 1371, 1464, 1814, 1510, 532, 1616, 1493, 1006, 1959, 498, 968, 625, 636, 321, 1075, 1932, 965, 1164, 1512, 1018, 986, 920, 1948, 846, 664, 1991, 816, 476, 1675, 1263, 2042, 1038, 1019, 676, 1601, 1269, 703, 1238, 1912, 216, 1110, 390, 43, 572, 20, 867, 1422, 35, 1481, 44, 1423, 1141, 573, 942, 380, 1697, 1987, 156, 1085, 1765, 569, 659, 1669, 908, 1397, 1294, 559, 1678, 1702, 121, 1801, 204, 1744, 1974, 1610, 132, 878, 1955 };
    int poleAngMFInds[] = { 361, 49, 1763, 1639, 1652, 1509, 1390, 621, 28, 666, 399, 1254, 275, 1040, 1531, 1017, 493, 962, 577, 1927, 211, 1949, 1121, 1532, 1842, 1710, 595, 749, 869, 1704, 1706, 1013, 1377, 946, 326, 1062, 1721, 246, 2011, 1413, 713, 1646, 634, 1103, 1081, 1199, 1739, 1179, 1201, 1426, 1701, 715, 1335, 1001, 1686, 89, 766, 1065, 854, 1455, 1880, 783, 1602, 1058, 1787, 1947, 781, 80, 281, 1712, 1583, 1095, 591, 305, 164, 691, 431, 737, 1579, 1044, 1366, 129, 167, 542, 776, 529, 178, 966, 1844, 452, 1794, 1115, 1035, 323, 1120, 1052, 656, 19, 1333, 977, 34, 1898, 465, 2003, 553, 1708, 1220, 39, 644, 688, 1773, 677, 555, 1323, 1368, 385, 1232, 945, 1881, 1791, 226, 615 };
    int cartVelMFInds[] = { 880, 604, 1870, 1055, 51, 469, 1180, 453, 411, 964, 1172, 1124, 1215, 315, 785, 1906, 1825, 747, 1887, 522, 563, 1542, 397, 1152, 247, 770, 928, 46, 1994, 1666, 1161, 264, 975, 813, 24, 510, 1857, 1277, 904, 1827, 1310, 1799, 731, 1270, 1917, 286, 1330, 892, 61, 48, 712, 1156, 978, 10, 239, 1309, 405, 1682, 483, 76, 1338, 763, 877, 244, 1454, 1407, 233, 752, 1552, 1233, 1673, 528, 1905, 1119, 126, 502, 1087, 514, 812, 1806, 1007, 997, 1281, 1922, 1889, 86, 821, 698, 2038, 1463, 181, 1581, 414, 741, 170, 1929, 1804, 1313, 890, 1176, 232, 1877, 1216, 1482, 539, 210, 1584, 1874, 771, 358, 1399, 1855, 1715, 554, 157, 963, 786, 1457, 292, 1474, 217, 1424 };
    int cartPosMFInds[] = { 1771, 1479, 772, 1502, 1846, 1031, 183, 1641, 1785, 1175, 423, 135, 842, 1876, 1727, 1146, 511, 97, 1471, 79, 1976, 1863, 240, 1832, 627, 1724, 807, 1519, 1230, 494, 1014, 800, 1108, 1535, 1878, 365, 436, 844, 1327, 269, 601, 921, 134, 291, 951, 1895, 1450, 1364, 1319, 1817, 256, 263, 1556, 1243, 190, 721, 1352, 1498, 1720, 957, 1151, 1707, 969, 574, 1346, 561, 1978, 909, 301, 630, 1445, 1266, 1933, 457, 2017, 1099, 1645, 1387, 2027, 927, 1775, 1192, 859, 681, 50, 757, 1302, 1361, 903, 608, 91, 1548, 29, 1680, 1221, 930, 1088, 1142, 1316, 1283, 1320, 1401, 1395, 340, 500, 1730, 1188, 242, 376, 613, 1508, 1615, 1217, 617, 1643, 1769, 1369, 2025, 979, 467, 8, 1608 };
    int numHighFreqMFs = sizeof(highFreqMFInds) / sizeof(highFreqMFInds[0]);
    int numPoleVelMFs = sizeof(poleVelMFInds) / sizeof(poleVelMFInds[0]);
    int numPoleAngMFs = sizeof(poleAngMFInds) / sizeof(poleAngMFInds[0]);
    int numCartVelMFs = sizeof(cartVelMFInds) / sizeof(cartVelMFInds[0]);    
    int numCartPosMFs = sizeof(cartPosMFInds) / sizeof(cartPosMFInds[0]);    
    plotMFChange("HighFreqMFs", highFreqMFInds, numHighFreqMFs, mfWeightDiffSums,
                 mfWeightDiffPercents, mfWeightSums, numMZ);
    plotMFChange("PoleVelMFs", poleVelMFInds, numPoleVelMFs, mfWeightDiffSums,
                 mfWeightDiffPercents, mfWeightSums, numMZ);
    plotMFChange("PoleAngMFs", poleAngMFInds, numPoleAngMFs, mfWeightDiffSums,
                 mfWeightDiffPercents, mfWeightSums, numMZ);
    plotMFChange("CartVelMFs", cartVelMFInds, numCartVelMFs, mfWeightDiffSums,
                 mfWeightDiffPercents, mfWeightSums, numMZ);
    plotMFChange("CartPosMFs", cartPosMFInds, numCartPosMFs, mfWeightDiffSums,
                 mfWeightDiffPercents, mfWeightSums, numMZ);
}

void WeightAnalyzer::plotMFChange(string vName, int *mfInds, int numMFInds, vector<vector<float> >&mfWeightDiffSums,
                                  vector<vector<float> >&mfWeightDiffPercents, vector<vector<float> >&mfWeightSums,
                                  int numMZ) {
    for (int mz=0; mz<numMZ; mz++) {
        vector<float> weights;
        vector<float> wDiff;
        vector<float> wDiffPerc;
        for (int i=0; i<numMFInds; i++) {
            weights.push_back(mfWeightSums[mz][mfInds[i]]);
            wDiff.push_back(mfWeightDiffSums[mz][mfInds[i]]);
            wDiffPerc.push_back(mfWeightDiffPercents[mz][mfInds[i]]);
        }

        {
            // TODO: Consider putting all these weight diffs into a single plot colored by state variable
            // Plot this re-ordered weight diff
            stringstream ss;
            ss << mz;
            string filename = plot_dir + vName + "_MZ" + ss.str() + "_ordered_weight_diff_percent.pdf";
            const vector<float> w(wDiffPerc);
            R["weightsvec"] = w;
            string txt =
                "library(ggplot2); "
                "data=data.frame(w=weightsvec); "
                "plot=ggplot(data=data, aes(x=1:nrow(data), y=w)) + geom_bar(stat=\"identity\") + xlab(\"MF Number\") + ylab(\"Connected Granule Percent Weight Change\") + labs(title = expression(\"MZ"+ss.str()+" " + vName + " Ordered MF Percent Weight Changes\"));"
                "ggsave(plot,file=\""+filename+"\"); ";
            R.parseEvalQ(txt);
        }

        {
            // Plot this re-ordered weights colored by diff
            stringstream ss;
            ss << mz;
            string filename = plot_dir + vName + "_MZ" + ss.str() + "ordered_weights.pdf";
            const vector<float> d(wDiff);
            const vector<float> w(weights);
            R["weightsvec"] = w;
            R["diffvec"] = d;
            string txt =
                "library(ggplot2); "
                "data=data.frame(w=weightsvec,WeightDiff=diffvec); "
                "plot=ggplot(data=data, aes(x=1:nrow(data), y=w, fill=WeightDiff)) + geom_bar(stat=\"identity\") + scale_fill_gradient2(low=\"darkred\",high=\"darkblue\", mid=\"white\", midpoint=0) + xlab(\"MF Number\") + ylab(\"Sum of Connected Granule Weights\") + labs(title = expression(\"MZ"+ss.str()+" " + vName + " Ordered MF Weights\"));"
                "ggsave(plot,file=\""+filename+"\"); ";
            R.parseEvalQ(txt);
        }
    }
}

void WeightAnalyzer::analyzeFile(string fname) {
    cout << "Analyzing file " << fname << endl;
    plot_dir = "plots_" + fname + "/";
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
        string filename = plot_dir + "MZ" + ss.str() + "_GRPC_weight_hist.pdf";
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
#endif /* BUILD_ANALYSIS */
