#include "../includes/main.h"
#include "cartpole.cpp"
#include "mfinput.cpp"

using namespace std;

float calcMZOutputForce(const bool* ncFirings, int numNC) {
    float inputSum = 0;
    for (int i=0; i<numNC; ++i)
        inputSum += ncFirings[i];
    inputSum = inputSum/(float)numNC;
    return inputSum;
}

int main(int argc, char **argv) {
    int numMZ = 2;

    // Create simulation core with 2 microzones
    cout<<"Creating Sim Core..."<< endl;
    CBMSimCore simCore(numMZ);

    cout<<"Creating visualization..."<< endl;
    CerebellumViz viz(&simCore);

    cout<<"Creating Cartpole Domain..."<<endl;
    CartPole cp;

    cout<<"Creating MFInputs..."<< endl;
    MFInput mf(simCore.getNumMF());
    mf.addStateVariable("poleAngle",cp.getMinPoleAngle(),cp.getMaxPoleAngle());
    mf.addStateVariable("poleVelocity",cp.getMinPoleVelocity(),cp.getMaxPoleVelocity());

    cerr<<"Starting run..."<<endl;
    int t;
    for(int i=0; i<2; i++) {
        t=time(0);
        cerr<<"iteration #"<<i<<": ";
        cerr.flush();
        for(int j=0; j<5000; j++) {
            // Update MFs from CP state vars
            mf.updateStateVariable("poleAngle",cp.getPoleAngle());
            mf.updateStateVariable("poleVelocity",cp.getPoleVelocity());

            // Calc MF activity
            const bool *mfAct = mf.calcActivity();

            // Update the mf activity to both microzones
            simCore.updateMFInput(mfAct);

            // Set the error for each microzone -- (mz#,error)
            cp.getErrorLeft()  ? simCore.updateErrDrive(0,1) : simCore.updateErrDrive(0,0);
            cp.getErrorRight() ? simCore.updateErrDrive(1,1) : simCore.updateErrDrive(1,0);

            // Run the simulation
            simCore.calcActivity();

            // Visualize the resulting firings
            viz.update();

            // Get the MZ output
            float mz0Force = calcMZOutputForce(simCore.exportAPNC(0),simCore.getNumNC());
            float mz1Force = calcMZOutputForce(simCore.exportAPNC(1),simCore.getNumNC());
            float netForce = mz0Force - mz1Force;
            cp.run(netForce);
        }
        cerr<<time(0)-t<<" sec"<<endl;
    }
}
