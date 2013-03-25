/*
 * main.cpp
 *
 *  Created on: Jan 9, 2013
 *      Author: consciousness
 */

#include "../includes/main.h"

using namespace std;

int main(int argc, char **argv)
{
	fstream infile;
	fstream outfile;
	fstream inStateFile;

	fstream cpfile;
	fstream apfile;
//
	ActivityParams *ap;
	ConnectivityParams *cp;
//
//	InNetActivityState *newIAS, *loadIAS;
//	InNetConnectivityState *newICS, *loadICS;
//
//	MZoneActivityState *newMAS, *loadMAS;
//	MZoneConnectivityState *newMCS, *loadMCS;
//
	CBMState *newCBMS, *loadCBMS;
//
	infile.open(argv[1], ios::in);
	cout<<argv[1]<<endl;
	cp=new ConnectivityParams(infile);
	infile.close();

	infile.open(argv[2], ios::in);
	cout<<argv[2]<<endl;
	ap=new ActivityParams(infile);
	infile.close();

	cout<<"here"<<endl;
//
//	newIAS=new InNetActivityState(cp, ap);
//	outfile.open("ias", ios::binary|ios::out);
//	newIAS->writeState(outfile);
//	outfile.close();
//	inStateFile.open("ias", ios::binary|ios::in);
//	loadIAS=new InNetActivityState(cp, inStateFile);
//	inStateFile.close();
//	cout<<"innetactivitystate is equal: "<<newIAS->equivalent(*loadIAS)<<endl;
//
//	newICS=new InNetConnectivityState(cp, ap->msPerTimeStep, time(0));
//	outfile.open("ics", ios::binary|ios::out);
//	newICS->writeState(outfile);
//	outfile.close();
//	inStateFile.open("ics", ios::binary|ios::in);
//	loadICS=new InNetConnectivityState(cp, inStateFile);
//	inStateFile.close();
//	cout<<"innetconnectivityState is equal: "<<newICS->equivalent(*loadICS)<<endl;
//
//	newMAS=new MZoneActivityState(cp, ap, time(0));
//	outfile.open("mas", ios::binary|ios::out);
//	newMAS->writeState(outfile);
//	outfile.close();
//	inStateFile.open("mas", ios::binary|ios::in);
//	loadMAS=new MZoneActivityState(cp, ap, inStateFile);
//	inStateFile.close();
//	cout<<"mzoneactivitystate is equal: "<<newMAS->equivalent(*loadMAS)<<endl;
//
//	newMCS=new MZoneConnectivityState(cp, time(0));
//	outfile.open("mcs", ios::binary|ios::out);
//	newMCS->writeState(outfile);
//	outfile.close();
//	inStateFile.open("mcs", ios::binary|ios::in);
//	loadMCS=new MZoneConnectivityState(cp, inStateFile);
//	inStateFile.close();
//	cout<<"mzoneconnectivitystate is equal: "<<newMCS->equivalent(*loadMCS)<<endl;
//
	cpfile.open(argv[1], ios::in);
	apfile.open(argv[2], ios::in);
//
	cout<<"constructing state"<<endl;
	newCBMS=new CBMState(apfile, cpfile, 1);

	cout<<"writing state"<<endl;
	outfile.open("cbms", ios::binary|ios::out);
	newCBMS->writeState(outfile);
	outfile.close();

	cout<<"loading state"<<endl;
	inStateFile.open("cbms", ios::binary|ios::in);
	loadCBMS=new CBMState(inStateFile);
	inStateFile.close();
	cout<<"CBMState is equal: "<<newCBMS->equivalent(*loadCBMS)<<endl;

	delete newCBMS;
	delete loadCBMS;

//	QApplication app(argc, argv);
//	MainW *mainW;
//
//	mainW=new MainW(&app);
//
//	app.setQuitOnLastWindowClosed(true);
//	app.setActiveWindow(mainW);
//
//	mainW->show();
//
//	return app.exec();
//	return 0;
}


