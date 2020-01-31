/*
 * main.cpp
 *
 *  Created on: Feb 2, 2009
 *      Author: wen
 */

#include "../includes/main.h"
#include "../includes/gui/mainw.h"
#include <QtGui/QApplication>
#include <QtCore/QMutex>

int main(int argc, char *argv[])
{
    randGen = new CRandomSFMT0(time(NULL));
    QApplication app(argc, argv);
    MainW *mainW=new MainW(NULL, &app);

//	if(argc<3)
//	{
//		cerr<<"not enough arguments, the argument syntax is:"<<endl;
//		cerr<<"cbm_new_CUDA simfile pshfile"<<endl;
//		return 1;
//	}
//
//	simOut.open(argv[1], ios::binary);
//	if(!simOut.good() || !simOut.is_open())
//	{
//		cerr<<"error opening sim state file: "<<argv[1]<<endl;
//		return 1;
//	}
//
//	pshOut.open(argv[2], ios::binary);
//	if(!pshOut.good() || !pshOut.is_open())
//	{
//		cerr<<"error opening psh file: "<<argv[2]<<endl;
//		return 1;
//	}

    app.setQuitOnLastWindowClosed(true);
    app.setActiveWindow(mainW);
    mainW->show();

    app.exec();

//	simOut.close();
//	pshOut.close();

    return 0;
}
