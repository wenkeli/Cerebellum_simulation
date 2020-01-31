/*
 * main.cpp
 *
 *  Created on: Feb 2, 2009
 *      Author: wen
 */

#include "../includes/main.h"
#include "../includes/mainw.h"
#include <QtGui/QApplication>
#include <QtCore/QMutex>

int main(int argc, char *argv[])
{
	QApplication app(argc, argv);
	MainW *mainW=new MainW(NULL, &app);

	simOut.open(argv[1], ios::binary);
	pshOut.open(argv[2], ios::binary);
	app.setQuitOnLastWindowClosed(true);
	app.setActiveWindow(mainW);
	mainW->show();

	app.exec();

	writeSimOut();
	writePSHOut();
	simOut.close();
	pshOut.close();
}
