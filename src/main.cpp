/*
 * main.cpp
 *
 *  Created on: Feb 2, 2009
 *      Author: wen
 */

#include "../includes/main.h"

int main(int argc, char *argv[])
{
	QApplication app(argc, argv);
	GenesisMW *mainW=new GenesisMW();

	app.setQuitOnLastWindowClosed(true);
	app.setActiveWindow(mainW);
	mainW->show();

	return app.exec();
}
