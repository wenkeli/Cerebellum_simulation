/*
 * main.cpp
 *
 *  Created on: Jan 5, 2010
 *      Author: wen
 */

#include "../includes/main.h"

int main(int argc, char **argv)
{
	QApplication app(argc, argv);
	MainW *mainw=new MainW(NULL, &app);

	app.setQuitOnLastWindowClosed(true);
	app.setActiveWindow(mainw);
	mainw->show();

	return app.exec();
}
