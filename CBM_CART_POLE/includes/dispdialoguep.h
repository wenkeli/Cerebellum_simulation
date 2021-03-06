#ifndef DISPDIALOGUEP_H
#define DISPDIALOGUEP_H

#include <QtGui/QWidget>
#include "ui_dispdialoguep.h"
#include "common.h"
#include "conndispw.h"
#include "globalvars.h"
//widget class that sets the properties for ConnDispW widget to display, including types,
//cells/fibers to display, etc. See dispdialoguep.ui for graphical information on
//the widget layout and signal slot setup.
class DispDialogueP : public QWidget
{
	//autogenerated macro from moc_dispdialoguep.h, necessary for qt signal slot mechanism
	//ignore
    Q_OBJECT

public:
	//ConnDispT t: set the display type defined by ConnDispT enum in parameters.h
    DispDialogueP(QWidget *parent, ConnDispT t);
    ~DispDialogueP();

private:
    Ui::DispDialoguePClass ui;
    ConnDispT dispT;
public slots:
	void dispConns(); //custom qt slot that handles spawning ConnDispW.
};

#endif // DISPDIALOGUEP_H
