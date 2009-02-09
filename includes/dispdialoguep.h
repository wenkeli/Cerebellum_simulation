#ifndef DISPDIALOGUEP_H
#define DISPDIALOGUEP_H

#include <QtGui/QWidget>
#include "ui_dispdialoguep.h"
#include "parameters.h"
#include "common.h"

class DispDialogueP : public QWidget
{
    Q_OBJECT

public:
    DispDialogueP(QWidget *parent = 0);
    ~DispDialogueP();
    void setDispT(ConnDispT);

private:
    Ui::DispDialoguePClass ui;
    ConnDispT dispT;
public slots:
	void dispConns();
};

#endif // DISPDIALOGUEP_H
