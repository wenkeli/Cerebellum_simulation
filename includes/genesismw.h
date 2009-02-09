#ifndef GENESISMW_H
#define GENESISMW_H

#include <QtGui/QMainWindow>
#include <Qt/qapplication.h>
#include "ui_genesismw.h"
#include "synapsegenesis.h"
#include "common.h"
#include "globalvars.h"


class GenesisMW : public QMainWindow
{
    Q_OBJECT

public:
    GenesisMW(QWidget *parent = 0);
    ~GenesisMW();
    QTextBrowser *getStatusBox();
    void setApp(QApplication *);

private:
    Ui::GenesisMWClass ui;
    QApplication *app;

public slots:
	void makeConns();
	void showMFGRMainP();
	void showMFGOMainP();
	void showGRGOMainP();
	void showGOGRMainP();
};

#endif // GENESISMW_H
