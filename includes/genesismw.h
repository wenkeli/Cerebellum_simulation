#ifndef GENESISMW_H
#define GENESISMW_H

#include <QtGui/QMainWindow>
#include "ui_genesismw.h"
#include "synapsegenesis.h"
#include "common.h"


class GenesisMW : public QMainWindow
{
    Q_OBJECT

public:
    GenesisMW(QWidget *parent = 0);
    ~GenesisMW();
    QTextBrowser *getStatusBox();

private:
    Ui::GenesisMWClass ui;

public slots:
	void makeConns();
	void showMFGRMainP();
};

#endif // GENESISMW_H
