#ifndef GENESISMW_H
#define GENESISMW_H

#include <QtGui/QMainWindow>
#include "ui_genesismw.h"

class GenesisMW : public QMainWindow
{
    Q_OBJECT

public:
    GenesisMW(QWidget *parent = 0);
    ~GenesisMW();

private:
    Ui::GenesisMWClass ui;
};

#endif // GENESISMW_H
