#ifndef MAINW_H
#define MAINW_H

#include <QtGui/QWidget>
#include "ui_mainw.h"

class MainW : public QWidget
{
    Q_OBJECT

public:
    MainW(QWidget *parent = 0);
    ~MainW();

private:
    Ui::MainWClass ui;
};

#endif // MAINW_H
