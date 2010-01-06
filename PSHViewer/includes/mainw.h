#ifndef MAINW_H
#define MAINW_H

#include <QtGui/QMainWindow>
#include <QtGui/QApplication>

#include "common.h"
#include "pshdispw.h"

#include "ui_mainw.h"

class MainW : public QMainWindow
{
    Q_OBJECT

public:
    MainW(QWidget *parent = 0, QApplication *application);
    ~MainW();

private:
    Ui::MainWClass ui;
    QApplication *app;

public slots:
	void dispSingleCell();
	void dispAllCells();
};

#endif // MAINW_H
