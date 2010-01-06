#ifndef MAINW_H
#define MAINW_H

#include <QtGui/QMainWindow>
#include <QtGui/QApplication>
#include <QtGui/QFileDialog>
#include <QtCore/QString>

#include "common.h"
#include "pshdispw.h"

#include "ui_mainw.h"

class MainW : public QMainWindow
{
    Q_OBJECT

public:
    MainW(QWidget *parent, QApplication *application);
    ~MainW();

private:
    Ui::MainWClass ui;
    QApplication *app;

public slots:
	void dispSingleCell();
	void dispAllCells();
	void loadPSHFile();
};

#endif // MAINW_H
