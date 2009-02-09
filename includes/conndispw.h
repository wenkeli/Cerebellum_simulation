#ifndef CONNDISPW_H
#define CONNDISPW_H

#include <sstream>
#include <QtGui/QWidget>
#include <QtGui/QPainter>
#include <QtGui/QColor>
#include "ui_conndispw.h"
#include "common.h"

class ConnDispW : public QWidget
{
    Q_OBJECT

public:
    ConnDispW(QWidget *parent = 0);
    ~ConnDispW();
    void setDispT(ConnDispT);
    void setBounds(int, int);

private:
    Ui::ConnDispWClass ui;
    int start, end;
    ConnDispT dispT;

protected:
	void paintEvent(QPaintEvent *event);
};

#endif // CONNDISPW_H
