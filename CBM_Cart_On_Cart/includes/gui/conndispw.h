#ifndef CONNDISPW_H
#define CONNDISPW_H

#include <sstream>
#include <QtGui/QWidget>
#include <QtGui/QPainter>
#include <QtGui/QPaintEvent>
#include <QtGui/QColor>
#include <QtGui/QPixmap>
#include "ui_conndispw.h" //qt uic generated header file, do not modify
#include "../common.h"

//widget class that displays cell connections of type defined in ConnDispT enum in parameters.h
class ConnDispW : public QWidget
{
	//autogenerated macro from moc_conndispw.h, necessary for qt signal slot mechanism
	//ignore
    Q_OBJECT

public:
	//s: starting cell/fiber to display, e: ending cell/fiber to display, t: type of display, see parameters.h for details
    ConnDispW(QWidget *parent, int s, int e, ConnDispT t);
    ~ConnDispW();

private:
    Ui::ConnDispWClass ui; //auto generated code from ui_conndispw.h
    int start, end; //the start and end of the cell/fiber to display
    ConnDispT dispT; //display type, see ConnDispT in parameters.h
    QPixmap *backBuf; //buffer to draw, so that things are only drawn once, enable more efficient updates
    bool valid; //represents if the input parameters are correct
    void paintBuf(); //draw the buffer

protected:
	//paint event that copies the buffer to the screen
	void paintEvent(QPaintEvent *event);
};

#endif // CONNDISPW_H
