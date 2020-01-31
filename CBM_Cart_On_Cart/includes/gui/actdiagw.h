#ifndef ACTDIAGW_H
#define ACTDIAGW_H

#include <QtGui/QWidget>
#include <QtGui/QPainter>
#include <QtGui/QPaintEvent>
#include <QtGui/QColor>
#include <QtGui/QPixmap>
#include <QtCore/QMutex>
#include "ui_actdiagw.h"
#include "../common.h"
#include "../globalvars.h"

class ActDiagW : public QWidget
{
    Q_OBJECT

public:
    ActDiagW(QWidget *parent = 0);
    ~ActDiagW();

public slots:
    void drawActivity(vector<bool> grAPs, vector<bool> goAPs);

private:
    Ui::ActDiagWClass ui;
    QPixmap *backBuf;
    float scaleX, scaleY;

protected:
	void paintEvent(QPaintEvent *);

};

#endif // ACTDIAGW_H
