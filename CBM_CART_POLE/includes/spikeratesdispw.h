#ifndef SPIKERATESDISPW_H
#define SPIKERATESDISPW_H

#include <sstream>
#include <string>
#include <QtGui/QWidget>
#include <QtCore/QMutex>
#include <QtCore/QMutexLocker>
#include <QtGui/QPainter>
#include <QtGui/QPaintEvent>
#include <QtGui/QColor>
#include <QtGui/QPixmap>
#include "common.h"
#include "globalvars.h"

#include "ui_spikeratesdispw.h"

class SpikeRatesDispW : public QWidget
{
    Q_OBJECT

public:
    SpikeRatesDispW( QWidget *, int, unsigned int *, QMutex *, int, string);
    ~SpikeRatesDispW();

private:
    Ui::SpikeRatesDispWClass ui;
    string windowTitle;
    QPixmap *backBuf;
    void paintBuf(float *, int *, int);

protected:
    void paintEvent(QPaintEvent *event);
};

#endif // SPIKERATESDISPW_H
