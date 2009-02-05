#ifndef MFGRCONNS_H
#define MFGRCONNS_H

#include <QtGui/QWidget>
#include <QtGui/QPainter>
#include <QtGui/QBrush>
#include "common.h"
#include "ui_mfgrconnsw.h"


class MFGRConnsW : public QWidget
{
    Q_OBJECT

public:
    MFGRConnsW(QWidget *parent = 0);
    ~MFGRConnsW();

private:
    Ui::MFGRConnsWClass ui;

protected:
	void paintEvent(QPaintEvent *event);
};

#endif // MFGRCONNS_H
