#ifndef MFGRMAINP_H
#define MFGRMAINP_H

#include <QtGui/QWidget>
#include "ui_mfgrmainp.h"
#include "common.h"

class MFGRmainP : public QWidget
{
    Q_OBJECT

public:
    MFGRmainP(QWidget *parent = 0);
    ~MFGRmainP();

private:
    Ui::MFGRmainPClass ui;

public slots:
	void drawMFGRConns();
};

#endif // MFGRMAINP_H
