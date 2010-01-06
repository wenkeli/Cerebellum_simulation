#ifndef PSHDISPW_H
#define PSHDISPW_H

#include <QtGui/QWidget>
#include <QtGui/QPaintEvent>
#include <QtGui/QPixmap>
#include <QtGui/QPainter>

#include "common.h"

#include "ui_pshdispw.h"

class PSHDispw : public QWidget
{
    Q_OBJECT

public:
    PSHDispw(QWidget *parent, int t, int cT, int cN);
    ~PSHDispw();

private:
    Ui::PSHDispwClass ui;
    QPixmap *backBuf;
    int type, wH, wW, cellT, cellN;

    void paintPSH();

protected:
	void paintEvent(QPaintEvent *event);
};

#endif // PSHDISPW_H
