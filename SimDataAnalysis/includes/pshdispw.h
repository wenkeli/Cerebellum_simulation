#ifndef PSHDISPW_H
#define PSHDISPW_H

#include <QtGui/QWidget>
#include <QtGui/QPaintEvent>
#include <QtGui/QPixmap>
#include <QtGui/QPainter>
#include <QtGui/QColor>
#include <QtCore/QString>

#include "common.h"

#include "ui_pshdispw.h"

class PSHDispw : public QWidget
{
    Q_OBJECT

public:
    PSHDispw(QWidget *parent, QPixmap *buf, QString wt);
    ~PSHDispw();

    void switchBuf(QPixmap *newBuf);

private:
    Ui::PSHDispwClass ui;
    QPixmap *backBuf;


protected:
	void paintEvent(QPaintEvent *event);
};

#endif // PSHDISPW_H
