#ifndef TESTPANEL_H
#define TESTPANEL_H

#include <QtGui/QWidget>
#include "uic/ui_testpanel.h"

class TestPanel : public QWidget
{
    Q_OBJECT

public:
    TestPanel(QWidget *parent = 0);
    ~TestPanel();

private:
    Ui::TestPanelClass ui;
};

#endif // TESTPANEL_H
