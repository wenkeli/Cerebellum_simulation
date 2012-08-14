/********************************************************************************
** Form generated from reading UI file 'testpanel.ui'
**
** Created: Tue Aug 14 15:30:25 2012
**      by: Qt User Interface Compiler version 4.8.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_TESTPANEL_H
#define UI_TESTPANEL_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QHeaderView>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_TestPanelClass
{
public:

    void setupUi(QWidget *TestPanelClass)
    {
        if (TestPanelClass->objectName().isEmpty())
            TestPanelClass->setObjectName(QString::fromUtf8("TestPanelClass"));
        TestPanelClass->resize(587, 354);

        retranslateUi(TestPanelClass);

        QMetaObject::connectSlotsByName(TestPanelClass);
    } // setupUi

    void retranslateUi(QWidget *TestPanelClass)
    {
        TestPanelClass->setWindowTitle(QApplication::translate("TestPanelClass", "TestPanel", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class TestPanelClass: public Ui_TestPanelClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_TESTPANEL_H
