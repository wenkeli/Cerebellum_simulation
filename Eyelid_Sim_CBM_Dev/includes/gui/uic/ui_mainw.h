/********************************************************************************
** Form generated from reading UI file 'mainw.ui'
**
** Created: Tue Aug 14 15:30:25 2012
**      by: Qt User Interface Compiler version 4.8.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINW_H
#define UI_MAINW_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QHeaderView>
#include <QtGui/QPushButton>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWClass
{
public:
    QPushButton *runButton;
    QPushButton *quitButton;

    void setupUi(QWidget *MainWClass)
    {
        if (MainWClass->objectName().isEmpty())
            MainWClass->setObjectName(QString::fromUtf8("MainWClass"));
        MainWClass->resize(162, 165);
        runButton = new QPushButton(MainWClass);
        runButton->setObjectName(QString::fromUtf8("runButton"));
        runButton->setGeometry(QRect(50, 10, 61, 25));
        quitButton = new QPushButton(MainWClass);
        quitButton->setObjectName(QString::fromUtf8("quitButton"));
        quitButton->setGeometry(QRect(50, 110, 71, 25));

        retranslateUi(MainWClass);
        QObject::connect(runButton, SIGNAL(clicked()), MainWClass, SLOT(run()));

        QMetaObject::connectSlotsByName(MainWClass);
    } // setupUi

    void retranslateUi(QWidget *MainWClass)
    {
        MainWClass->setWindowTitle(QApplication::translate("MainWClass", "MainW", 0, QApplication::UnicodeUTF8));
        runButton->setText(QApplication::translate("MainWClass", "run", 0, QApplication::UnicodeUTF8));
        quitButton->setText(QApplication::translate("MainWClass", "Quit", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class MainWClass: public Ui_MainWClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINW_H
