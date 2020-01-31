/********************************************************************************
** Form generated from reading UI file 'simdispw.ui'
**
** Created: Mon Aug 8 12:06:53 2011
**      by: Qt User Interface Compiler version 4.7.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_SIMDISPW_H
#define UI_SIMDISPW_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QHeaderView>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_SimDispWClass
{
public:

    void setupUi(QWidget *SimDispWClass)
    {
        if (SimDispWClass->objectName().isEmpty())
            SimDispWClass->setObjectName(QString::fromUtf8("SimDispWClass"));
        SimDispWClass->resize(786, 696);
        SimDispWClass->setStyleSheet(QString::fromUtf8("background-color: rgb(0, 0, 0);"));

        retranslateUi(SimDispWClass);

        QMetaObject::connectSlotsByName(SimDispWClass);
    } // setupUi

    void retranslateUi(QWidget *SimDispWClass)
    {
        SimDispWClass->setWindowTitle(QApplication::translate("SimDispWClass", "SimDispW", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class SimDispWClass: public Ui_SimDispWClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_SIMDISPW_H
