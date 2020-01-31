/********************************************************************************
** Form generated from reading UI file 'spikeratesdispw.ui'
**
** Created: Mon Aug 8 12:06:53 2011
**      by: Qt User Interface Compiler version 4.7.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_SPIKERATESDISPW_H
#define UI_SPIKERATESDISPW_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QHeaderView>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_SpikeRatesDispWClass
{
public:

    void setupUi(QWidget *SpikeRatesDispWClass)
    {
        if (SpikeRatesDispWClass->objectName().isEmpty())
            SpikeRatesDispWClass->setObjectName(QString::fromUtf8("SpikeRatesDispWClass"));
        SpikeRatesDispWClass->resize(400, 300);
        SpikeRatesDispWClass->setStyleSheet(QString::fromUtf8("background-color: rgb(0, 0, 0);"));

        retranslateUi(SpikeRatesDispWClass);

        QMetaObject::connectSlotsByName(SpikeRatesDispWClass);
    } // setupUi

    void retranslateUi(QWidget *SpikeRatesDispWClass)
    {
        SpikeRatesDispWClass->setWindowTitle(QApplication::translate("SpikeRatesDispWClass", "SpikeRatesDispW", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class SpikeRatesDispWClass: public Ui_SpikeRatesDispWClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_SPIKERATESDISPW_H
