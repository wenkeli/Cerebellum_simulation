/********************************************************************************
** Form generated from reading UI file 'pshdispw.ui'
**
** Created: Thu May 10 13:25:43 2012
**      by: Qt User Interface Compiler version 4.7.4
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_PSHDISPW_H
#define UI_PSHDISPW_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QHeaderView>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_PSHDispwClass
{
public:

    void setupUi(QWidget *PSHDispwClass)
    {
        if (PSHDispwClass->objectName().isEmpty())
            PSHDispwClass->setObjectName(QString::fromUtf8("PSHDispwClass"));
        PSHDispwClass->resize(443, 206);
        PSHDispwClass->setStyleSheet(QString::fromUtf8("background-color: rgb(0, 0, 0);"));

        retranslateUi(PSHDispwClass);

        QMetaObject::connectSlotsByName(PSHDispwClass);
    } // setupUi

    void retranslateUi(QWidget *PSHDispwClass)
    {
        PSHDispwClass->setWindowTitle(QApplication::translate("PSHDispwClass", "PSHDispw", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class PSHDispwClass: public Ui_PSHDispwClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_PSHDISPW_H
