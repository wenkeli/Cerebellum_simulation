/********************************************************************************
** Form generated from reading UI file 'actdiagw.ui'
**
** Created: Mon Aug 8 12:06:53 2011
**      by: Qt User Interface Compiler version 4.7.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_ACTDIAGW_H
#define UI_ACTDIAGW_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QHeaderView>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ActDiagWClass
{
public:

    void setupUi(QWidget *ActDiagWClass)
    {
        if (ActDiagWClass->objectName().isEmpty())
            ActDiagWClass->setObjectName(QString::fromUtf8("ActDiagWClass"));
        ActDiagWClass->resize(400, 300);
        ActDiagWClass->setStyleSheet(QString::fromUtf8("background-color: rgb(0, 0, 0);"));

        retranslateUi(ActDiagWClass);

        QMetaObject::connectSlotsByName(ActDiagWClass);
    } // setupUi

    void retranslateUi(QWidget *ActDiagWClass)
    {
        ActDiagWClass->setWindowTitle(QApplication::translate("ActDiagWClass", "ActDiagW", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class ActDiagWClass: public Ui_ActDiagWClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_ACTDIAGW_H
