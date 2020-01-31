/********************************************************************************
** Form generated from reading UI file 'conndispw.ui'
**
** Created: Mon Aug 8 12:06:53 2011
**      by: Qt User Interface Compiler version 4.7.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_CONNDISPW_H
#define UI_CONNDISPW_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QHeaderView>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ConnDispWClass
{
public:

    void setupUi(QWidget *ConnDispWClass)
    {
        if (ConnDispWClass->objectName().isEmpty())
            ConnDispWClass->setObjectName(QString::fromUtf8("ConnDispWClass"));
        ConnDispWClass->resize(691, 193);
        ConnDispWClass->setStyleSheet(QString::fromUtf8("background-color: rgb(0, 0, 0);"));

        retranslateUi(ConnDispWClass);

        QMetaObject::connectSlotsByName(ConnDispWClass);
    } // setupUi

    void retranslateUi(QWidget *ConnDispWClass)
    {
        ConnDispWClass->setWindowTitle(QApplication::translate("ConnDispWClass", "ConnDispW", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class ConnDispWClass: public Ui_ConnDispWClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CONNDISPW_H
