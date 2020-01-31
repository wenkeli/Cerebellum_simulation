/********************************************************************************
** Form generated from reading UI file 'dispdialoguep.ui'
**
** Created: Fri Apr 8 14:04:28 2011
**      by: Qt User Interface Compiler version 4.7.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_DISPDIALOGUEP_H
#define UI_DISPDIALOGUEP_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QPushButton>
#include <QtGui/QSpinBox>
#include <QtGui/QTextBrowser>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_DispDialoguePClass
{
public:
    QTextBrowser *statusBox;
    QSpinBox *startNumSel;
    QSpinBox *endNumSel;
    QLabel *label;
    QLabel *label_2;
    QPushButton *displayButton;
    QWidget *verticalLayoutWidget;
    QVBoxLayout *verticalLayout;
    QLabel *maxL;
    QLabel *maxValLab;

    void setupUi(QWidget *DispDialoguePClass)
    {
        if (DispDialoguePClass->objectName().isEmpty())
            DispDialoguePClass->setObjectName(QString::fromUtf8("DispDialoguePClass"));
        DispDialoguePClass->resize(490, 80);
        statusBox = new QTextBrowser(DispDialoguePClass);
        statusBox->setObjectName(QString::fromUtf8("statusBox"));
        statusBox->setGeometry(QRect(0, 0, 241, 81));
        startNumSel = new QSpinBox(DispDialoguePClass);
        startNumSel->setObjectName(QString::fromUtf8("startNumSel"));
        startNumSel->setGeometry(QRect(300, 0, 101, 22));
        endNumSel = new QSpinBox(DispDialoguePClass);
        endNumSel->setObjectName(QString::fromUtf8("endNumSel"));
        endNumSel->setGeometry(QRect(300, 50, 101, 22));
        label = new QLabel(DispDialoguePClass);
        label->setObjectName(QString::fromUtf8("label"));
        label->setGeometry(QRect(250, 0, 46, 21));
        label_2 = new QLabel(DispDialoguePClass);
        label_2->setObjectName(QString::fromUtf8("label_2"));
        label_2->setGeometry(QRect(260, 50, 31, 21));
        displayButton = new QPushButton(DispDialoguePClass);
        displayButton->setObjectName(QString::fromUtf8("displayButton"));
        displayButton->setGeometry(QRect(410, 0, 75, 23));
        verticalLayoutWidget = new QWidget(DispDialoguePClass);
        verticalLayoutWidget->setObjectName(QString::fromUtf8("verticalLayoutWidget"));
        verticalLayoutWidget->setGeometry(QRect(410, 30, 71, 41));
        verticalLayout = new QVBoxLayout(verticalLayoutWidget);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        maxL = new QLabel(verticalLayoutWidget);
        maxL->setObjectName(QString::fromUtf8("maxL"));

        verticalLayout->addWidget(maxL);

        maxValLab = new QLabel(verticalLayoutWidget);
        maxValLab->setObjectName(QString::fromUtf8("maxValLab"));

        verticalLayout->addWidget(maxValLab);


        retranslateUi(DispDialoguePClass);
        QObject::connect(displayButton, SIGNAL(clicked()), DispDialoguePClass, SLOT(dispConns()));
        QObject::connect(startNumSel, SIGNAL(valueChanged(int)), endNumSel, SLOT(setValue(int)));

        QMetaObject::connectSlotsByName(DispDialoguePClass);
    } // setupUi

    void retranslateUi(QWidget *DispDialoguePClass)
    {
        DispDialoguePClass->setWindowTitle(QApplication::translate("DispDialoguePClass", "DispDialogueP", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("DispDialoguePClass", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600;\">Start</span></p></body></html>", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("DispDialoguePClass", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600;\">End</span></p></body></html>", 0, QApplication::UnicodeUTF8));
        displayButton->setText(QApplication::translate("DispDialoguePClass", "Display", 0, QApplication::UnicodeUTF8));
        maxL->setText(QApplication::translate("DispDialoguePClass", "Max:", 0, QApplication::UnicodeUTF8));
        maxValLab->setText(QApplication::translate("DispDialoguePClass", "maxnumber", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class DispDialoguePClass: public Ui_DispDialoguePClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DISPDIALOGUEP_H
