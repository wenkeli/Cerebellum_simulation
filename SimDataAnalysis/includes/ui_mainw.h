/********************************************************************************
** Form generated from reading UI file 'mainw.ui'
**
** Created: Mon Aug 15 18:13:12 2011
**      by: Qt User Interface Compiler version 4.7.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINW_H
#define UI_MAINW_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QComboBox>
#include <QtGui/QGridLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QMainWindow>
#include <QtGui/QPushButton>
#include <QtGui/QSpinBox>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWClass
{
public:
    QWidget *centralwidget;
    QWidget *verticalLayoutWidget;
    QVBoxLayout *verticalLayout;
    QPushButton *loadPSHButton;
    QPushButton *loadSimButton;
    QPushButton *singleCellNPButton;
    QPushButton *multicellNPButton;
    QPushButton *calcPFPCPlastButton;
    QPushButton *exportPFPCPlastActButton;
    QPushButton *calcSpikeRatesButton;
    QPushButton *exportSpikeRatesButton;
    QPushButton *quitButton;
    QWidget *gridLayoutWidget;
    QGridLayout *gridLayout;
    QLabel *label_2;
    QLabel *label;
    QLabel *label_4;
    QSpinBox *multiCellPageBox;
    QSpinBox *multiCellStrideBox;
    QSpinBox *singleCellNumBox;
    QLabel *label_3;
    QComboBox *dispCellTypeBox;
    QLabel *label_5;
    QSpinBox *pfPCPlastUSTimeSpinBox;

    void setupUi(QMainWindow *MainWClass)
    {
        if (MainWClass->objectName().isEmpty())
            MainWClass->setObjectName(QString::fromUtf8("MainWClass"));
        MainWClass->resize(459, 354);
        centralwidget = new QWidget(MainWClass);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        verticalLayoutWidget = new QWidget(centralwidget);
        verticalLayoutWidget->setObjectName(QString::fromUtf8("verticalLayoutWidget"));
        verticalLayoutWidget->setGeometry(QRect(270, 10, 175, 311));
        verticalLayout = new QVBoxLayout(verticalLayoutWidget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        loadPSHButton = new QPushButton(verticalLayoutWidget);
        loadPSHButton->setObjectName(QString::fromUtf8("loadPSHButton"));

        verticalLayout->addWidget(loadPSHButton);

        loadSimButton = new QPushButton(verticalLayoutWidget);
        loadSimButton->setObjectName(QString::fromUtf8("loadSimButton"));

        verticalLayout->addWidget(loadSimButton);

        singleCellNPButton = new QPushButton(verticalLayoutWidget);
        singleCellNPButton->setObjectName(QString::fromUtf8("singleCellNPButton"));

        verticalLayout->addWidget(singleCellNPButton);

        multicellNPButton = new QPushButton(verticalLayoutWidget);
        multicellNPButton->setObjectName(QString::fromUtf8("multicellNPButton"));

        verticalLayout->addWidget(multicellNPButton);

        calcPFPCPlastButton = new QPushButton(verticalLayoutWidget);
        calcPFPCPlastButton->setObjectName(QString::fromUtf8("calcPFPCPlastButton"));

        verticalLayout->addWidget(calcPFPCPlastButton);

        exportPFPCPlastActButton = new QPushButton(verticalLayoutWidget);
        exportPFPCPlastActButton->setObjectName(QString::fromUtf8("exportPFPCPlastActButton"));

        verticalLayout->addWidget(exportPFPCPlastActButton);

        calcSpikeRatesButton = new QPushButton(verticalLayoutWidget);
        calcSpikeRatesButton->setObjectName(QString::fromUtf8("calcSpikeRatesButton"));

        verticalLayout->addWidget(calcSpikeRatesButton);

        exportSpikeRatesButton = new QPushButton(verticalLayoutWidget);
        exportSpikeRatesButton->setObjectName(QString::fromUtf8("exportSpikeRatesButton"));

        verticalLayout->addWidget(exportSpikeRatesButton);

        quitButton = new QPushButton(verticalLayoutWidget);
        quitButton->setObjectName(QString::fromUtf8("quitButton"));

        verticalLayout->addWidget(quitButton);

        gridLayoutWidget = new QWidget(centralwidget);
        gridLayoutWidget->setObjectName(QString::fromUtf8("gridLayoutWidget"));
        gridLayoutWidget->setGeometry(QRect(13, 10, 251, 171));
        gridLayout = new QGridLayout(gridLayoutWidget);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        gridLayout->setContentsMargins(0, 0, 0, 0);
        label_2 = new QLabel(gridLayoutWidget);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        gridLayout->addWidget(label_2, 1, 0, 1, 1);

        label = new QLabel(gridLayoutWidget);
        label->setObjectName(QString::fromUtf8("label"));

        gridLayout->addWidget(label, 2, 0, 1, 1);

        label_4 = new QLabel(gridLayoutWidget);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        gridLayout->addWidget(label_4, 3, 0, 1, 1);

        multiCellPageBox = new QSpinBox(gridLayoutWidget);
        multiCellPageBox->setObjectName(QString::fromUtf8("multiCellPageBox"));

        gridLayout->addWidget(multiCellPageBox, 1, 1, 1, 1);

        multiCellStrideBox = new QSpinBox(gridLayoutWidget);
        multiCellStrideBox->setObjectName(QString::fromUtf8("multiCellStrideBox"));

        gridLayout->addWidget(multiCellStrideBox, 2, 1, 1, 1);

        singleCellNumBox = new QSpinBox(gridLayoutWidget);
        singleCellNumBox->setObjectName(QString::fromUtf8("singleCellNumBox"));

        gridLayout->addWidget(singleCellNumBox, 3, 1, 1, 1);

        label_3 = new QLabel(gridLayoutWidget);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        gridLayout->addWidget(label_3, 0, 0, 1, 1);

        dispCellTypeBox = new QComboBox(gridLayoutWidget);
        dispCellTypeBox->setObjectName(QString::fromUtf8("dispCellTypeBox"));

        gridLayout->addWidget(dispCellTypeBox, 0, 1, 1, 1);

        label_5 = new QLabel(gridLayoutWidget);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        gridLayout->addWidget(label_5, 4, 0, 1, 1);

        pfPCPlastUSTimeSpinBox = new QSpinBox(gridLayoutWidget);
        pfPCPlastUSTimeSpinBox->setObjectName(QString::fromUtf8("pfPCPlastUSTimeSpinBox"));

        gridLayout->addWidget(pfPCPlastUSTimeSpinBox, 4, 1, 1, 1);

        MainWClass->setCentralWidget(centralwidget);

        retranslateUi(MainWClass);
        QObject::connect(loadPSHButton, SIGNAL(clicked()), MainWClass, SLOT(loadPSHFile()));
        QObject::connect(singleCellNPButton, SIGNAL(clicked()), MainWClass, SLOT(dispSingleCellNP()));
        QObject::connect(multicellNPButton, SIGNAL(clicked()), MainWClass, SLOT(dispMultiCellNP()));
        QObject::connect(multiCellPageBox, SIGNAL(valueChanged(int)), MainWClass, SLOT(updateMultiCellDisp(int)));
        QObject::connect(multiCellStrideBox, SIGNAL(valueChanged(int)), MainWClass, SLOT(updateMultiCellBound(int)));
        QObject::connect(singleCellNumBox, SIGNAL(valueChanged(int)), MainWClass, SLOT(updateSingleCellDisp(int)));
        QObject::connect(dispCellTypeBox, SIGNAL(currentIndexChanged(int)), MainWClass, SLOT(updateCellType(int)));
        QObject::connect(calcPFPCPlastButton, SIGNAL(clicked()), MainWClass, SLOT(calcPFPCPlasticity()));
        QObject::connect(exportPFPCPlastActButton, SIGNAL(clicked()), MainWClass, SLOT(exportPFPCPlastAct()));
        QObject::connect(calcSpikeRatesButton, SIGNAL(clicked()), MainWClass, SLOT(calcSpikeRates()));
        QObject::connect(exportSpikeRatesButton, SIGNAL(clicked()), MainWClass, SLOT(exportSpikeRates()));
        QObject::connect(loadSimButton, SIGNAL(clicked()), MainWClass, SLOT(loadSimFile()));

        QMetaObject::connectSlotsByName(MainWClass);
    } // setupUi

    void retranslateUi(QMainWindow *MainWClass)
    {
        MainWClass->setWindowTitle(QApplication::translate("MainWClass", "MainWindow", 0, QApplication::UnicodeUTF8));
        loadPSHButton->setText(QApplication::translate("MainWClass", "Load PSH data", 0, QApplication::UnicodeUTF8));
        loadSimButton->setText(QApplication::translate("MainWClass", "Load Sim State", 0, QApplication::UnicodeUTF8));
        singleCellNPButton->setText(QApplication::translate("MainWClass", "new single cell PSH panel", 0, QApplication::UnicodeUTF8));
        multicellNPButton->setText(QApplication::translate("MainWClass", "new multi cell PSH panel", 0, QApplication::UnicodeUTF8));
        calcPFPCPlastButton->setText(QApplication::translate("MainWClass", "calc PF PC plasticity", 0, QApplication::UnicodeUTF8));
        exportPFPCPlastActButton->setText(QApplication::translate("MainWClass", "export PFPC plast Activity", 0, QApplication::UnicodeUTF8));
        calcSpikeRatesButton->setText(QApplication::translate("MainWClass", "calc spike rates", 0, QApplication::UnicodeUTF8));
        exportSpikeRatesButton->setText(QApplication::translate("MainWClass", "export spike rates", 0, QApplication::UnicodeUTF8));
        quitButton->setText(QApplication::translate("MainWClass", "quit", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("MainWClass", "multi disp page #", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("MainWClass", "multi disp page stride", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("MainWClass", "single disp cell #", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("MainWClass", "Display cell type", 0, QApplication::UnicodeUTF8));
        label_5->setText(QApplication::translate("MainWClass", "PFPC plasticity US time", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class MainWClass: public Ui_MainWClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINW_H
