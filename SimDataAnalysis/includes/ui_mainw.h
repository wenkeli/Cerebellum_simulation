/********************************************************************************
** Form generated from reading UI file 'mainw.ui'
**
** Created: Wed Sep 7 18:42:48 2011
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
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QMainWindow>
#include <QtGui/QPushButton>
#include <QtGui/QSpinBox>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWClass
{
public:
    QWidget *centralwidget;
    QWidget *horizontalLayoutWidget;
    QHBoxLayout *horizontalLayout;
    QPushButton *loadSimButton;
    QPushButton *loadPSHButton;
    QPushButton *quitButton;
    QWidget *gridLayoutWidget_2;
    QGridLayout *gridLayout_2;
    QLabel *label;
    QSpinBox *multiCellStrideBox;
    QLabel *label_2;
    QSpinBox *multiCellPageBox;
    QLabel *label_4;
    QSpinBox *singleCellNumBox;
    QComboBox *dispCellTypeBox;
    QLabel *label_3;
    QPushButton *singleCellNPButton;
    QPushButton *multicellNPButton;
    QPushButton *calcSpikeRatesButton;
    QPushButton *exportSpikeRatesButton;
    QWidget *gridLayoutWidget_3;
    QGridLayout *gridLayout_3;
    QLabel *label_5;
    QSpinBox *pfPCPlastUSTimeSpinBox;
    QPushButton *calcPFPCPlastButton;
    QPushButton *exportPFPCPlastActButton;
    QWidget *gridLayoutWidget;
    QGridLayout *gridLayout;
    QLabel *label_6;
    QSpinBox *grIndConAnaSpinBox;
    QPushButton *dispGROutGOButton;
    QPushButton *dispGRInMFGOButton;

    void setupUi(QMainWindow *MainWClass)
    {
        if (MainWClass->objectName().isEmpty())
            MainWClass->setObjectName(QString::fromUtf8("MainWClass"));
        MainWClass->resize(614, 367);
        centralwidget = new QWidget(MainWClass);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        horizontalLayoutWidget = new QWidget(centralwidget);
        horizontalLayoutWidget->setObjectName(QString::fromUtf8("horizontalLayoutWidget"));
        horizontalLayoutWidget->setGeometry(QRect(30, 300, 561, 41));
        horizontalLayout = new QHBoxLayout(horizontalLayoutWidget);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        loadSimButton = new QPushButton(horizontalLayoutWidget);
        loadSimButton->setObjectName(QString::fromUtf8("loadSimButton"));

        horizontalLayout->addWidget(loadSimButton);

        loadPSHButton = new QPushButton(horizontalLayoutWidget);
        loadPSHButton->setObjectName(QString::fromUtf8("loadPSHButton"));

        horizontalLayout->addWidget(loadPSHButton);

        quitButton = new QPushButton(horizontalLayoutWidget);
        quitButton->setObjectName(QString::fromUtf8("quitButton"));

        horizontalLayout->addWidget(quitButton);

        gridLayoutWidget_2 = new QWidget(centralwidget);
        gridLayoutWidget_2->setObjectName(QString::fromUtf8("gridLayoutWidget_2"));
        gridLayoutWidget_2->setGeometry(QRect(30, 20, 561, 151));
        gridLayout_2 = new QGridLayout(gridLayoutWidget_2);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        gridLayout_2->setContentsMargins(0, 0, 0, 0);
        label = new QLabel(gridLayoutWidget_2);
        label->setObjectName(QString::fromUtf8("label"));

        gridLayout_2->addWidget(label, 2, 0, 1, 1);

        multiCellStrideBox = new QSpinBox(gridLayoutWidget_2);
        multiCellStrideBox->setObjectName(QString::fromUtf8("multiCellStrideBox"));

        gridLayout_2->addWidget(multiCellStrideBox, 2, 1, 1, 1);

        label_2 = new QLabel(gridLayoutWidget_2);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        gridLayout_2->addWidget(label_2, 3, 0, 1, 1);

        multiCellPageBox = new QSpinBox(gridLayoutWidget_2);
        multiCellPageBox->setObjectName(QString::fromUtf8("multiCellPageBox"));

        gridLayout_2->addWidget(multiCellPageBox, 3, 1, 1, 1);

        label_4 = new QLabel(gridLayoutWidget_2);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        gridLayout_2->addWidget(label_4, 2, 2, 1, 1);

        singleCellNumBox = new QSpinBox(gridLayoutWidget_2);
        singleCellNumBox->setObjectName(QString::fromUtf8("singleCellNumBox"));

        gridLayout_2->addWidget(singleCellNumBox, 2, 3, 1, 1);

        dispCellTypeBox = new QComboBox(gridLayoutWidget_2);
        dispCellTypeBox->setObjectName(QString::fromUtf8("dispCellTypeBox"));

        gridLayout_2->addWidget(dispCellTypeBox, 0, 2, 1, 1);

        label_3 = new QLabel(gridLayoutWidget_2);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        gridLayout_2->addWidget(label_3, 0, 1, 1, 1);

        singleCellNPButton = new QPushButton(gridLayoutWidget_2);
        singleCellNPButton->setObjectName(QString::fromUtf8("singleCellNPButton"));

        gridLayout_2->addWidget(singleCellNPButton, 4, 2, 1, 2);

        multicellNPButton = new QPushButton(gridLayoutWidget_2);
        multicellNPButton->setObjectName(QString::fromUtf8("multicellNPButton"));

        gridLayout_2->addWidget(multicellNPButton, 4, 0, 1, 2);

        calcSpikeRatesButton = new QPushButton(gridLayoutWidget_2);
        calcSpikeRatesButton->setObjectName(QString::fromUtf8("calcSpikeRatesButton"));

        gridLayout_2->addWidget(calcSpikeRatesButton, 1, 1, 1, 1);

        exportSpikeRatesButton = new QPushButton(gridLayoutWidget_2);
        exportSpikeRatesButton->setObjectName(QString::fromUtf8("exportSpikeRatesButton"));

        gridLayout_2->addWidget(exportSpikeRatesButton, 1, 2, 1, 1);

        gridLayoutWidget_3 = new QWidget(centralwidget);
        gridLayoutWidget_3->setObjectName(QString::fromUtf8("gridLayoutWidget_3"));
        gridLayoutWidget_3->setGeometry(QRect(350, 210, 241, 51));
        gridLayout_3 = new QGridLayout(gridLayoutWidget_3);
        gridLayout_3->setObjectName(QString::fromUtf8("gridLayout_3"));
        gridLayout_3->setContentsMargins(0, 0, 0, 0);
        label_5 = new QLabel(gridLayoutWidget_3);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        gridLayout_3->addWidget(label_5, 0, 0, 1, 1);

        pfPCPlastUSTimeSpinBox = new QSpinBox(gridLayoutWidget_3);
        pfPCPlastUSTimeSpinBox->setObjectName(QString::fromUtf8("pfPCPlastUSTimeSpinBox"));

        gridLayout_3->addWidget(pfPCPlastUSTimeSpinBox, 0, 1, 1, 1);

        calcPFPCPlastButton = new QPushButton(gridLayoutWidget_3);
        calcPFPCPlastButton->setObjectName(QString::fromUtf8("calcPFPCPlastButton"));

        gridLayout_3->addWidget(calcPFPCPlastButton, 1, 0, 1, 1);

        exportPFPCPlastActButton = new QPushButton(gridLayoutWidget_3);
        exportPFPCPlastActButton->setObjectName(QString::fromUtf8("exportPFPCPlastActButton"));

        gridLayout_3->addWidget(exportPFPCPlastActButton, 1, 1, 1, 1);

        gridLayoutWidget = new QWidget(centralwidget);
        gridLayoutWidget->setObjectName(QString::fromUtf8("gridLayoutWidget"));
        gridLayoutWidget->setGeometry(QRect(30, 210, 291, 51));
        gridLayout = new QGridLayout(gridLayoutWidget);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        gridLayout->setContentsMargins(0, 0, 0, 0);
        label_6 = new QLabel(gridLayoutWidget);
        label_6->setObjectName(QString::fromUtf8("label_6"));

        gridLayout->addWidget(label_6, 0, 0, 1, 1);

        grIndConAnaSpinBox = new QSpinBox(gridLayoutWidget);
        grIndConAnaSpinBox->setObjectName(QString::fromUtf8("grIndConAnaSpinBox"));

        gridLayout->addWidget(grIndConAnaSpinBox, 0, 1, 1, 1);

        dispGROutGOButton = new QPushButton(gridLayoutWidget);
        dispGROutGOButton->setObjectName(QString::fromUtf8("dispGROutGOButton"));

        gridLayout->addWidget(dispGROutGOButton, 1, 1, 1, 1);

        dispGRInMFGOButton = new QPushButton(gridLayoutWidget);
        dispGRInMFGOButton->setObjectName(QString::fromUtf8("dispGRInMFGOButton"));

        gridLayout->addWidget(dispGRInMFGOButton, 1, 0, 1, 1);

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
        QObject::connect(dispGRInMFGOButton, SIGNAL(clicked()), MainWClass, SLOT(showGRInMFGOPSHs()));
        QObject::connect(dispGROutGOButton, SIGNAL(clicked()), MainWClass, SLOT(showGROutGOPSHs()));

        QMetaObject::connectSlotsByName(MainWClass);
    } // setupUi

    void retranslateUi(QMainWindow *MainWClass)
    {
        MainWClass->setWindowTitle(QApplication::translate("MainWClass", "MainWindow", 0, QApplication::UnicodeUTF8));
        loadSimButton->setText(QApplication::translate("MainWClass", "Load Sim State", 0, QApplication::UnicodeUTF8));
        loadPSHButton->setText(QApplication::translate("MainWClass", "Load PSH data", 0, QApplication::UnicodeUTF8));
        quitButton->setText(QApplication::translate("MainWClass", "quit", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("MainWClass", "multi disp page stride", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("MainWClass", "multi disp page #", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("MainWClass", "single disp cell #", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("MainWClass", "Display cell type", 0, QApplication::UnicodeUTF8));
        singleCellNPButton->setText(QApplication::translate("MainWClass", "new single cell PSH panel", 0, QApplication::UnicodeUTF8));
        multicellNPButton->setText(QApplication::translate("MainWClass", "new multi cell PSH panel", 0, QApplication::UnicodeUTF8));
        calcSpikeRatesButton->setText(QApplication::translate("MainWClass", "calc spike rates", 0, QApplication::UnicodeUTF8));
        exportSpikeRatesButton->setText(QApplication::translate("MainWClass", "export spike rates", 0, QApplication::UnicodeUTF8));
        label_5->setText(QApplication::translate("MainWClass", "PFPC US time", 0, QApplication::UnicodeUTF8));
        calcPFPCPlastButton->setText(QApplication::translate("MainWClass", "calc PFPC", 0, QApplication::UnicodeUTF8));
        exportPFPCPlastActButton->setText(QApplication::translate("MainWClass", "export PFPC", 0, QApplication::UnicodeUTF8));
        label_6->setText(QApplication::translate("MainWClass", "Granule cell #", 0, QApplication::UnicodeUTF8));
        dispGROutGOButton->setText(QApplication::translate("MainWClass", "Disp GROutGO PSHs", 0, QApplication::UnicodeUTF8));
        dispGRInMFGOButton->setText(QApplication::translate("MainWClass", "Disp GRInMFGO PSHs", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class MainWClass: public Ui_MainWClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINW_H
