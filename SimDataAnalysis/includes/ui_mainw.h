/********************************************************************************
** Form generated from reading UI file 'mainw.ui'
**
** Created: Fri May 11 08:33:29 2012
**      by: Qt User Interface Compiler version 4.7.4
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
    QLabel *label_3;
    QLabel *label_13;
    QComboBox *dispCellTypeBox;
    QLabel *label;
    QSpinBox *multiCellStrideBox;
    QSpinBox *multiCellPageBox;
    QLabel *label_2;
    QPushButton *multicellNPButton;
    QPushButton *singleCellNPButton;
    QLabel *label_4;
    QSpinBox *singleCellNumBox;
    QPushButton *fuseBinsButton;
    QLabel *label_18;
    QSpinBox *numBinsFuseBox;
    QWidget *gridLayoutWidget_3;
    QGridLayout *gridLayout_3;
    QLabel *label_5;
    QSpinBox *pfPCPlastUSTimeSpinBox;
    QPushButton *calcPFPCPlastButton;
    QPushButton *exportPFPCPlastActButton;
    QLabel *label_8;
    QWidget *gridLayoutWidget;
    QGridLayout *gridLayout;
    QLabel *label_6;
    QSpinBox *grIndConAnaSpinBox;
    QPushButton *dispGROutGOButton;
    QPushButton *dispGRInMFGOButton;
    QLabel *label_7;
    QWidget *gridLayoutWidget_4;
    QGridLayout *gridLayout_4;
    QPushButton *calcSpikeRatesButton;
    QPushButton *exportSpikeRatesButton;
    QLabel *label_9;
    QComboBox *comboBox;
    QLabel *label_10;
    QWidget *gridLayoutWidget_5;
    QGridLayout *gridLayout_5;
    QLabel *label_12;
    QLabel *label_14;
    QSpinBox *clusterNumBox;
    QLabel *label_15;
    QSpinBox *clusterCellNumBox;
    QPushButton *newClusterPButton;
    QPushButton *newClusterCellPButton;
    QLabel *label_11;
    QComboBox *clusterCellTypeBox;
    QPushButton *makeClusterButton;
    QWidget *gridLayoutWidget_6;
    QGridLayout *gridLayout_6;
    QLabel *label_16;
    QLabel *label_17;
    QPushButton *dispSpatialButton;
    QSpinBox *spatialBinNBox;
    QPushButton *exportInNetBinButton;
    QWidget *gridLayoutWidget_7;
    QGridLayout *gridLayout_7;
    QLabel *label_20;
    QLabel *label_21;
    QLabel *label_22;
    QSpinBox *dimension1StartBNBox;
    QSpinBox *dimension1EndBNBox;
    QSpinBox *dimension2StartBNBox;
    QSpinBox *dimension2EndBNBox;
    QSpinBox *dimension3StartBNBox;
    QSpinBox *dimension3EndBNBox;
    QLabel *label_23;
    QLabel *label_24;
    QPushButton *generate3DClusterButton;
    QLabel *label_19;
    QLabel *label_25;
    QComboBox *cluster3DCellTypeBox;

    void setupUi(QMainWindow *MainWClass)
    {
        if (MainWClass->objectName().isEmpty())
            MainWClass->setObjectName(QString::fromUtf8("MainWClass"));
        MainWClass->resize(608, 699);
        centralwidget = new QWidget(MainWClass);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        horizontalLayoutWidget = new QWidget(centralwidget);
        horizontalLayoutWidget->setObjectName(QString::fromUtf8("horizontalLayoutWidget"));
        horizontalLayoutWidget->setGeometry(QRect(30, 640, 561, 41));
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
        gridLayoutWidget_2->setGeometry(QRect(30, 0, 561, 171));
        gridLayout_2 = new QGridLayout(gridLayoutWidget_2);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        gridLayout_2->setContentsMargins(0, 0, 0, 0);
        label_3 = new QLabel(gridLayoutWidget_2);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        gridLayout_2->addWidget(label_3, 1, 1, 1, 1);

        label_13 = new QLabel(gridLayoutWidget_2);
        label_13->setObjectName(QString::fromUtf8("label_13"));

        gridLayout_2->addWidget(label_13, 1, 0, 1, 1);

        dispCellTypeBox = new QComboBox(gridLayoutWidget_2);
        dispCellTypeBox->setObjectName(QString::fromUtf8("dispCellTypeBox"));

        gridLayout_2->addWidget(dispCellTypeBox, 1, 2, 1, 1);

        label = new QLabel(gridLayoutWidget_2);
        label->setObjectName(QString::fromUtf8("label"));

        gridLayout_2->addWidget(label, 2, 0, 1, 1);

        multiCellStrideBox = new QSpinBox(gridLayoutWidget_2);
        multiCellStrideBox->setObjectName(QString::fromUtf8("multiCellStrideBox"));

        gridLayout_2->addWidget(multiCellStrideBox, 2, 1, 1, 1);

        multiCellPageBox = new QSpinBox(gridLayoutWidget_2);
        multiCellPageBox->setObjectName(QString::fromUtf8("multiCellPageBox"));

        gridLayout_2->addWidget(multiCellPageBox, 3, 1, 1, 1);

        label_2 = new QLabel(gridLayoutWidget_2);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        gridLayout_2->addWidget(label_2, 3, 0, 1, 1);

        multicellNPButton = new QPushButton(gridLayoutWidget_2);
        multicellNPButton->setObjectName(QString::fromUtf8("multicellNPButton"));

        gridLayout_2->addWidget(multicellNPButton, 4, 0, 1, 2);

        singleCellNPButton = new QPushButton(gridLayoutWidget_2);
        singleCellNPButton->setObjectName(QString::fromUtf8("singleCellNPButton"));

        gridLayout_2->addWidget(singleCellNPButton, 4, 2, 1, 2);

        label_4 = new QLabel(gridLayoutWidget_2);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        gridLayout_2->addWidget(label_4, 3, 2, 1, 1);

        singleCellNumBox = new QSpinBox(gridLayoutWidget_2);
        singleCellNumBox->setObjectName(QString::fromUtf8("singleCellNumBox"));

        gridLayout_2->addWidget(singleCellNumBox, 3, 3, 1, 1);

        fuseBinsButton = new QPushButton(gridLayoutWidget_2);
        fuseBinsButton->setObjectName(QString::fromUtf8("fuseBinsButton"));

        gridLayout_2->addWidget(fuseBinsButton, 5, 2, 1, 2);

        label_18 = new QLabel(gridLayoutWidget_2);
        label_18->setObjectName(QString::fromUtf8("label_18"));

        gridLayout_2->addWidget(label_18, 5, 0, 1, 1);

        numBinsFuseBox = new QSpinBox(gridLayoutWidget_2);
        numBinsFuseBox->setObjectName(QString::fromUtf8("numBinsFuseBox"));

        gridLayout_2->addWidget(numBinsFuseBox, 5, 1, 1, 1);

        gridLayoutWidget_3 = new QWidget(centralwidget);
        gridLayoutWidget_3->setObjectName(QString::fromUtf8("gridLayoutWidget_3"));
        gridLayoutWidget_3->setGeometry(QRect(350, 300, 241, 91));
        gridLayout_3 = new QGridLayout(gridLayoutWidget_3);
        gridLayout_3->setObjectName(QString::fromUtf8("gridLayout_3"));
        gridLayout_3->setContentsMargins(0, 0, 0, 0);
        label_5 = new QLabel(gridLayoutWidget_3);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        gridLayout_3->addWidget(label_5, 1, 0, 1, 1);

        pfPCPlastUSTimeSpinBox = new QSpinBox(gridLayoutWidget_3);
        pfPCPlastUSTimeSpinBox->setObjectName(QString::fromUtf8("pfPCPlastUSTimeSpinBox"));

        gridLayout_3->addWidget(pfPCPlastUSTimeSpinBox, 1, 1, 1, 1);

        calcPFPCPlastButton = new QPushButton(gridLayoutWidget_3);
        calcPFPCPlastButton->setObjectName(QString::fromUtf8("calcPFPCPlastButton"));

        gridLayout_3->addWidget(calcPFPCPlastButton, 2, 0, 1, 1);

        exportPFPCPlastActButton = new QPushButton(gridLayoutWidget_3);
        exportPFPCPlastActButton->setObjectName(QString::fromUtf8("exportPFPCPlastActButton"));

        gridLayout_3->addWidget(exportPFPCPlastActButton, 2, 1, 1, 1);

        label_8 = new QLabel(gridLayoutWidget_3);
        label_8->setObjectName(QString::fromUtf8("label_8"));

        gridLayout_3->addWidget(label_8, 0, 0, 1, 2);

        gridLayoutWidget = new QWidget(centralwidget);
        gridLayoutWidget->setObjectName(QString::fromUtf8("gridLayoutWidget"));
        gridLayoutWidget->setGeometry(QRect(30, 300, 311, 91));
        gridLayout = new QGridLayout(gridLayoutWidget);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        gridLayout->setContentsMargins(0, 0, 0, 0);
        label_6 = new QLabel(gridLayoutWidget);
        label_6->setObjectName(QString::fromUtf8("label_6"));

        gridLayout->addWidget(label_6, 2, 0, 1, 1);

        grIndConAnaSpinBox = new QSpinBox(gridLayoutWidget);
        grIndConAnaSpinBox->setObjectName(QString::fromUtf8("grIndConAnaSpinBox"));

        gridLayout->addWidget(grIndConAnaSpinBox, 2, 1, 1, 1);

        dispGROutGOButton = new QPushButton(gridLayoutWidget);
        dispGROutGOButton->setObjectName(QString::fromUtf8("dispGROutGOButton"));

        gridLayout->addWidget(dispGROutGOButton, 3, 1, 1, 1);

        dispGRInMFGOButton = new QPushButton(gridLayoutWidget);
        dispGRInMFGOButton->setObjectName(QString::fromUtf8("dispGRInMFGOButton"));

        gridLayout->addWidget(dispGRInMFGOButton, 3, 0, 1, 1);

        label_7 = new QLabel(gridLayoutWidget);
        label_7->setObjectName(QString::fromUtf8("label_7"));

        gridLayout->addWidget(label_7, 1, 0, 1, 2);

        gridLayoutWidget_4 = new QWidget(centralwidget);
        gridLayoutWidget_4->setObjectName(QString::fromUtf8("gridLayoutWidget_4"));
        gridLayoutWidget_4->setGeometry(QRect(30, 170, 181, 101));
        gridLayout_4 = new QGridLayout(gridLayoutWidget_4);
        gridLayout_4->setObjectName(QString::fromUtf8("gridLayout_4"));
        gridLayout_4->setContentsMargins(0, 0, 0, 0);
        calcSpikeRatesButton = new QPushButton(gridLayoutWidget_4);
        calcSpikeRatesButton->setObjectName(QString::fromUtf8("calcSpikeRatesButton"));

        gridLayout_4->addWidget(calcSpikeRatesButton, 2, 0, 1, 1);

        exportSpikeRatesButton = new QPushButton(gridLayoutWidget_4);
        exportSpikeRatesButton->setObjectName(QString::fromUtf8("exportSpikeRatesButton"));

        gridLayout_4->addWidget(exportSpikeRatesButton, 2, 1, 1, 1);

        label_9 = new QLabel(gridLayoutWidget_4);
        label_9->setObjectName(QString::fromUtf8("label_9"));

        gridLayout_4->addWidget(label_9, 1, 0, 1, 1);

        comboBox = new QComboBox(gridLayoutWidget_4);
        comboBox->setObjectName(QString::fromUtf8("comboBox"));

        gridLayout_4->addWidget(comboBox, 1, 1, 1, 1);

        label_10 = new QLabel(gridLayoutWidget_4);
        label_10->setObjectName(QString::fromUtf8("label_10"));

        gridLayout_4->addWidget(label_10, 0, 0, 1, 2);

        gridLayoutWidget_5 = new QWidget(centralwidget);
        gridLayoutWidget_5->setObjectName(QString::fromUtf8("gridLayoutWidget_5"));
        gridLayoutWidget_5->setGeometry(QRect(220, 170, 371, 131));
        gridLayout_5 = new QGridLayout(gridLayoutWidget_5);
        gridLayout_5->setObjectName(QString::fromUtf8("gridLayout_5"));
        gridLayout_5->setContentsMargins(0, 0, 0, 0);
        label_12 = new QLabel(gridLayoutWidget_5);
        label_12->setObjectName(QString::fromUtf8("label_12"));

        gridLayout_5->addWidget(label_12, 0, 0, 1, 4);

        label_14 = new QLabel(gridLayoutWidget_5);
        label_14->setObjectName(QString::fromUtf8("label_14"));

        gridLayout_5->addWidget(label_14, 2, 0, 1, 1);

        clusterNumBox = new QSpinBox(gridLayoutWidget_5);
        clusterNumBox->setObjectName(QString::fromUtf8("clusterNumBox"));

        gridLayout_5->addWidget(clusterNumBox, 2, 1, 1, 1);

        label_15 = new QLabel(gridLayoutWidget_5);
        label_15->setObjectName(QString::fromUtf8("label_15"));

        gridLayout_5->addWidget(label_15, 2, 2, 1, 1);

        clusterCellNumBox = new QSpinBox(gridLayoutWidget_5);
        clusterCellNumBox->setObjectName(QString::fromUtf8("clusterCellNumBox"));

        gridLayout_5->addWidget(clusterCellNumBox, 2, 3, 1, 1);

        newClusterPButton = new QPushButton(gridLayoutWidget_5);
        newClusterPButton->setObjectName(QString::fromUtf8("newClusterPButton"));

        gridLayout_5->addWidget(newClusterPButton, 3, 0, 1, 2);

        newClusterCellPButton = new QPushButton(gridLayoutWidget_5);
        newClusterCellPButton->setObjectName(QString::fromUtf8("newClusterCellPButton"));

        gridLayout_5->addWidget(newClusterCellPButton, 3, 2, 1, 2);

        label_11 = new QLabel(gridLayoutWidget_5);
        label_11->setObjectName(QString::fromUtf8("label_11"));

        gridLayout_5->addWidget(label_11, 1, 0, 1, 1);

        clusterCellTypeBox = new QComboBox(gridLayoutWidget_5);
        clusterCellTypeBox->setObjectName(QString::fromUtf8("clusterCellTypeBox"));

        gridLayout_5->addWidget(clusterCellTypeBox, 1, 1, 1, 1);

        makeClusterButton = new QPushButton(gridLayoutWidget_5);
        makeClusterButton->setObjectName(QString::fromUtf8("makeClusterButton"));

        gridLayout_5->addWidget(makeClusterButton, 1, 2, 1, 2);

        gridLayoutWidget_6 = new QWidget(centralwidget);
        gridLayoutWidget_6->setObjectName(QString::fromUtf8("gridLayoutWidget_6"));
        gridLayoutWidget_6->setGeometry(QRect(160, 390, 311, 98));
        gridLayout_6 = new QGridLayout(gridLayoutWidget_6);
        gridLayout_6->setObjectName(QString::fromUtf8("gridLayout_6"));
        gridLayout_6->setContentsMargins(0, 0, 0, 0);
        label_16 = new QLabel(gridLayoutWidget_6);
        label_16->setObjectName(QString::fromUtf8("label_16"));

        gridLayout_6->addWidget(label_16, 0, 0, 1, 2);

        label_17 = new QLabel(gridLayoutWidget_6);
        label_17->setObjectName(QString::fromUtf8("label_17"));

        gridLayout_6->addWidget(label_17, 1, 0, 1, 1);

        dispSpatialButton = new QPushButton(gridLayoutWidget_6);
        dispSpatialButton->setObjectName(QString::fromUtf8("dispSpatialButton"));

        gridLayout_6->addWidget(dispSpatialButton, 2, 0, 1, 1);

        spatialBinNBox = new QSpinBox(gridLayoutWidget_6);
        spatialBinNBox->setObjectName(QString::fromUtf8("spatialBinNBox"));

        gridLayout_6->addWidget(spatialBinNBox, 1, 1, 1, 1);

        exportInNetBinButton = new QPushButton(gridLayoutWidget_6);
        exportInNetBinButton->setObjectName(QString::fromUtf8("exportInNetBinButton"));

        gridLayout_6->addWidget(exportInNetBinButton, 2, 1, 1, 1);

        gridLayoutWidget_7 = new QWidget(centralwidget);
        gridLayoutWidget_7->setObjectName(QString::fromUtf8("gridLayoutWidget_7"));
        gridLayoutWidget_7->setGeometry(QRect(40, 490, 541, 151));
        gridLayout_7 = new QGridLayout(gridLayoutWidget_7);
        gridLayout_7->setObjectName(QString::fromUtf8("gridLayout_7"));
        gridLayout_7->setContentsMargins(0, 0, 0, 0);
        label_20 = new QLabel(gridLayoutWidget_7);
        label_20->setObjectName(QString::fromUtf8("label_20"));

        gridLayout_7->addWidget(label_20, 2, 1, 1, 1);

        label_21 = new QLabel(gridLayoutWidget_7);
        label_21->setObjectName(QString::fromUtf8("label_21"));

        gridLayout_7->addWidget(label_21, 2, 2, 1, 1);

        label_22 = new QLabel(gridLayoutWidget_7);
        label_22->setObjectName(QString::fromUtf8("label_22"));

        gridLayout_7->addWidget(label_22, 2, 3, 1, 1);

        dimension1StartBNBox = new QSpinBox(gridLayoutWidget_7);
        dimension1StartBNBox->setObjectName(QString::fromUtf8("dimension1StartBNBox"));

        gridLayout_7->addWidget(dimension1StartBNBox, 3, 1, 1, 1);

        dimension1EndBNBox = new QSpinBox(gridLayoutWidget_7);
        dimension1EndBNBox->setObjectName(QString::fromUtf8("dimension1EndBNBox"));

        gridLayout_7->addWidget(dimension1EndBNBox, 4, 1, 1, 1);

        dimension2StartBNBox = new QSpinBox(gridLayoutWidget_7);
        dimension2StartBNBox->setObjectName(QString::fromUtf8("dimension2StartBNBox"));

        gridLayout_7->addWidget(dimension2StartBNBox, 3, 2, 1, 1);

        dimension2EndBNBox = new QSpinBox(gridLayoutWidget_7);
        dimension2EndBNBox->setObjectName(QString::fromUtf8("dimension2EndBNBox"));

        gridLayout_7->addWidget(dimension2EndBNBox, 4, 2, 1, 1);

        dimension3StartBNBox = new QSpinBox(gridLayoutWidget_7);
        dimension3StartBNBox->setObjectName(QString::fromUtf8("dimension3StartBNBox"));

        gridLayout_7->addWidget(dimension3StartBNBox, 3, 3, 1, 1);

        dimension3EndBNBox = new QSpinBox(gridLayoutWidget_7);
        dimension3EndBNBox->setObjectName(QString::fromUtf8("dimension3EndBNBox"));

        gridLayout_7->addWidget(dimension3EndBNBox, 4, 3, 1, 1);

        label_23 = new QLabel(gridLayoutWidget_7);
        label_23->setObjectName(QString::fromUtf8("label_23"));

        gridLayout_7->addWidget(label_23, 3, 0, 1, 1);

        label_24 = new QLabel(gridLayoutWidget_7);
        label_24->setObjectName(QString::fromUtf8("label_24"));

        gridLayout_7->addWidget(label_24, 4, 0, 1, 1);

        generate3DClusterButton = new QPushButton(gridLayoutWidget_7);
        generate3DClusterButton->setObjectName(QString::fromUtf8("generate3DClusterButton"));

        gridLayout_7->addWidget(generate3DClusterButton, 5, 1, 1, 2);

        label_19 = new QLabel(gridLayoutWidget_7);
        label_19->setObjectName(QString::fromUtf8("label_19"));

        gridLayout_7->addWidget(label_19, 1, 0, 1, 1);

        label_25 = new QLabel(gridLayoutWidget_7);
        label_25->setObjectName(QString::fromUtf8("label_25"));

        gridLayout_7->addWidget(label_25, 1, 1, 1, 1);

        cluster3DCellTypeBox = new QComboBox(gridLayoutWidget_7);
        cluster3DCellTypeBox->setObjectName(QString::fromUtf8("cluster3DCellTypeBox"));

        gridLayout_7->addWidget(cluster3DCellTypeBox, 1, 2, 1, 1);

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
        QObject::connect(clusterNumBox, SIGNAL(valueChanged(int)), MainWClass, SLOT(updateClusterDisp(int)));
        QObject::connect(clusterCellNumBox, SIGNAL(valueChanged(int)), MainWClass, SLOT(updateClusterCellDisp(int)));
        QObject::connect(makeClusterButton, SIGNAL(clicked()), MainWClass, SLOT(makeClusters()));
        QObject::connect(clusterCellTypeBox, SIGNAL(currentIndexChanged(int)), MainWClass, SLOT(updateClusterCellType(int)));
        QObject::connect(newClusterPButton, SIGNAL(clicked()), MainWClass, SLOT(dispClusterNP()));
        QObject::connect(newClusterCellPButton, SIGNAL(clicked()), MainWClass, SLOT(dispClusterCellNP()));
        QObject::connect(dispSpatialButton, SIGNAL(clicked()), MainWClass, SLOT(dispInNetSpatialNP()));
        QObject::connect(spatialBinNBox, SIGNAL(valueChanged(int)), MainWClass, SLOT(updateInNetSpatial(int)));
        QObject::connect(exportInNetBinButton, SIGNAL(clicked()), MainWClass, SLOT(exportInNetBinData()));
        QObject::connect(generate3DClusterButton, SIGNAL(clicked()), MainWClass, SLOT(generate3DClusterData()));

        QMetaObject::connectSlotsByName(MainWClass);
    } // setupUi

    void retranslateUi(QMainWindow *MainWClass)
    {
        MainWClass->setWindowTitle(QApplication::translate("MainWClass", "MainWindow", 0, QApplication::UnicodeUTF8));
        loadSimButton->setText(QApplication::translate("MainWClass", "Load Sim State", 0, QApplication::UnicodeUTF8));
        loadPSHButton->setText(QApplication::translate("MainWClass", "Load PSH data", 0, QApplication::UnicodeUTF8));
        quitButton->setText(QApplication::translate("MainWClass", "quit", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("MainWClass", "Display cell type", 0, QApplication::UnicodeUTF8));
        label_13->setText(QApplication::translate("MainWClass", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Sans Serif'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<table style=\"-qt-table-type: root; margin-top:4px; margin-bottom:4px; margin-left:4px; margin-right:4px;\">\n"
"<tr>\n"
"<td style=\"border: none;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">PSH display</span></p></td></tr></table></body></html>", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("MainWClass", "multi disp page stride", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("MainWClass", "multi disp page #", 0, QApplication::UnicodeUTF8));
        multicellNPButton->setText(QApplication::translate("MainWClass", "new multi cell PSH panel", 0, QApplication::UnicodeUTF8));
        singleCellNPButton->setText(QApplication::translate("MainWClass", "new single cell PSH panel", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("MainWClass", "single disp cell #", 0, QApplication::UnicodeUTF8));
        fuseBinsButton->setText(QApplication::translate("MainWClass", "fuseBins", 0, QApplication::UnicodeUTF8));
        label_18->setText(QApplication::translate("MainWClass", "#bins to fuse", 0, QApplication::UnicodeUTF8));
        label_5->setText(QApplication::translate("MainWClass", "PFPC US time", 0, QApplication::UnicodeUTF8));
        calcPFPCPlastButton->setText(QApplication::translate("MainWClass", "calc PFPC", 0, QApplication::UnicodeUTF8));
        exportPFPCPlastActButton->setText(QApplication::translate("MainWClass", "export PFPC", 0, QApplication::UnicodeUTF8));
        label_8->setText(QApplication::translate("MainWClass", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Sans Serif'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<table style=\"-qt-table-type: root; margin-top:4px; margin-bottom:4px; margin-left:4px; margin-right:4px;\">\n"
"<tr>\n"
"<td style=\"border: none;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">PFPC plasticity analysis</span></p></td></tr></table></body></html>", 0, QApplication::UnicodeUTF8));
        label_6->setText(QApplication::translate("MainWClass", "Granule cell #", 0, QApplication::UnicodeUTF8));
        dispGROutGOButton->setText(QApplication::translate("MainWClass", "Disp GROutGO PSHs", 0, QApplication::UnicodeUTF8));
        dispGRInMFGOButton->setText(QApplication::translate("MainWClass", "Disp GRInMFGO PSHs", 0, QApplication::UnicodeUTF8));
        label_7->setText(QApplication::translate("MainWClass", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Sans Serif'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<table style=\"-qt-table-type: root; margin-top:4px; margin-bottom:4px; margin-left:4px; margin-right:4px;\">\n"
"<tr>\n"
"<td style=\"border: none;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">Granule cell connectivity analysis</span></p></td></tr></table></body></html>", 0, QApplication::UnicodeUTF8));
        calcSpikeRatesButton->setText(QApplication::translate("MainWClass", "calc rates", 0, QApplication::UnicodeUTF8));
        exportSpikeRatesButton->setText(QApplication::translate("MainWClass", "export rates", 0, QApplication::UnicodeUTF8));
        label_9->setText(QApplication::translate("MainWClass", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Sans Serif'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<table style=\"-qt-table-type: root; margin-top:4px; margin-bottom:4px; margin-left:4px; margin-right:4px;\">\n"
"<tr>\n"
"<td style=\"border: none;\">\n"
"<p align=\"right\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Cell type</p></td></tr></table></body></html>", 0, QApplication::UnicodeUTF8));
        label_10->setText(QApplication::translate("MainWClass", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Sans Serif'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<table style=\"-qt-table-type: root; margin-top:4px; margin-bottom:4px; margin-left:4px; margin-right:4px;\">\n"
"<tr>\n"
"<td style=\"border: none;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">Spike rate analysis</span></p></td></tr></table></body></html>", 0, QApplication::UnicodeUTF8));
        label_12->setText(QApplication::translate("MainWClass", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Sans Serif'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<table border=\"0\" style=\"-qt-table-type: root; margin-top:4px; margin-bottom:4px; margin-left:4px; margin-right:4px;\">\n"
"<tr>\n"
"<td style=\"border: none;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">PSH cluster analysis</span></p></td></tr></table></body></html>", 0, QApplication::UnicodeUTF8));
        label_14->setText(QApplication::translate("MainWClass", "cluster #", 0, QApplication::UnicodeUTF8));
        label_15->setText(QApplication::translate("MainWClass", "cluster cell #", 0, QApplication::UnicodeUTF8));
        newClusterPButton->setText(QApplication::translate("MainWClass", "new cluster panel", 0, QApplication::UnicodeUTF8));
        newClusterCellPButton->setText(QApplication::translate("MainWClass", "new cluster cell panel", 0, QApplication::UnicodeUTF8));
        label_11->setText(QApplication::translate("MainWClass", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Sans Serif'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<table style=\"-qt-table-type: root; margin-top:4px; margin-bottom:4px; margin-left:4px; margin-right:4px;\">\n"
"<tr>\n"
"<td style=\"border: none;\">\n"
"<p align=\"right\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Cell type</p></td></tr></table></body></html>", 0, QApplication::UnicodeUTF8));
        makeClusterButton->setText(QApplication::translate("MainWClass", "make clusters", 0, QApplication::UnicodeUTF8));
        label_16->setText(QApplication::translate("MainWClass", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Sans Serif'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<table style=\"-qt-table-type: root; margin-top:4px; margin-bottom:4px; margin-left:4px; margin-right:4px;\">\n"
"<tr>\n"
"<td style=\"border: none;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">Input net spatial view</span></p></td></tr></table></body></html>", 0, QApplication::UnicodeUTF8));
        label_17->setText(QApplication::translate("MainWClass", "bin #", 0, QApplication::UnicodeUTF8));
        dispSpatialButton->setText(QApplication::translate("MainWClass", "Display Spatial", 0, QApplication::UnicodeUTF8));
        exportInNetBinButton->setText(QApplication::translate("MainWClass", "Export bin data", 0, QApplication::UnicodeUTF8));
        label_20->setText(QApplication::translate("MainWClass", "dimension1", 0, QApplication::UnicodeUTF8));
        label_21->setText(QApplication::translate("MainWClass", "dimension2", 0, QApplication::UnicodeUTF8));
        label_22->setText(QApplication::translate("MainWClass", "dimension3", 0, QApplication::UnicodeUTF8));
        label_23->setText(QApplication::translate("MainWClass", "startBin#", 0, QApplication::UnicodeUTF8));
        label_24->setText(QApplication::translate("MainWClass", "endBin#", 0, QApplication::UnicodeUTF8));
        generate3DClusterButton->setText(QApplication::translate("MainWClass", "generate 3D cluster data", 0, QApplication::UnicodeUTF8));
        label_19->setText(QApplication::translate("MainWClass", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Sans Serif'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<table border=\"0\" style=\"-qt-table-type: root; margin-top:4px; margin-bottom:4px; margin-left:4px; margin-right:4px;\">\n"
"<tr>\n"
"<td style=\"border: none;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">3D clustering</span></p></td></tr></table></body></html>", 0, QApplication::UnicodeUTF8));
        label_25->setText(QApplication::translate("MainWClass", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Sans Serif'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<table style=\"-qt-table-type: root; margin-top:4px; margin-bottom:4px; margin-left:4px; margin-right:4px;\">\n"
"<tr>\n"
"<td style=\"border: none;\">\n"
"<p align=\"right\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Cell Type</p></td></tr></table></body></html>", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class MainWClass: public Ui_MainWClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINW_H
