/********************************************************************************
** Form generated from reading UI file 'mainw.ui'
**
** Created: Thu Sep 15 13:25:33 2011
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
    QLabel *label_13;
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
    QPushButton *viewClusterButton;
    QPushButton *viewClusterCellButton;
    QLabel *label_11;
    QComboBox *clusterCellTypeBox;
    QPushButton *makeClusterButton;

    void setupUi(QMainWindow *MainWClass)
    {
        if (MainWClass->objectName().isEmpty())
            MainWClass->setObjectName(QString::fromUtf8("MainWClass"));
        MainWClass->resize(614, 523);
        centralwidget = new QWidget(MainWClass);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        horizontalLayoutWidget = new QWidget(centralwidget);
        horizontalLayoutWidget->setObjectName(QString::fromUtf8("horizontalLayoutWidget"));
        horizontalLayoutWidget->setGeometry(QRect(30, 450, 561, 41));
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

        gridLayout_2->addWidget(label, 3, 0, 1, 1);

        multiCellStrideBox = new QSpinBox(gridLayoutWidget_2);
        multiCellStrideBox->setObjectName(QString::fromUtf8("multiCellStrideBox"));

        gridLayout_2->addWidget(multiCellStrideBox, 3, 1, 1, 1);

        label_2 = new QLabel(gridLayoutWidget_2);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        gridLayout_2->addWidget(label_2, 4, 0, 1, 1);

        multiCellPageBox = new QSpinBox(gridLayoutWidget_2);
        multiCellPageBox->setObjectName(QString::fromUtf8("multiCellPageBox"));

        gridLayout_2->addWidget(multiCellPageBox, 4, 1, 1, 1);

        label_4 = new QLabel(gridLayoutWidget_2);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        gridLayout_2->addWidget(label_4, 3, 2, 1, 1);

        singleCellNumBox = new QSpinBox(gridLayoutWidget_2);
        singleCellNumBox->setObjectName(QString::fromUtf8("singleCellNumBox"));

        gridLayout_2->addWidget(singleCellNumBox, 3, 3, 1, 1);

        dispCellTypeBox = new QComboBox(gridLayoutWidget_2);
        dispCellTypeBox->setObjectName(QString::fromUtf8("dispCellTypeBox"));

        gridLayout_2->addWidget(dispCellTypeBox, 1, 2, 1, 1);

        label_3 = new QLabel(gridLayoutWidget_2);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        gridLayout_2->addWidget(label_3, 1, 1, 1, 1);

        singleCellNPButton = new QPushButton(gridLayoutWidget_2);
        singleCellNPButton->setObjectName(QString::fromUtf8("singleCellNPButton"));

        gridLayout_2->addWidget(singleCellNPButton, 5, 2, 1, 2);

        multicellNPButton = new QPushButton(gridLayoutWidget_2);
        multicellNPButton->setObjectName(QString::fromUtf8("multicellNPButton"));

        gridLayout_2->addWidget(multicellNPButton, 5, 0, 1, 2);

        label_13 = new QLabel(gridLayoutWidget_2);
        label_13->setObjectName(QString::fromUtf8("label_13"));

        gridLayout_2->addWidget(label_13, 0, 1, 1, 2);

        gridLayoutWidget_3 = new QWidget(centralwidget);
        gridLayoutWidget_3->setObjectName(QString::fromUtf8("gridLayoutWidget_3"));
        gridLayoutWidget_3->setGeometry(QRect(350, 340, 241, 91));
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
        gridLayoutWidget->setGeometry(QRect(30, 340, 311, 91));
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
        gridLayoutWidget_4->setGeometry(QRect(30, 190, 181, 101));
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
        gridLayoutWidget_5->setGeometry(QRect(220, 190, 371, 131));
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

        viewClusterButton = new QPushButton(gridLayoutWidget_5);
        viewClusterButton->setObjectName(QString::fromUtf8("viewClusterButton"));

        gridLayout_5->addWidget(viewClusterButton, 3, 0, 1, 2);

        viewClusterCellButton = new QPushButton(gridLayoutWidget_5);
        viewClusterCellButton->setObjectName(QString::fromUtf8("viewClusterCellButton"));

        gridLayout_5->addWidget(viewClusterCellButton, 3, 2, 1, 2);

        label_11 = new QLabel(gridLayoutWidget_5);
        label_11->setObjectName(QString::fromUtf8("label_11"));

        gridLayout_5->addWidget(label_11, 1, 0, 1, 1);

        clusterCellTypeBox = new QComboBox(gridLayoutWidget_5);
        clusterCellTypeBox->setObjectName(QString::fromUtf8("clusterCellTypeBox"));

        gridLayout_5->addWidget(clusterCellTypeBox, 1, 1, 1, 1);

        makeClusterButton = new QPushButton(gridLayoutWidget_5);
        makeClusterButton->setObjectName(QString::fromUtf8("makeClusterButton"));

        gridLayout_5->addWidget(makeClusterButton, 1, 2, 1, 2);

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
        label_13->setText(QApplication::translate("MainWClass", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Sans Serif'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<table style=\"-qt-table-type: root; margin-top:4px; margin-bottom:4px; margin-left:4px; margin-right:4px;\">\n"
"<tr>\n"
"<td style=\"border: none;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">PSH display</span></p></td></tr></table></body></html>", 0, QApplication::UnicodeUTF8));
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
        viewClusterButton->setText(QApplication::translate("MainWClass", "view cluster motif", 0, QApplication::UnicodeUTF8));
        viewClusterCellButton->setText(QApplication::translate("MainWClass", "view cluster cell", 0, QApplication::UnicodeUTF8));
        label_11->setText(QApplication::translate("MainWClass", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Sans Serif'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<table style=\"-qt-table-type: root; margin-top:4px; margin-bottom:4px; margin-left:4px; margin-right:4px;\">\n"
"<tr>\n"
"<td style=\"border: none;\">\n"
"<p align=\"right\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Cell type</p></td></tr></table></body></html>", 0, QApplication::UnicodeUTF8));
        makeClusterButton->setText(QApplication::translate("MainWClass", "make clusters", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class MainWClass: public Ui_MainWClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINW_H
