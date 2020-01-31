/********************************************************************************
** Form generated from reading UI file 'simctrlp.ui'
**
** Created: Fri Apr 8 14:04:28 2011
**      by: Qt User Interface Compiler version 4.7.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_SIMCTRLP_H
#define UI_SIMCTRLP_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QComboBox>
#include <QtGui/QFormLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QPushButton>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_SimCtrlPClass
{
public:
    QWidget *formLayoutWidget;
    QFormLayout *formLayout;
    QLabel *displayLab;
    QComboBox *dispMode;
    QWidget *verticalLayoutWidget;
    QVBoxLayout *verticalLayout;
    QPushButton *startSim;
    QPushButton *pauseSim;
    QPushButton *stopSim;
    QPushButton *exportPSH;
    QPushButton *exportSim;
    QWidget *verticalLayoutWidget_2;
    QVBoxLayout *verticalLayout_2;
    QCheckBox *calcSRHist;
    QCheckBox *dispPSH;
    QCheckBox *dispRaster;
    QCheckBox *dispGRGOActs;
    QPushButton *dispSpikeRates;

    void setupUi(QWidget *SimCtrlPClass)
    {
        if (SimCtrlPClass->objectName().isEmpty())
            SimCtrlPClass->setObjectName(QString::fromUtf8("SimCtrlPClass"));
        SimCtrlPClass->resize(382, 190);
        formLayoutWidget = new QWidget(SimCtrlPClass);
        formLayoutWidget->setObjectName(QString::fromUtf8("formLayoutWidget"));
        formLayoutWidget->setGeometry(QRect(10, 0, 231, 25));
        formLayout = new QFormLayout(formLayoutWidget);
        formLayout->setSpacing(6);
        formLayout->setContentsMargins(11, 11, 11, 11);
        formLayout->setObjectName(QString::fromUtf8("formLayout"));
        formLayout->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
        formLayout->setContentsMargins(0, 0, 0, 0);
        displayLab = new QLabel(formLayoutWidget);
        displayLab->setObjectName(QString::fromUtf8("displayLab"));

        formLayout->setWidget(0, QFormLayout::LabelRole, displayLab);

        dispMode = new QComboBox(formLayoutWidget);
        dispMode->setObjectName(QString::fromUtf8("dispMode"));

        formLayout->setWidget(0, QFormLayout::FieldRole, dispMode);

        verticalLayoutWidget = new QWidget(SimCtrlPClass);
        verticalLayoutWidget->setObjectName(QString::fromUtf8("verticalLayoutWidget"));
        verticalLayoutWidget->setGeometry(QRect(250, 0, 121, 171));
        verticalLayout = new QVBoxLayout(verticalLayoutWidget);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        startSim = new QPushButton(verticalLayoutWidget);
        startSim->setObjectName(QString::fromUtf8("startSim"));

        verticalLayout->addWidget(startSim);

        pauseSim = new QPushButton(verticalLayoutWidget);
        pauseSim->setObjectName(QString::fromUtf8("pauseSim"));

        verticalLayout->addWidget(pauseSim);

        stopSim = new QPushButton(verticalLayoutWidget);
        stopSim->setObjectName(QString::fromUtf8("stopSim"));

        verticalLayout->addWidget(stopSim);

        exportPSH = new QPushButton(verticalLayoutWidget);
        exportPSH->setObjectName(QString::fromUtf8("exportPSH"));

        verticalLayout->addWidget(exportPSH);

        exportSim = new QPushButton(verticalLayoutWidget);
        exportSim->setObjectName(QString::fromUtf8("exportSim"));

        verticalLayout->addWidget(exportSim);

        verticalLayoutWidget_2 = new QWidget(SimCtrlPClass);
        verticalLayoutWidget_2->setObjectName(QString::fromUtf8("verticalLayoutWidget_2"));
        verticalLayoutWidget_2->setGeometry(QRect(30, 30, 191, 141));
        verticalLayout_2 = new QVBoxLayout(verticalLayoutWidget_2);
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setContentsMargins(11, 11, 11, 11);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
        calcSRHist = new QCheckBox(verticalLayoutWidget_2);
        calcSRHist->setObjectName(QString::fromUtf8("calcSRHist"));

        verticalLayout_2->addWidget(calcSRHist);

        dispPSH = new QCheckBox(verticalLayoutWidget_2);
        dispPSH->setObjectName(QString::fromUtf8("dispPSH"));

        verticalLayout_2->addWidget(dispPSH);

        dispRaster = new QCheckBox(verticalLayoutWidget_2);
        dispRaster->setObjectName(QString::fromUtf8("dispRaster"));

        verticalLayout_2->addWidget(dispRaster);

        dispGRGOActs = new QCheckBox(verticalLayoutWidget_2);
        dispGRGOActs->setObjectName(QString::fromUtf8("dispGRGOActs"));

        verticalLayout_2->addWidget(dispGRGOActs);

        dispSpikeRates = new QPushButton(verticalLayoutWidget_2);
        dispSpikeRates->setObjectName(QString::fromUtf8("dispSpikeRates"));

        verticalLayout_2->addWidget(dispSpikeRates);


        retranslateUi(SimCtrlPClass);
        QObject::connect(startSim, SIGNAL(clicked()), SimCtrlPClass, SLOT(startSim()));
        QObject::connect(pauseSim, SIGNAL(clicked()), SimCtrlPClass, SLOT(pauseSim()));
        QObject::connect(stopSim, SIGNAL(clicked()), SimCtrlPClass, SLOT(stopSim()));
        QObject::connect(dispMode, SIGNAL(activated(int)), SimCtrlPClass, SLOT(changeDispMode(int)));
        QObject::connect(dispPSH, SIGNAL(stateChanged(int)), SimCtrlPClass, SLOT(changePSHMode(int)));
        QObject::connect(dispSpikeRates, SIGNAL(clicked()), SimCtrlPClass, SLOT(dispSpikeRates()));
        QObject::connect(exportPSH, SIGNAL(clicked()), SimCtrlPClass, SLOT(exportPSH()));
        QObject::connect(dispGRGOActs, SIGNAL(stateChanged(int)), SimCtrlPClass, SLOT(changeActMode(int)));
        QObject::connect(exportSim, SIGNAL(clicked()), SimCtrlPClass, SLOT(exportSim()));
        QObject::connect(dispRaster, SIGNAL(stateChanged(int)), SimCtrlPClass, SLOT(changeRasterMode(int)));
        QObject::connect(calcSRHist, SIGNAL(stateChanged(int)), SimCtrlPClass, SLOT(changeSRHistMode(int)));

        QMetaObject::connectSlotsByName(SimCtrlPClass);
    } // setupUi

    void retranslateUi(QWidget *SimCtrlPClass)
    {
        SimCtrlPClass->setWindowTitle(QApplication::translate("SimCtrlPClass", "SimCtrlP", 0, QApplication::UnicodeUTF8));
        displayLab->setText(QApplication::translate("SimCtrlPClass", "display", 0, QApplication::UnicodeUTF8));
        startSim->setText(QApplication::translate("SimCtrlPClass", "start", 0, QApplication::UnicodeUTF8));
        pauseSim->setText(QApplication::translate("SimCtrlPClass", "pause", 0, QApplication::UnicodeUTF8));
        stopSim->setText(QApplication::translate("SimCtrlPClass", "stop", 0, QApplication::UnicodeUTF8));
        exportPSH->setText(QApplication::translate("SimCtrlPClass", "Export PSH", 0, QApplication::UnicodeUTF8));
        exportSim->setText(QApplication::translate("SimCtrlPClass", "Export sim state", 0, QApplication::UnicodeUTF8));
        calcSRHist->setText(QApplication::translate("SimCtrlPClass", "Calculate SR histograms", 0, QApplication::UnicodeUTF8));
        dispPSH->setText(QApplication::translate("SimCtrlPClass", "Calculate/Display PSH", 0, QApplication::UnicodeUTF8));
        dispRaster->setText(QApplication::translate("SimCtrlPClass", "Display activity raster", 0, QApplication::UnicodeUTF8));
        dispGRGOActs->setText(QApplication::translate("SimCtrlPClass", "Display GR GO activity", 0, QApplication::UnicodeUTF8));
        dispSpikeRates->setText(QApplication::translate("SimCtrlPClass", "Display spikeRates", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class SimCtrlPClass: public Ui_SimCtrlPClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_SIMCTRLP_H
