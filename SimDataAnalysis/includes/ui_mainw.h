/********************************************************************************
** Form generated from reading UI file 'mainw.ui'
**
** Created: Fri Jul 8 15:07:24 2011
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
    QPushButton *singleCellNPButton;
    QPushButton *multicellNPButton;
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

    void setupUi(QMainWindow *MainWClass)
    {
        if (MainWClass->objectName().isEmpty())
            MainWClass->setObjectName(QString::fromUtf8("MainWClass"));
        MainWClass->resize(459, 224);
        centralwidget = new QWidget(MainWClass);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        verticalLayoutWidget = new QWidget(centralwidget);
        verticalLayoutWidget->setObjectName(QString::fromUtf8("verticalLayoutWidget"));
        verticalLayoutWidget->setGeometry(QRect(260, 10, 173, 197));
        verticalLayout = new QVBoxLayout(verticalLayoutWidget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        loadPSHButton = new QPushButton(verticalLayoutWidget);
        loadPSHButton->setObjectName(QString::fromUtf8("loadPSHButton"));

        verticalLayout->addWidget(loadPSHButton);

        singleCellNPButton = new QPushButton(verticalLayoutWidget);
        singleCellNPButton->setObjectName(QString::fromUtf8("singleCellNPButton"));

        verticalLayout->addWidget(singleCellNPButton);

        multicellNPButton = new QPushButton(verticalLayoutWidget);
        multicellNPButton->setObjectName(QString::fromUtf8("multicellNPButton"));

        verticalLayout->addWidget(multicellNPButton);

        quitButton = new QPushButton(verticalLayoutWidget);
        quitButton->setObjectName(QString::fromUtf8("quitButton"));

        verticalLayout->addWidget(quitButton);

        gridLayoutWidget = new QWidget(centralwidget);
        gridLayoutWidget->setObjectName(QString::fromUtf8("gridLayoutWidget"));
        gridLayoutWidget->setGeometry(QRect(20, 10, 231, 121));
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

        MainWClass->setCentralWidget(centralwidget);

        retranslateUi(MainWClass);
        QObject::connect(loadPSHButton, SIGNAL(clicked()), MainWClass, SLOT(loadPSHFile()));
        QObject::connect(singleCellNPButton, SIGNAL(clicked()), MainWClass, SLOT(dispSingleCellNP()));
        QObject::connect(multicellNPButton, SIGNAL(clicked()), MainWClass, SLOT(dispMultiCellNP()));
        QObject::connect(multiCellPageBox, SIGNAL(valueChanged(int)), MainWClass, SLOT(updateMultiCellDisp(int)));
        QObject::connect(multiCellStrideBox, SIGNAL(valueChanged(int)), MainWClass, SLOT(updateMultiCellBound(int)));
        QObject::connect(singleCellNumBox, SIGNAL(valueChanged(int)), MainWClass, SLOT(updateSingleCellDisp(int)));
        QObject::connect(dispCellTypeBox, SIGNAL(currentIndexChanged(int)), MainWClass, SLOT(updateCellType(int)));

        QMetaObject::connectSlotsByName(MainWClass);
    } // setupUi

    void retranslateUi(QMainWindow *MainWClass)
    {
        MainWClass->setWindowTitle(QApplication::translate("MainWClass", "MainWindow", 0, QApplication::UnicodeUTF8));
        loadPSHButton->setText(QApplication::translate("MainWClass", "Load PSH data", 0, QApplication::UnicodeUTF8));
        singleCellNPButton->setText(QApplication::translate("MainWClass", "new single cell PSH panel", 0, QApplication::UnicodeUTF8));
        multicellNPButton->setText(QApplication::translate("MainWClass", "new multi cell PSH panel", 0, QApplication::UnicodeUTF8));
        quitButton->setText(QApplication::translate("MainWClass", "quit", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("MainWClass", "multi disp page #", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("MainWClass", "multi disp page stride", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("MainWClass", "single disp cell #", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("MainWClass", "Display cell type", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class MainWClass: public Ui_MainWClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINW_H
