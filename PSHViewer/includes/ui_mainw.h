/********************************************************************************
** Form generated from reading UI file 'mainw.ui'
**
** Created: Fri Mar 18 20:22:40 2011
**      by: Qt User Interface Compiler version 4.6.3
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
#include <QtGui/QFormLayout>
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
    QWidget *formLayoutWidget;
    QFormLayout *formLayout;
    QLabel *label_3;
    QComboBox *dispCellType;
    QLabel *label;
    QSpinBox *grDispStartNum;
    QLabel *label_2;
    QSpinBox *dispSingleCellNum;
    QWidget *verticalLayoutWidget;
    QVBoxLayout *verticalLayout;
    QPushButton *dispAllCells;
    QPushButton *dispSingleCell;
    QPushButton *calcTempMetrics;
    QPushButton *loadPSH;
    QPushButton *quitButton;

    void setupUi(QMainWindow *MainWClass)
    {
        if (MainWClass->objectName().isEmpty())
            MainWClass->setObjectName(QString::fromUtf8("MainWClass"));
        MainWClass->resize(384, 189);
        centralwidget = new QWidget(MainWClass);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        formLayoutWidget = new QWidget(centralwidget);
        formLayoutWidget->setObjectName(QString::fromUtf8("formLayoutWidget"));
        formLayoutWidget->setGeometry(QRect(10, 10, 213, 91));
        formLayout = new QFormLayout(formLayoutWidget);
        formLayout->setObjectName(QString::fromUtf8("formLayout"));
        formLayout->setFieldGrowthPolicy(QFormLayout::ExpandingFieldsGrow);
        formLayout->setContentsMargins(0, 0, 0, 0);
        label_3 = new QLabel(formLayoutWidget);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        formLayout->setWidget(0, QFormLayout::LabelRole, label_3);

        dispCellType = new QComboBox(formLayoutWidget);
        dispCellType->setObjectName(QString::fromUtf8("dispCellType"));

        formLayout->setWidget(0, QFormLayout::FieldRole, dispCellType);

        label = new QLabel(formLayoutWidget);
        label->setObjectName(QString::fromUtf8("label"));

        formLayout->setWidget(1, QFormLayout::LabelRole, label);

        grDispStartNum = new QSpinBox(formLayoutWidget);
        grDispStartNum->setObjectName(QString::fromUtf8("grDispStartNum"));

        formLayout->setWidget(1, QFormLayout::FieldRole, grDispStartNum);

        label_2 = new QLabel(formLayoutWidget);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        formLayout->setWidget(2, QFormLayout::LabelRole, label_2);

        dispSingleCellNum = new QSpinBox(formLayoutWidget);
        dispSingleCellNum->setObjectName(QString::fromUtf8("dispSingleCellNum"));

        formLayout->setWidget(2, QFormLayout::FieldRole, dispSingleCellNum);

        verticalLayoutWidget = new QWidget(centralwidget);
        verticalLayoutWidget->setObjectName(QString::fromUtf8("verticalLayoutWidget"));
        verticalLayoutWidget->setGeometry(QRect(240, 10, 131, 163));
        verticalLayout = new QVBoxLayout(verticalLayoutWidget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        dispAllCells = new QPushButton(verticalLayoutWidget);
        dispAllCells->setObjectName(QString::fromUtf8("dispAllCells"));

        verticalLayout->addWidget(dispAllCells);

        dispSingleCell = new QPushButton(verticalLayoutWidget);
        dispSingleCell->setObjectName(QString::fromUtf8("dispSingleCell"));

        verticalLayout->addWidget(dispSingleCell);

        calcTempMetrics = new QPushButton(verticalLayoutWidget);
        calcTempMetrics->setObjectName(QString::fromUtf8("calcTempMetrics"));

        verticalLayout->addWidget(calcTempMetrics);

        loadPSH = new QPushButton(verticalLayoutWidget);
        loadPSH->setObjectName(QString::fromUtf8("loadPSH"));

        verticalLayout->addWidget(loadPSH);

        quitButton = new QPushButton(verticalLayoutWidget);
        quitButton->setObjectName(QString::fromUtf8("quitButton"));

        verticalLayout->addWidget(quitButton);

        MainWClass->setCentralWidget(centralwidget);

        retranslateUi(MainWClass);
        QObject::connect(dispAllCells, SIGNAL(clicked()), MainWClass, SLOT(dispAllCells()));
        QObject::connect(dispSingleCell, SIGNAL(clicked()), MainWClass, SLOT(dispSingleCell()));
        QObject::connect(loadPSH, SIGNAL(clicked()), MainWClass, SLOT(loadPSHFile()));
        QObject::connect(calcTempMetrics, SIGNAL(clicked()), MainWClass, SLOT(calcTempMetrics()));

        QMetaObject::connectSlotsByName(MainWClass);
    } // setupUi

    void retranslateUi(QMainWindow *MainWClass)
    {
        MainWClass->setWindowTitle(QApplication::translate("MainWClass", "MainWindow", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("MainWClass", "Display cell type", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("MainWClass", "Granule start page", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("MainWClass", "Display single cell #", 0, QApplication::UnicodeUTF8));
        dispAllCells->setText(QApplication::translate("MainWClass", "display all cells", 0, QApplication::UnicodeUTF8));
        dispSingleCell->setText(QApplication::translate("MainWClass", "display single cell", 0, QApplication::UnicodeUTF8));
        calcTempMetrics->setText(QApplication::translate("MainWClass", "Calc temp metrics", 0, QApplication::UnicodeUTF8));
        loadPSH->setText(QApplication::translate("MainWClass", "load PSH file", 0, QApplication::UnicodeUTF8));
        quitButton->setText(QApplication::translate("MainWClass", "quit", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class MainWClass: public Ui_MainWClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINW_H
