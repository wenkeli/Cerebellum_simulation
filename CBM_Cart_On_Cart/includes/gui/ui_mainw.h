/********************************************************************************
** Form generated from reading UI file 'mainw.ui'
**
** Created: Mon Aug 8 12:06:53 2011
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
#include <QtGui/QHeaderView>
#include <QtGui/QMainWindow>
#include <QtGui/QPushButton>
#include <QtGui/QStatusBar>
#include <QtGui/QTextBrowser>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWClass
{
public:
    QWidget *centralwidget;
    QWidget *verticalLayoutWidget;
    QVBoxLayout *verticalLayout;
    QPushButton *connectButton;
    QPushButton *loadSimButton;
    QPushButton *viewMFGRButton;
    QPushButton *viewMFGOButton;
    QPushButton *viewGRGOButton;
    QPushButton *viewGOGRButton;
    QPushButton *runSimButton;
    QPushButton *quitButton;
    QTextBrowser *statusBox;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWClass)
    {
        if (MainWClass->objectName().isEmpty())
            MainWClass->setObjectName(QString::fromUtf8("MainWClass"));
        MainWClass->resize(574, 513);
        centralwidget = new QWidget(MainWClass);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        verticalLayoutWidget = new QWidget(centralwidget);
        verticalLayoutWidget->setObjectName(QString::fromUtf8("verticalLayoutWidget"));
        verticalLayoutWidget->setGeometry(QRect(390, 220, 183, 265));
        verticalLayout = new QVBoxLayout(verticalLayoutWidget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        connectButton = new QPushButton(verticalLayoutWidget);
        connectButton->setObjectName(QString::fromUtf8("connectButton"));

        verticalLayout->addWidget(connectButton);

        loadSimButton = new QPushButton(verticalLayoutWidget);
        loadSimButton->setObjectName(QString::fromUtf8("loadSimButton"));

        verticalLayout->addWidget(loadSimButton);

        viewMFGRButton = new QPushButton(verticalLayoutWidget);
        viewMFGRButton->setObjectName(QString::fromUtf8("viewMFGRButton"));

        verticalLayout->addWidget(viewMFGRButton);

        viewMFGOButton = new QPushButton(verticalLayoutWidget);
        viewMFGOButton->setObjectName(QString::fromUtf8("viewMFGOButton"));

        verticalLayout->addWidget(viewMFGOButton);

        viewGRGOButton = new QPushButton(verticalLayoutWidget);
        viewGRGOButton->setObjectName(QString::fromUtf8("viewGRGOButton"));

        verticalLayout->addWidget(viewGRGOButton);

        viewGOGRButton = new QPushButton(verticalLayoutWidget);
        viewGOGRButton->setObjectName(QString::fromUtf8("viewGOGRButton"));

        verticalLayout->addWidget(viewGOGRButton);

        runSimButton = new QPushButton(verticalLayoutWidget);
        runSimButton->setObjectName(QString::fromUtf8("runSimButton"));

        verticalLayout->addWidget(runSimButton);

        quitButton = new QPushButton(verticalLayoutWidget);
        quitButton->setObjectName(QString::fromUtf8("quitButton"));

        verticalLayout->addWidget(quitButton);

        statusBox = new QTextBrowser(centralwidget);
        statusBox->setObjectName(QString::fromUtf8("statusBox"));
        statusBox->setGeometry(QRect(0, 0, 391, 491));
        MainWClass->setCentralWidget(centralwidget);
        statusbar = new QStatusBar(MainWClass);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        MainWClass->setStatusBar(statusbar);

        retranslateUi(MainWClass);
        QObject::connect(connectButton, SIGNAL(clicked()), MainWClass, SLOT(makeConns()));
        QObject::connect(viewMFGRButton, SIGNAL(clicked()), MainWClass, SLOT(showMFGRMainP()));
        QObject::connect(viewMFGOButton, SIGNAL(clicked()), MainWClass, SLOT(showMFGOMainP()));
        QObject::connect(viewGRGOButton, SIGNAL(clicked()), MainWClass, SLOT(showGRGOMainP()));
        QObject::connect(viewGOGRButton, SIGNAL(clicked()), MainWClass, SLOT(showGOGRMainP()));
        QObject::connect(runSimButton, SIGNAL(clicked()), MainWClass, SLOT(runSimulation()));
        QObject::connect(loadSimButton, SIGNAL(clicked()), MainWClass, SLOT(loadSim()));

        QMetaObject::connectSlotsByName(MainWClass);
    } // setupUi

    void retranslateUi(QMainWindow *MainWClass)
    {
        MainWClass->setWindowTitle(QApplication::translate("MainWClass", "MainWindow", 0, QApplication::UnicodeUTF8));
        connectButton->setText(QApplication::translate("MainWClass", "make connections", 0, QApplication::UnicodeUTF8));
        loadSimButton->setText(QApplication::translate("MainWClass", "load sim state", 0, QApplication::UnicodeUTF8));
        viewMFGRButton->setText(QApplication::translate("MainWClass", "view MF to GR connections", 0, QApplication::UnicodeUTF8));
        viewMFGOButton->setText(QApplication::translate("MainWClass", "view MF to GO connections", 0, QApplication::UnicodeUTF8));
        viewGRGOButton->setText(QApplication::translate("MainWClass", "view GR to GO connections", 0, QApplication::UnicodeUTF8));
        viewGOGRButton->setText(QApplication::translate("MainWClass", "view GO to GR connections", 0, QApplication::UnicodeUTF8));
        runSimButton->setText(QApplication::translate("MainWClass", "Run simulation", 0, QApplication::UnicodeUTF8));
        quitButton->setText(QApplication::translate("MainWClass", "quit", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class MainWClass: public Ui_MainWClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINW_H
