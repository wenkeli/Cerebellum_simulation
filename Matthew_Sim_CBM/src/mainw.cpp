#include "../includes/mainw.h"

#include <QKeyEvent>
#include <QVBoxLayout>

using namespace std;

MainW::MainW(QApplication *app, QWidget *parent, int numMZ, int randSeed, string conPF, string actPF)
    : QWidget(parent), thread(this, numMZ, randSeed, conPF, actPF)
{
    QVBoxLayout *vbox = new QVBoxLayout(this);
    vbox->setSpacing(1);

    QPushButton *inputNetTButton = new QPushButton("InputNet Temporal View", this);
    inputNetTButton->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    connect(inputNetTButton, SIGNAL(clicked()), this, SLOT(displayInputNetTView()));

    QPushButton *stellateTButton = new QPushButton("Stellate Temporal View", this);
    stellateTButton->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    connect(stellateTButton, SIGNAL(clicked()), this, SLOT(displayStellateTView()));

    QPushButton *basketTButton = new QPushButton("Basket Temporal View", this);
    basketTButton->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    connect(basketTButton, SIGNAL(clicked()), this, SLOT(displayBasketTView()));

    QPushButton *purkinjeTButton = new QPushButton("Purkinje Temporal View", this);
    purkinjeTButton->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    connect(purkinjeTButton, SIGNAL(clicked()), this, SLOT(displayPurkinjeTView()));

    QPushButton *nucleusTButton = new QPushButton("Nucleus Temporal View", this);
    nucleusTButton->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    connect(nucleusTButton, SIGNAL(clicked()), this, SLOT(displayNucleusTView()));

    QPushButton *oliveTButton = new QPushButton("Inf Olive Temporal View", this);
    oliveTButton->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    connect(oliveTButton, SIGNAL(clicked()), this, SLOT(displayOliveTView()));

    vbox->addWidget(inputNetTButton);
    vbox->addWidget(stellateTButton);
    vbox->addWidget(basketTButton);
    vbox->addWidget(purkinjeTButton);
    vbox->addWidget(nucleusTButton);
    vbox->addWidget(oliveTButton);    
    
    setWindowTitle("Display Suite");
    setAttribute(Qt::WA_DeleteOnClose);

    thread.start();
}

MainW::~MainW()
{
    thread.alive = false;
    thread.wait();
}

void MainW::keyPressEvent(QKeyEvent *event)
{
    if(event->key() == Qt::Key_Escape)
        cout << "Pressed Escape" << endl;
}
 
void MainW::keyReleaseEvent(QKeyEvent *event)
{
    if(event->key() == Qt::Key_Escape) {
        qApp->quit();
    }
}
