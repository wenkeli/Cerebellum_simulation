#include "../includes/mainw.hpp"

#include <QtGui/QKeyEvent>
#include <QtGui/QVBoxLayout>

using namespace std;

static const int mfActivationWidth = 2048 * .03;

MainW::MainW(QWidget *parent, int numMZ, int randSeed, string conPF, string actPF,
             Environment *env)
    : QWidget(parent), thread(this, numMZ, randSeed, conPF, actPF, env)
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

    // Initialize the views
    displayInputNetTView();
    displayStellateTView();
    displayBasketTView();
    displayPurkinjeTView();
    displayNucleusTView();
    displayOliveTView();

    // This causes the app to quit as soon as the thread finishes
    connect(&thread, SIGNAL(finished()), qApp, SLOT(quit()));

    thread.start();
}

MainW::~MainW()
{
    thread.alive = false;
    thread.wait();

    if (inputNetTView) delete inputNetTView;
    if (scTView) delete scTView;
    if (bcTView) delete bcTView;
    if (pcTView) delete pcTView;
    if (ncTView) delete ncTView;
    if (ioTView) delete ioTView;        
}

//-------------------- Methods to Create Various Display Windows --------------------//
void MainW::displayInputNetTView() {
    int trialLen = thread.trialLength;
    inputNetTView = new ActTemporalView(thread.numGO, 1, trialLen, trialLen/4, thread.numGO, Qt::white, "inputNet");
    connect(&thread, SIGNAL(updateINTW(std::vector<ct_uint8_t>, int)),
            inputNetTView, SLOT(drawRaster(std::vector<ct_uint8_t>, int)),
            Qt::QueuedConnection);
    connect(&thread, SIGNAL(blankTW(QColor)), inputNetTView, SLOT(drawBlank(QColor)), Qt::QueuedConnection);
    inputNetTView->show();
    inputNetTView->update();
}
void MainW::displayStellateTView() {
    int trialLen = thread.trialLength;
    scTView=new ActTemporalView(thread.numSC, 1, trialLen, trialLen/4, thread.numSC, Qt::white, "stellate");
    connect(&thread, SIGNAL(updateSCTW(std::vector<ct_uint8_t>, int)),
            scTView, SLOT(drawRaster(std::vector<ct_uint8_t>, int)),
            Qt::QueuedConnection);
    connect(&thread, SIGNAL(blankTW(QColor)), scTView, SLOT(drawBlank(QColor)), Qt::QueuedConnection);
    scTView->show();
    scTView->update();
}
void MainW::displayBasketTView() {
    int trialLen = thread.trialLength;
    bcTView=new ActTemporalView(thread.numBC, 1, trialLen, trialLen/4, thread.numBC, Qt::green, "basket");
    connect(&thread, SIGNAL(updateBCTW(std::vector<ct_uint8_t>, int)),
            bcTView, SLOT(drawRaster(std::vector<ct_uint8_t>, int)),
            Qt::QueuedConnection);
    connect(&thread, SIGNAL(blankTW(QColor)), bcTView, SLOT(drawBlank(QColor)), Qt::QueuedConnection);
    bcTView->show();
    bcTView->update();
}
void MainW::displayPurkinjeTView() {
    int trialLen = thread.trialLength;
    pcTView=new ActTemporalView(thread.numPC, 8, trialLen, trialLen/4, thread.numPC*8, Qt::red, "purkinje");
    connect(&thread, SIGNAL(updatePCTW(std::vector<ct_uint8_t>, std::vector<float>, int)),
            pcTView, SLOT(drawVmRaster(std::vector<ct_uint8_t>, std::vector<float>, int)),
            Qt::QueuedConnection);
    connect(&thread, SIGNAL(blankTW(QColor)), pcTView, SLOT(drawBlank(QColor)), Qt::QueuedConnection);
    pcTView->show();
    pcTView->update();
}
void MainW::displayNucleusTView() {
    int trialLen = thread.trialLength;
    ncTView=new ActTemporalView(thread.numNC, 16, trialLen, trialLen/4, thread.numNC*16, Qt::green, "nucleus");
    connect(&thread, SIGNAL(updateNCTW(std::vector<ct_uint8_t>, std::vector<float>, int)),
            ncTView, SLOT(drawVmRaster(std::vector<ct_uint8_t>, std::vector<float>, int)),
            Qt::QueuedConnection);
    connect(&thread, SIGNAL(blankTW(QColor)), ncTView, SLOT(drawBlank(QColor)), Qt::QueuedConnection);
    ncTView->show();
    ncTView->update();
}
void MainW::displayOliveTView() {
    int trialLen = thread.trialLength;
    ioTView=new ActTemporalView(thread.numIO, 32, trialLen, trialLen/4, thread.numIO*32, Qt::white, "inferior olive");
    connect(&thread, SIGNAL(updateIOTW(std::vector<ct_uint8_t>, std::vector<float>, int)),
            ioTView, SLOT(drawVmRaster(std::vector<ct_uint8_t>, std::vector<float>, int)),
            Qt::QueuedConnection);
    connect(&thread, SIGNAL(blankTW(QColor)), ioTView, SLOT(drawBlank(QColor)), Qt::QueuedConnection);
    ioTView->show();
    ioTView->update();
}

void MainW::keyPressEvent(QKeyEvent *event)
{
    int indx;
    switch (event->key()) {
    case Qt::Key_Escape:
        cout << "Escape Pressed" << endl;
        break;
    case Qt::Key_Backspace:
        thread.activateCF();
        break;
    case Qt::Key_1:
        indx = 1;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread.activateMF(i);
        break;
    case Qt::Key_2:
        indx = 2;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread.activateMF(i);
        break;
    case Qt::Key_3:
        indx = 3;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread.activateMF(i);
        break;
    case Qt::Key_4:
        indx = 4;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread.activateMF(i);
        break;
    case Qt::Key_5:
        indx = 5;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread.activateMF(i);
        break;
    case Qt::Key_6:
        indx = 6;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread.activateMF(i);
        break;
    case Qt::Key_7:
        indx = 7;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread.activateMF(i);
        break;
    case Qt::Key_8:
        indx = 8;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread.activateMF(i);
        break;
    case Qt::Key_9:
        indx = 9;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread.activateMF(i);
        break;
    case Qt::Key_0:
        indx = 0;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread.activateMF(i);
        break;

    default:
        break;
    }
}
 
void MainW::keyReleaseEvent(QKeyEvent *event)
{
    int indx;
    switch (event->key()) {
    case Qt::Key_Escape:
        qApp->quit();
        break;
    case Qt::Key_Backspace:
        break;
    case Qt::Key_1:
        indx=1;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread.deactivateMF(i);
        break;
    case Qt::Key_2:
        indx=2;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread.deactivateMF(i);
        break;
    case Qt::Key_3:
        indx=3;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread.deactivateMF(i);
        break;
    case Qt::Key_4:
        indx=4;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread.deactivateMF(i);
        break;
    case Qt::Key_5:
        indx=5;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread.deactivateMF(i);
        break;
    case Qt::Key_6:
        indx=6;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread.deactivateMF(i);
        break;
    case Qt::Key_7:
        indx=7;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread.deactivateMF(i);
        break;
    case Qt::Key_8:
        indx=8;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread.deactivateMF(i);
        break;
    case Qt::Key_9:
        indx=9;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread.deactivateMF(i);
        break;
    case Qt::Key_0:
        indx=0;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread.deactivateMF(i);
        break;
    default:
        break;
    }
}
