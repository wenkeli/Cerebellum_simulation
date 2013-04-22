#include "../includes/mainw.hpp"

#include <QtGui/QKeyEvent>

using namespace std;

static const int mfActivationWidth = 2048 * .03;

MainW::MainW(QWidget *parent, SimThread *thread, Environment *env)
    : QWidget(parent), thread(thread),
      inputNetTView(thread->numGO, 1, thread->trialLength, thread->trialLength/4,
                    thread->numGO, Qt::white, "InputNet (Mossy Fiber) Temporal View"),
      scTView(thread->numSC, 1, thread->trialLength, thread->trialLength/4,
              thread->numSC, Qt::white, "Stellate Temporal View"),
      vbox(this),
      inputNetTButton("InputNet Temporal View", this),
      stellateTButton("Stellate Temporal View", this),
      basketTButton  ("Basket Temporal View", this),
      purkinjeTButton("Purkinje Temporal View", this),
      nucleusTButton ("Nucleus Temporal View", this),
      oliveTButton   ("Inf Olive Temporal View", this)
{
    connect(thread, SIGNAL(updateINTW(std::vector<ct_uint8_t>, int)),
            &inputNetTView, SLOT(drawRaster(std::vector<ct_uint8_t>, int)),
            Qt::QueuedConnection);
    connect(thread, SIGNAL(blankTW(QColor)), &inputNetTView, SLOT(drawBlank(QColor)),
            Qt::QueuedConnection);

    connect(thread, SIGNAL(updateSCTW(std::vector<ct_uint8_t>, int)),
            &scTView, SLOT(drawRaster(std::vector<ct_uint8_t>, int)),
            Qt::QueuedConnection);
    connect(thread, SIGNAL(blankTW(QColor)), &scTView, SLOT(drawBlank(QColor)), Qt::QueuedConnection);

    vector<string> mzNames = env->getMZNames();
    // Create the basket/purkinje/nucleus/io temporal views for each MZ
    int tl = thread->trialLength;
    for (int i=0; i<thread->numMZ; i++) {
        string windowName = string(" MZ") + boost::lexical_cast<string>(i) + " " + mzNames[i] + " Basket Temporal View";
        ActTemporalView* bcTView = new ActTemporalView(thread->numBC, 1, tl, tl/4, thread->numBC, Qt::green,
                                                       windowName.c_str());
        connect(thread, SIGNAL(updateBCTW(std::vector<ct_uint8_t>, int, int)),
                this, SLOT(drawBCRaster(std::vector<ct_uint8_t>, int, int)), Qt::QueuedConnection);
        connect(thread, SIGNAL(blankTW(QColor)), bcTView, SLOT(drawBlank(QColor)), Qt::QueuedConnection);
        bcTViews.push_back(bcTView);

        windowName = string(" MZ") + boost::lexical_cast<string>(i) + " " + mzNames[i] + " Purkinje Temporal View";
        ActTemporalView* pcTView = new ActTemporalView(thread->numPC, 8, tl, tl/4, thread->numPC*8,
                                                       Qt::red, windowName.c_str());
        connect(thread, SIGNAL(updatePCTW(std::vector<ct_uint8_t>, std::vector<float>, int, int)),
                this, SLOT(drawPCVmRaster(std::vector<ct_uint8_t>, std::vector<float>, int, int)),
                Qt::QueuedConnection);
        connect(thread, SIGNAL(blankTW(QColor)), pcTView, SLOT(drawBlank(QColor)), Qt::QueuedConnection);
        pcTViews.push_back(pcTView);

        windowName = string(" MZ") + boost::lexical_cast<string>(i) + " " + mzNames[i] + " Nucleus Temporal View";
        ActTemporalView* ncTView = new ActTemporalView(thread->numNC, 16, tl, tl/4, thread->numNC*16,
                                                      Qt::green, windowName.c_str());
        connect(thread, SIGNAL(updateNCTW(std::vector<ct_uint8_t>, std::vector<float>, int, int)),
                this, SLOT(drawNCVmRaster(std::vector<ct_uint8_t>, std::vector<float>, int, int)),
                Qt::QueuedConnection);
        connect(thread, SIGNAL(blankTW(QColor)), ncTView, SLOT(drawBlank(QColor)), Qt::QueuedConnection);
        ncTViews.push_back(ncTView);

        windowName = string(" MZ") + boost::lexical_cast<string>(i) + " " + mzNames[i] + " Inferior Olive Temporal View";
        ActTemporalView* ioTView = new ActTemporalView(thread->numIO, 32, tl, tl/4, thread->numIO*32, Qt::white,
                                                       windowName.c_str());
        connect(thread, SIGNAL(updateIOTW(std::vector<ct_uint8_t>, std::vector<float>, int, int)),
                this, SLOT(drawIOVmRaster(std::vector<ct_uint8_t>, std::vector<float>, int, int)),
                Qt::QueuedConnection);
        connect(thread, SIGNAL(blankTW(QColor)), ioTView, SLOT(drawBlank(QColor)), Qt::QueuedConnection);
        ioTViews.push_back(ioTView);
    }

    // Connect the buttons to the toggle visible methods
    connect(&inputNetTButton, SIGNAL(clicked()), &inputNetTView, SLOT(toggleVisible()));
    connect(&stellateTButton, SIGNAL(clicked()), &scTView, SLOT(toggleVisible()));
    for (int i=0; i<thread->numMZ; i++) {
        connect(&basketTButton, SIGNAL(clicked()), bcTViews[i], SLOT(toggleVisible()));
        connect(&purkinjeTButton, SIGNAL(clicked()), pcTViews[i], SLOT(toggleVisible()));
        connect(&nucleusTButton, SIGNAL(clicked()), ncTViews[i], SLOT(toggleVisible()));
        connect(&oliveTButton, SIGNAL(clicked()), ioTViews[i], SLOT(toggleVisible()));
    }

    // Make the buttons expanable
    inputNetTButton.setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    stellateTButton.setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    basketTButton.setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    purkinjeTButton.setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    nucleusTButton.setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    oliveTButton.setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    // Add the buttons to vbox
    vbox.setSpacing(1);
    vbox.addWidget(&inputNetTButton);
    vbox.addWidget(&stellateTButton);
    vbox.addWidget(&basketTButton);
    vbox.addWidget(&purkinjeTButton);
    vbox.addWidget(&nucleusTButton);
    vbox.addWidget(&oliveTButton);    
    
    setWindowTitle("Display Suite");
    setAttribute(Qt::WA_DeleteOnClose);

    // This causes the app to quit as soon as the thread finishes
    connect(thread, SIGNAL(finished()), qApp, SLOT(quit()));

    thread->start();
}

MainW::~MainW()
{
    thread->alive = false;
    thread->wait();

    for (uint i=0; i<bcTViews.size(); i++)
        delete bcTViews[i];
    for (uint i=0; i<pcTViews.size(); i++)
        delete pcTViews[i];
    for (uint i=0; i<ncTViews.size(); i++)
        delete ncTViews[i];
    for (uint i=0; i<ioTViews.size(); i++)
        delete ioTViews[i];
}

void MainW::drawBCRaster(vector<ct_uint8_t> aps, int t, int mz) {
    assert(uint(mz) < bcTViews.size());
    bcTViews[mz]->drawRaster(aps, t);
}
void MainW::drawPCVmRaster(vector<ct_uint8_t> aps, vector<float> vm, int t, int mz) {
    assert(uint(mz) < pcTViews.size());
    pcTViews[mz]->drawVmRaster(aps, vm, t);
}
void MainW::drawNCVmRaster(vector<ct_uint8_t> aps, vector<float> vm, int t, int mz) {
    assert(uint(mz) < ncTViews.size());
    ncTViews[mz]->drawVmRaster(aps, vm, t);
}
void MainW::drawIOVmRaster(vector<ct_uint8_t> aps, vector<float> vm, int t, int mz) {
    assert(uint(mz) < ioTViews.size());
    ioTViews[mz]->drawVmRaster(aps, vm, t);
}

void MainW::keyPressEvent(QKeyEvent *event)
{
    int indx;
    switch (event->key()) {
    case Qt::Key_Escape:
        cout << "Escape Pressed" << endl;
        break;
    case Qt::Key_Backspace:
        thread->activateCF();
        break;
    case Qt::Key_1:
        indx = 1;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread->activateMF(i);
        break;
    case Qt::Key_2:
        indx = 2;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread->activateMF(i);
        break;
    case Qt::Key_3:
        indx = 3;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread->activateMF(i);
        break;
    case Qt::Key_4:
        indx = 4;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread->activateMF(i);
        break;
    case Qt::Key_5:
        indx = 5;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread->activateMF(i);
        break;
    case Qt::Key_6:
        indx = 6;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread->activateMF(i);
        break;
    case Qt::Key_7:
        indx = 7;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread->activateMF(i);
        break;
    case Qt::Key_8:
        indx = 8;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread->activateMF(i);
        break;
    case Qt::Key_9:
        indx = 9;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread->activateMF(i);
        break;
    case Qt::Key_0:
        indx = 0;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread->activateMF(i);
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
    case Qt::Key_S:
        cout << "Saving simulator" << endl;
        thread->saveSimState();
        break;
    case Qt::Key_P:
        thread->paused = !thread->paused;
        cout << (thread->paused ? "Paused" : "Unpaused") << endl;
        break;
    case Qt::Key_1:
        indx=1;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread->deactivateMF(i);
        break;
    case Qt::Key_2:
        indx=2;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread->deactivateMF(i);
        break;
    case Qt::Key_3:
        indx=3;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread->deactivateMF(i);
        break;
    case Qt::Key_4:
        indx=4;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread->deactivateMF(i);
        break;
    case Qt::Key_5:
        indx=5;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread->deactivateMF(i);
        break;
    case Qt::Key_6:
        indx=6;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread->deactivateMF(i);
        break;
    case Qt::Key_7:
        indx=7;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread->deactivateMF(i);
        break;
    case Qt::Key_8:
        indx=8;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread->deactivateMF(i);
        break;
    case Qt::Key_9:
        indx=9;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread->deactivateMF(i);
        break;
    case Qt::Key_0:
        indx=0;
        for (int i=indx*mfActivationWidth; i<(indx+1)*mfActivationWidth; i++)
            thread->deactivateMF(i);
        break;
    default:
        break;
    }
}
