#include "../includes/mainw.h"

#include <QPushButton>
#include <QCheckBox>
#include <QKeyEvent>

using namespace std;

MainW::MainW(QApplication *app, QWidget *parent, int numMZ, int randSeed, string conPF, string actPF)
    : QWidget(parent), thread(this, numMZ, randSeed, conPF, actPF)
{
    resize(250, 150);
    setWindowTitle("Display Suite");
    setAttribute(Qt::WA_DeleteOnClose);

    QCheckBox *check = new QCheckBox("Test Check Box", this);
    connect(check, SIGNAL(stateChanged(int)), this, SLOT(checked(int)));

    thread.start();
}

void MainW::checked(int checked)
{
    if (checked) {
        thread.handleCheck();
        thread.running = false;
    } else {
        thread.handleCheck();
        thread.running = true;
    }
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
