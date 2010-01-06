#include "../includes/mainw.h"
#include "../includes/moc_mainw.h"

MainW::MainW(QWidget *parent, QApplication *a)
    : QMainWindow(parent)
{
	ui.setupUi(this);
	ui.dispCellType->addItem("Mossy fibers");
	ui.dispCellType->addItem("Golgi Cells");
	ui.dispCellType->addItem("Granule cells");

	ui.dispSingleCellNum->setMinimum(0);
	ui.dispSingleCellNum->setMaximum(NUMGR-1);

	ui.grDispStartNum->setMinimum(0);
	ui.grDispStartNum->setMaximum(NUMGR-ALLVIEWPH-1);

	this->setAttribute(Qt::WA_DeleteOnClose);

	app=a;
	connect(this, SIGNAL(destroyed()), app, SLOT(quit()));
	connect(ui.quitButton, SIGNAL(clicked()), app, SLOT(quit()));
}

MainW::~MainW()
{

}

void MainW::dispAllCells()
{
	PSHDispw *panel=new PSHDispw(NULL, 0, ui.dispCellType->currentIndex(), ui.grDispStartNum->value());
	panel->show();
}

void MainW::dispSingleCell()
{
	PSHDispw *panel=new PSHDispw(NULL, 1, ui.dispCellType->currentIndex(), ui.dispSingleCellNum->value());
	panel->show();
}
