#include "../includes/mainw.h"
#include "../includes/moc_mainw.h"

MainW::MainW(QWidget *parent, QApplication *application)
    : QMainWindow(parent)
{
	ui.setupUi(this);
	ui.dispCellType->addItem("Mossy fibers");
	ui.dispCellType->addItem("Golgi Cells");
	ui.dispCellType->addItem("Granule cells");

	this->setAttribute(Qt::WA_DeleteOnClose);

	app=application;
	connect(this, SIGNAL(destroyed()), app, SLOT(quit()));
}

MainW::~MainW()
{

}

void MainW::dispAllCells()
{
	PSHDispw *panel=new PSHDispw(NULL, 0, ui.dispCellType->currentIndex(), ui.grDispStartNum->value());
}

void MainW::dispSingleCell()
{
	PSHDispw *panel=new PSHDispw(NULL, 1, ui.dispCellType->currentIndex(), ui.dispSingleCellNum->value());
}
