/*
 * see dispdialoguep.h for more details on function descriptions and private members
 */
#include "../includes/dispdialoguep.h"
#include "../includes/moc_dispdialoguep.h"

DispDialogueP::DispDialogueP(QWidget *parent, ConnDispT t)
    : QWidget(parent)
{
	stringstream formatStr;
	ui.setupUi(this);
	dispT=t;
	this->setAttribute(Qt::WA_DeleteOnClose); //when the window is closed, delete the object
	if(dispT==MFGO || dispT==MFGR)
	{
		this->setWindowTitle("Please select the mossy fibers to display");
		ui.startNumSel->setRange(0, NUMMF-1);
		ui.endNumSel->setRange(0, NUMMF-1);
		formatStr<<NUMMF-1;
		ui.maxValLab->setText(formatStr.str().c_str());
		return;
	}
	if(dispT==GRGO || dispT==GOGR)
	{
		this->setWindowTitle("Please select the golgi cells to display");
		ui.startNumSel->setRange(0, NUMGO-1);
		ui.endNumSel->setRange(0, NUMGO-1);
		formatStr<<NUMGO-1;
		ui.maxValLab->setText(formatStr.str().c_str());
		return;
	}
}

DispDialogueP::~DispDialogueP()
{

}

void DispDialogueP::dispConns()
{
	//display connections call, depends on the display type.
	int start, end;
	ConnDispW *connDisp;

	//error check
	if(!initialized)
	{
		ui.statusBox->textCursor().insertText("Error: connections not initialized. Can not display.\n");
		ui.statusBox->repaint();
		return;
	}
	start=ui.startNumSel->value();
	end=ui.endNumSel->value();
	if(start>end || start<0)
	{
		ui.statusBox->textCursor().insertText("Error: Invaid start and end numbers, please check\n");
		ui.statusBox->repaint();
		return;
	}

	if((dispT==MFGO || dispT==MFGR) && end>=NUMMF)
	{
		ui.statusBox->textCursor().insertText("Error: wrong mossy fiber end #\n");
		ui.statusBox->repaint();
		return;
	}
	if((dispT==GRGO || dispT==GOGR) && end>=NUMGO)
	{
		ui.statusBox->textCursor().insertText("Error: wrong golgi cell end #\n");
		ui.statusBox->repaint();
		return;
	}
	//end error check

	ui.statusBox->textCursor().insertText("drawing...\n");
	ui.statusBox->repaint();
	//make a new ConnDispW window and draw the connections
	connDisp=new ConnDispW(NULL, start, end, dispT);
	connDisp->show();

	ui.statusBox->textCursor().insertText("done!\n");
	ui.statusBox->repaint();
}
