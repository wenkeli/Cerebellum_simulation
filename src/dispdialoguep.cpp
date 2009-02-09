#include "../includes/dispdialoguep.h"
#include "../includes/moc_dispdialoguep.h"
#include "../includes/conndispw.h"

DispDialogueP::DispDialogueP(QWidget *parent)
    : QWidget(parent)
{
	ui.setupUi(this);
}

DispDialogueP::~DispDialogueP()
{

}

void DispDialogueP::setDispT(ConnDispT t)
{
	stringstream formatStr;

	dispT=t;
	if(t==MFGO || t==MFGR)
	{
		this->setWindowTitle("Please select the mossy fibers to display");
		ui.startNumSel->setRange(0, NUMMF-1);
		ui.endNumSel->setRange(0, NUMMF-1);
		formatStr<<NUMMF;
		ui.maxValLab->setText(formatStr.str().c_str());
		return;
	}
	if(t==GRGO)
	{
		this->setWindowTitle("Please select the granule cells to display");
		ui.startNumSel->setRange(0, NUMGR-1);
		ui.endNumSel->setRange(0, NUMGR-1);
		formatStr<<NUMGR;
		ui.maxValLab->setText(formatStr.str().c_str());
		return;
	}
	if(t==GOGR)
	{
		this->setWindowTitle("Please select the golgi cells to display");
		ui.startNumSel->setRange(0, NUMGO-1);
		ui.endNumSel->setRange(0, NUMGO-1);
		formatStr<<NUMGO;
		ui.maxValLab->setText(formatStr.str().c_str());
		return;
	}
}

void DispDialogueP::dispConns()
{
	//display connections call, depends on the display type.
	int start, end;
	ConnDispW *connDisp;

	if(!connsMade)
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
	if(dispT==GRGO && end>=NUMGR)
	{
		ui.statusBox->textCursor().insertText("Error: wrong granule cell end #\n");
		ui.statusBox->repaint();
		return;
	}
	if(dispT==GRGO && end>=NUMGO)
	{

		ui.statusBox->textCursor().insertText("Error: wrong golgi cell end #\n");
		ui.statusBox->repaint();
		return;
	}

	connDisp=new ConnDispW();

	ui.statusBox->textCursor().insertText("drawing...\n");
	ui.statusBox->repaint();

	connDisp->setDispT(dispT);
	connDisp->setBounds(start, end);
	connDisp->show();

	ui.statusBox->textCursor().insertText("done!\n");
	ui.statusBox->repaint();
}
