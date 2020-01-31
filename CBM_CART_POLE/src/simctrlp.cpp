#include "../includes/simctrlp.h"
#include "../includes/moc_simctrlp.h"

SimCtrlP::SimCtrlP(QWidget *parent)
    : QWidget(parent)
{
	ui.setupUi(this);
	ui.stopSim->setDisabled(true);
	ui.pauseSim->setDisabled(true);
	ui.dispMode->setEditable(false);
	ui.dispMode->addItem("Mossy fibers");
	ui.dispMode->addItem("Granule cells");
	ui.dispMode->addItem("Golgi cells");
	ui.dispMode->addItem("Stellate, Basket, Purkinje cells");
	ui.dispMode->addItem("IO, nucleus, purkinje cells");

	this->setAttribute(Qt::WA_DeleteOnClose);

	paused=false;

}

SimCtrlP::~SimCtrlP()
{
	stopSim();
}

void SimCtrlP::startSim()
{
	if(!initialized)
	{
		return;
	}
	ui.startSim->setDisabled(true);
	ui.pauseSim->setDisabled(false);
	ui.stopSim->setDisabled(false);

	panel=new SimDispW();
	activityW=new ActDiagW();

	simThread=new SimThread(this, panel, activityW);
	simThread->start(QThread::TimeCriticalPriority);
//	panel->show();
}

void SimCtrlP::pauseSim()
{
	if(!paused)
	{
		paused=true;
		simPauseLock.lock();
		ui.pauseSim->setText("unpause");
	}
	else
	{
		paused=false;
		simPauseLock.unlock();
		ui.pauseSim->setText("pause");
	}
}

void SimCtrlP::stopSim()
{
	ui.stopSim->setDisabled(true);
	ui.pauseSim->setDisabled(true);
	ui.dispMode->setDisabled(true);
	simStopLock.lock();
	simStop=true;
	simStopLock.unlock();
}

void SimCtrlP::dispSpikeRates()
{
	simDispTypeLock.lock();
	if(simDispType==1)
	{
		SpikeRatesDispW *histogram;
		histogram=new SpikeRatesDispW(NULL, NUMGR, spikeSumGR, &accessSpikeSumLock, 200, "GR");
		histogram->show();
	}
	else if(simDispType==3)
	{
		SpikeRatesDispW *histogram0, *histogram1, *histogram2;
		histogram0=new SpikeRatesDispW(NULL, NUMSC, spikeSumSC, &accessSpikeSumLock, 20, "SC");
		histogram1=new SpikeRatesDispW(NULL, NUMBC, spikeSumBC, &accessSpikeSumLock, 20, "BC");
		histogram2=new SpikeRatesDispW(NULL, NUMPC, spikeSumPC, &accessSpikeSumLock, 10, "PC");
		histogram0->show();
		histogram1->show();
		histogram2->show();
	}
	else if(simDispType==4)
	{
		SpikeRatesDispW *histogram;
		histogram=new SpikeRatesDispW(NULL, NUMNC, spikeSumNC, &accessSpikeSumLock, 3, "NC");
		histogram->show();
	}
	else
	{
		SpikeRatesDispW *histogram;
		histogram=new SpikeRatesDispW(NULL, NUMGO, spikeSumGO, &accessSpikeSumLock, 25, "GO");
		histogram->show();
	}
	simDispTypeLock.unlock();
}

void SimCtrlP::exportPSH()
{
	writePSHOut();
}

void SimCtrlP::exportSim()
{
	writeSimOut();
}

void SimCtrlP::changeDispMode(int dispMode)
{
	cout<<"disp mode: "<<dispMode<<endl;
	simDispTypeLock.lock();
	simDispType=dispMode;
	simDispTypeLock.unlock();
}

void SimCtrlP::changeActMode(int actMode)
{
	cout<<"Act mode: "<<actMode<<endl;
	simDispActsLock.lock();
	simDispActs=(actMode!=Qt::Unchecked);
	if(simDispActs)
	{
		activityW->show();
	}
	else
	{
		activityW->hide();
	}
	simDispActsLock.unlock();
}

void SimCtrlP::changePSHMode(int pshMode)
{
	cout<<"PSH mode: "<<pshMode<<endl;
	simPSHCheckLock.lock();
	simPSHCheck=(pshMode!=Qt::Unchecked);
	simPSHCheckLock.unlock();
}

void SimCtrlP::changeRasterMode(int rasterMode)
{
	cout<<"Raster mode: "<<rasterMode<<endl;
	simDispRasterLock.lock();
	simDispRaster=(rasterMode!=Qt::Unchecked);
	if(simDispRaster)
	{
		panel->show();
	}
	else
	{
		panel->hide();
	}
	simDispRasterLock.unlock();
}

void SimCtrlP::changeSRHistMode(int srhMode)
{
	cout<<"SR histogram mode: "<<srhMode<<endl;
	simCalcSpikeHistLock.lock();
	simCalcSpikeHist=(srhMode!=Qt::Unchecked);
	simCalcSpikeHistLock.unlock();
}
