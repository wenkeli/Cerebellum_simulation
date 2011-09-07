#include "../includes/mainw.h"
#include "../includes/moc_mainw.h"

MainW::MainW(QWidget *parent, QApplication *a)
    : QMainWindow(parent)
{
	ui.setupUi(this);
	ui.dispCellTypeBox->setEditable(false);

	pshLoaded=false;
	simLoaded=false;

	cellTypes[0]="MF";
	cellTypes[1]="GO";
	cellTypes[2]="GR";
	cellTypes[3]="SC";
	cellTypes[4]="BC";
	cellTypes[5]="PC";
	cellTypes[6]="IO";
	cellTypes[7]="NC";
//	cout<<"here1"<<endl;

	pshs[0]=&mfPSH;
	pshs[1]=&goPSH;
	pshs[2]=&grPSH;
	pshs[3]=&scPSH;
	pshs[4]=&bcPSH[0];
	pshs[5]=&pcPSH[0];
	pshs[6]=&ioPSH[0];
	pshs[7]=&ncPSH[0];

	srAnalysis[0]=&mfSR;
	srAnalysis[1]=&goSR;
	srAnalysis[2]=&grSR;
	srAnalysis[3]=&scSR;
	srAnalysis[4]=&bcSR[0];
	srAnalysis[5]=&pcSR[0];
	srAnalysis[6]=&ioSR[0];
	srAnalysis[7]=&ncSR[0];

//	cout<<"here2"<<endl;

	curSingleWindow=NULL;
	curMultiWindow=NULL;

//	cout<<"here3"<<endl;

	for(int i=0; i<8; i++)
	{
		ui.dispCellTypeBox->addItem(cellTypes[i]);
	}

//	cout<<"here4";

	this->setAttribute(Qt::WA_DeleteOnClose);

	app=a;
//	cout<<"here5"<<endl;

	calcTempMetricBinN=0;
	connect(this, SIGNAL(destroyed()), app, SLOT(quit()));
	connect(ui.quitButton, SIGNAL(clicked()), app, SLOT(quit()));

//	cout<<"here6"<<endl;

	ui.dispCellTypeBox->setDisabled(true);

	ui.dispCellTypeBox->setEditable(false);

	ui.multiCellPageBox->setDisabled(true);
	ui.multiCellStrideBox->setDisabled(true);
	ui.singleCellNumBox->setDisabled(true);
	ui.singleCellNPButton->setDisabled(true);
	ui.multicellNPButton->setDisabled(true);

	ui.pfPCPlastUSTimeSpinBox->setDisabled(true);
	ui.calcPFPCPlastButton->setDisabled(true);
	ui.exportPFPCPlastActButton->setDisabled(true);

	ui.calcSpikeRatesButton->setDisabled(true);
	ui.exportSpikeRatesButton->setDisabled(true);
//	cout<<"here7"<<endl;
}

MainW::~MainW()
{
	delete mfPSH;
	delete goPSH;
	delete grPSH;
	delete scPSH;

	for(int i=0; i<NUMMZONES; i++)
	{
		delete bcPSH[i];
		delete pcPSH[i];
		delete ioPSH[i];
		delete ncPSH[i];
	}
	delete grPopTimingAnalysis;
}

void MainW::dispMultiCellNP()
{
	int startN, endN;

	startN=ui.multiCellPageBox->value()*ui.multiCellStrideBox->value();
	endN=startN+ui.multiCellStrideBox->value();

	if(curMultiWindow!=NULL)
	{
		curMultiWindow->setAttribute(Qt::WA_DeleteOnClose);
	}

	curMultiWindow=new PSHDispw(NULL,
			(*curPSH)->paintPSHPop(startN, endN),
			cellTypes[ui.dispCellTypeBox->currentIndex()]);
}

void MainW::dispSingleCellNP()
{
	if(curSingleWindow!=NULL)
	{
		curSingleWindow->setAttribute(Qt::WA_DeleteOnClose);
	}
	curSingleWindow=new PSHDispw(NULL,
			(*curPSH)->paintPSHInd(ui.singleCellNumBox->value()),
			cellTypes[ui.dispCellTypeBox->currentIndex()]);
}

void MainW::updateSingleCellDisp(int cellN)
{
	if(curSingleWindow==NULL)
	{
		return;
	}

	curSingleWindow->switchBuf((*curPSH)->paintPSHInd(cellN));
}

void MainW::updateMultiCellDisp(int page)
{
	int startN, endN;
	if(curMultiWindow==NULL)
	{
		return;
	}

	startN=page*ui.multiCellStrideBox->value();
	endN=startN+ui.multiCellStrideBox->value();

	curMultiWindow->switchBuf((*curPSH)->paintPSHPop(startN, endN));
}

void MainW::updateMultiCellBound(int stride)
{
	int numCells;
	numCells=(*curPSH)->getCellNum();

	ui.multiCellPageBox->setMaximum(numCells/stride);
}

void MainW::updateCellType(int type)
{
	int strideMax;
	curPSH=pshs[type];
	curSRAnalysis=srAnalysis[type];
	if((*curPSH)==NULL)
	{
		return;
	}
	strideMax=(*curPSH)->getCellNum();
	if(strideMax>1024)
	{
		strideMax=1024;
	}
	ui.multiCellPageBox->setMinimum(0);
	ui.multiCellStrideBox->setMinimum(4);
	ui.multiCellStrideBox->setMaximum(strideMax);
	updateMultiCellBound(ui.multiCellStrideBox->value());

	ui.singleCellNumBox->setMinimum(0);
	ui.singleCellNumBox->setMaximum((int)((*curPSH)->getCellNum())-1);
}

void MainW::loadPSHFile()
{
	ifstream infile;
	QString fileName;

	fileName=QFileDialog::getOpenFileName(this, "Please select the PSH file to open", "/", "");


	cout<<"PSH file name: "<<fileName.toStdString()<<endl;

	infile.open(fileName.toStdString().c_str(), ios::binary);
	if(!infile.good() || !infile.is_open())
	{
		cerr<<"error opening file "<<fileName.toStdString()<<endl;
		return;
	}

	delete mfPSH;
	delete goPSH;
	delete grPSH;
	delete scPSH;

	for(int i=0; i<NUMMZONES; i++)
	{
		delete bcPSH[i];
		delete pcPSH[i];
		delete ioPSH[i];
		delete ncPSH[i];
	}
	delete grPopTimingAnalysis;

	delete mfSR;
	delete goSR;
	delete grSR;
	delete scSR;
	for(int i=0; i<NUMMZONES; i++)
	{
		delete bcSR[i];
		delete pcSR[i];
		delete ioSR[i];
		delete ncSR[i];
	}

	mfPSH=new PSHData(infile);
	goPSH=new PSHData(infile);
	grPSH=new PSHDataGPU(infile);
	scPSH=new PSHData(infile);
	for(int i=0; i<NUMMZONES; i++)
	{
		bcPSH[i]=new PSHData(infile);
		pcPSH[i]=new PSHData(infile);
		ioPSH[i]=new PSHData(infile);
		ncPSH[i]=new PSHData(infile);
	}

	grPopTimingAnalysis=new GRPSHPopAnalysis(grPSH);

	mfSR=new SpikeRateAnalysis(mfPSH);
	goSR=new SpikeRateAnalysis(goPSH);
	grSR=new SpikeRateAnalysis(grPSH);
	scSR=new SpikeRateAnalysis(scPSH);
	for(int i=0; i<NUMMZONES; i++)
	{
		bcSR[i]=new SpikeRateAnalysis(bcPSH[i]);
		pcSR[i]=new SpikeRateAnalysis(pcPSH[i]);
		ioSR[i]=new SpikeRateAnalysis(ioPSH[i]);
		ncSR[i]=new SpikeRateAnalysis(ncPSH[i]);
	}

	cout<<"done!"<<endl;
	infile.close();

	pshLoaded=true;

	if(simLoaded)
	{
		delete grConAnalysis;
		grConAnalysis=new GRConPSHAnalysis(goPSH, mfPSH, simInNetMod);
		//enable button
	}

	ui.dispCellTypeBox->setEnabled(true);
	updateCellType(ui.dispCellTypeBox->currentIndex());

	ui.multiCellPageBox->setEnabled(true);
	ui.multiCellStrideBox->setEnabled(true);

	ui.singleCellNumBox->setEnabled(true);
	ui.singleCellNPButton->setEnabled(true);
	ui.multicellNPButton->setEnabled(true);

	ui.pfPCPlastUSTimeSpinBox->setEnabled(true);
	ui.pfPCPlastUSTimeSpinBox->setMinimum(0);
	ui.pfPCPlastUSTimeSpinBox->setMaximum(grPSH->getStimNumBins()*grPSH->getBinTimeSize());
	ui.calcPFPCPlastButton->setEnabled(true);
	ui.exportPFPCPlastActButton->setEnabled(true);

	ui.calcSpikeRatesButton->setEnabled(true);
	ui.exportSpikeRatesButton->setEnabled(true);
}

void MainW::calcPFPCPlasticity()
{
	grPopTimingAnalysis->calcPFPCPlast(ui.pfPCPlastUSTimeSpinBox->value());
}

void MainW::exportPFPCPlastAct()
{
	ofstream outfile;
	QString fileName;

	fileName=QFileDialog::getSaveFileName(this, "Please select where to save PFPC activity file", "/", "");

	cout<<"PSH file name: "<<fileName.toStdString()<<endl;

	outfile.open(fileName.toStdString().c_str());
	if(!outfile.good() || !outfile.is_open())
	{
		cerr<<"error opening file "<<fileName.toStdString()<<endl;
		return;
	}

	grPopTimingAnalysis->exportPFPCPlastAct(outfile);

	cout<<"done"<<endl;
	outfile.close();
}

void MainW::showGRInMFGOPSHs()
{

}

void MainW::showGROutGOPSHs()
{

}

void MainW::calcSpikeRates()
{
	(*curSRAnalysis)->calcSpikeRates();
}

void MainW::exportSpikeRates()
{
	ofstream outfile;
	QString fileName;

	fileName=QFileDialog::getSaveFileName(this, "Please select where to save spike rate file", "/", "");

	cout<<"SR file name: "<<fileName.toStdString()<<endl;

	outfile.open(fileName.toStdString().c_str());
	if(!outfile.good() || !outfile.is_open())
	{
		cerr<<"error opening file "<<fileName.toStdString()<<endl;
		return;
	}

	(*curSRAnalysis)->exportSpikeRates(outfile);

	cout<<"done"<<endl;
	outfile.close();
}

void MainW::loadSimFile()
{
	ifstream infile;
	QString fileName;

	fileName=QFileDialog::getOpenFileName(this, "Please select the sim state file to open", "/", "");


	cout<<"sim state file name: "<<fileName.toStdString()<<endl;

	infile.open(fileName.toStdString().c_str(), ios::binary);
	if(!infile.good() || !infile.is_open())
	{
		cerr<<"error opening file "<<fileName.toStdString()<<endl;
		return;
	}

	for(int i=0; i<NUMMZONES; i++)
	{
		delete simErrMod[i];
		simErrMod[i]=new SimErrorEC(infile);

		delete simOutMod[i];
		simOutMod[i]=new SimOutputEC(infile);
	}

	delete simExternalMod;
	simExternalMod=new SimExternalEC(infile);

	delete simMFInputMod;
	simMFInputMod=new SimMFInputEC(infile);

	delete simInNetMod;
	simInNetMod=new SimInNet(infile);

	for(int i=0; i<NUMMZONES; i++)
	{
		delete simMZoneMod[i];
		simMZoneMod[i]=new SimMZone(infile);
	}

	cout<<"done!"<<endl;
	infile.close();

	simLoaded=true;
	if(pshLoaded)
	{
		delete grConAnalysis;
		grConAnalysis=new GRConPSHAnalysis(goPSH, mfPSH, simInNetMod);
		//enable button
	}
}

void MainW::exportSim()
{
//	ofstream outf;
//
//	QString fileName;
//
//	fileName=QFileDialog::getOpenFileName(this, "Please select where you want to save the sim", "/", "");
//
//
//	cout<<"Sim file name: "<<fileName.toStdString()<<endl;
//
//	outf.open(fileName.toStdString().c_str(), ios::binary);
//	if(!outf.good() || !outf.is_open())
//	{
//		cerr<<"error opening file "<<fileName.toStdString()<<endl;
//		return;
//	}
//
//	cout<<"writing state data"<<endl;
//	outf.seekp(0, ios_base::beg);

//
//	outf.flush();
//	cout<<"done"<<endl;
//
//	outf.close();
}

void MainW::exportSinglePSH()
{
//	ofstream outf;
//
//	QString fileName;
//
//	unsigned int cellN;
//	unsigned int cellT;
//
//	fileName=QFileDialog::getOpenFileName(this, "Please select where you want to save the single cell PSH", "/", "");
//
//
//	cout<<"Single PSH file name: "<<fileName.toStdString()<<endl;
//
//	outf.open(fileName.toStdString().c_str(), ios::out);
//	if(!outf.good() || !outf.is_open())
//	{
//		cerr<<"error opening file "<<fileName.toStdString()<<endl;
//		return;
//	}
//
//	cout<<"writing single PSH... ";
//	cout.flush();
//
//	cellN=ui.dispSingleCellNum->value();
//	cellT=ui.dispCellType->currentIndex();
//
//	outf.close();
//	cout<<"done!"<<endl;
}

//void MainW::calcTempMetrics()
//{
//	ofstream outfile;
//	QString fileName;
//
//	fileName=QFileDialog::getOpenFileName(this, "Please specify where to save the data", "/", "");
//	outfile.open(fileName.toStdString().c_str(), ios::out);
//	if(!outfile.good() || !outfile.is_open())
//	{
//		cerr<<"error opening file "<<fileName.toStdString()<<endl;
//		return;
//	}
//
//	cout<<"Calculating total spikes"<<endl;
//	calcGRTotalSpikes();
//	calcGRTotalSpikesPC();
////	cout<<"calculating individual temporal specificity"<<endl;
////	calcGRTempSpecific();
////	cout<<"calculating population metrics"<<endl;
////	calcGRPopTempMetric();
//	cout<<"calculating population plasticity metrics, all GR"<<endl;
////	calcGRPlastTempMetric(outfile);
//
//	cout<<"calculating population plastiticy metrics, per PC"<<endl;
//	calcGRPlastTempMetricPC(outfile);
//
//	cout<<"writing results"<<endl;
//
////	outfile<<"specGRSpM activeGRSpM totalGRSpM "<<
////			"specGRActM activeGRActM totalGRActM "<<
////			"spTotGRActR spActGRActR actTotGRActR "<<
////			"SpecLTD AmpLTD"<<endl;
////	for(int i=0; i<NUMBINS; i++)
////	{
////		outfile<<specGRPopSpMean[i]<<" "<<activeGRPopSpMean[i]<<" "<<totalGRPopSpMean[i]<<
////				" "<<specGRPopActMean[i]<<" "<<activeGRPopActMean[i]<<" "<<totalGRPopActMean[i]<<
////				" "<<spTotGRPopActR[i]<<" "<<spActGRPopActR[i]<<" "<<actTotGRPopActR[i]<<
////				" "<<grPopActSpecPlast[i]<<" "<<grPopActAmpPlast[i]<<endl;
////	}
////
////	outfile<<endl<<endl;
//	for(int i=0; i<NUMBINS; i++)
//	{
//		outfile<<grBinTotalSpikes[i]<<" ";
//	}
//	outfile<<endl;
//	for(int i=calcTempMetricBinN; i<=calcTempMetricBinN; i++)
//	{
//		for(int j=0; j<NUMBINS; j++)
//		{
//			outfile<<grPopActDiffPlast[i][j]<<" ";
//		}
//		outfile<<endl;
//	}
//	outfile<<endl<<endl;
//
//
//	for(int i=0; i<NUMPC; i++)
//	{
//		for(int j=0; j<NUMBINS; j++)
//		{
//			outfile<<grBinTotalSpikesPC[i][j]<<" ";
//		}
//		outfile<<endl;
//
//		for(int j=calcTempMetricBinN; j<=calcTempMetricBinN; j++)
//		{
//			for(int k=0; k<NUMBINS; k++)
//			{
//				outfile<<grPopActDiffPlastPC[j][i][k]<<" ";
//			}
//			outfile<<endl;
//		}
//		outfile<<endl;
//	}
//
//	outfile.close();
//
//	for(int i=0; i<NUMGR; i++)
//	{
//		pfSynWeightPC[i]=grWeightsPlastPC[calcTempMetricBinN][i]/2;
//		//grWeightsPlast
//	}
//	cout<<"done!"<<endl;
//}

//void MainW::changeTempMetricBinN(int bN)
//{
//	if(bN>=grPSH->getTotalNumBins())
//	{
//		calcTempMetricBinN=grPSH->getTotalNumBins()-1;
//	}
//	else if(bN<0)
//	{
//		calcTempMetricBinN=0;
//	}
//	else
//	{
//		calcTempMetricBinN=bN;
//	}
//}


//void MainW::calcGRTotalSpikes()
//{
//	if(grTotalCalced)
//	{
//		return;
//	}
//
//	for(int i=0; i<NUMGR; i++)
//	{
//		grTotalSpikes[i]=0;
//	}
//
//	for(int i=0; i<NUMBINS; i++)
//	{
//		grBinTotalSpikes[i]=0;
//
//		for(int j=0; j<NUMGR; j++)
//		{
//			grTotalSpikes[j]=grTotalSpikes[j]+pshGR[i][j];
//
//			grBinTotalSpikes[i]=grBinTotalSpikes[i]+pshGR[i][j];
//		}
//		grBinTotalSpikes[i]=grBinTotalSpikes[i]/((float)numTrials);
//	}
//
//	for(int i=0; i<NUMGR; i++)
//	{
//		grTotalSpikes[i]=grTotalSpikes[i]/((float)numTrials);
//	}
//
//	grTotalCalced=true;
//}

//void MainW::calcGRTotalSpikesPC()
//{
//	for(int i=0; i<NUMPC; i++)
//	{
//		for(int j=0; j<NUMBINS; j++)
//		{
//			grBinTotalSpikesPC[i][j]=0;
//			for(int k=i*(NUMGR/NUMPC); k<(i+1)*(NUMGR/NUMPC); k++)
//			{
//				grBinTotalSpikesPC[i][j]=grBinTotalSpikesPC[i][j]+pshGR[j][k];
//			}
//			grBinTotalSpikesPC[i][j]=grBinTotalSpikesPC[i][j]/((float)numTrials);
//		}
//	}
//}

//void MainW::calcGRTempSpecific()
//{
//	for(int i=0; i<NUMGR; i++)
//	{
//		short peakBin;
//		float peakVal;
//
//		peakBin=-1;
//		peakVal=0;
//		for(int j=0; j<NUMBINS; j++)
//		{
//			int tempSum;
//			tempSum=0;
//			for(int k=j-TEMPMETSLIDINGW+1; k<=j; k++)
//			{
//				if(k<0)
//				{
//					continue;
//				}
//				tempSum=tempSum+pshGRTrans[i][k];
////				cout<<"k="<<k<<" tempSum="<<tempSum<<endl;
//			}
//			if(grTotalSpikes[i]>0)
//			{
//				grTempSpecificity[i][j]=((float)tempSum)/((float)grTotalSpikes[i]);
//			}
//			else
//			{
////				cout<<grTotalSpikes[i]<<" "<<tempSum<<" "<<endl;
//				grTempSpecificity[i][j]=0;
//			}
////			cout<<grTempSpecificity[i][j]<<endl;
//			if(grTempSpecificity[i][j]>peakVal)
//			{
//				peakVal=grTempSpecificity[i][j];
//				peakBin=j;
//			}
//		}
//
//		grTempSpPeakBin[i]=peakBin;
//		grTempSpPeakVal[i]=peakVal;
//	}
//}

//void MainW::calcGRPopTempMetric()
//{
//	for(int i=0; i<NUMBINS; i++)
//	{
////		vector<int> grSpecInd;
//		bool grIsSpecific[NUMGR];
//		bool grIsActive[NUMGR];
//		int numSpecGR;
//		int numActiveGR;
//		float specGRSpSum;
//		float activeGRSpSum;
//		float totalGRSpSum;
//		unsigned int specGRActSum;
//		unsigned int activeGRActSum;
//		unsigned int totalGRActSum;
////		int grSpSumTemp;
//
//		memset((char *)grIsSpecific, 0, NUMGR*sizeof(bool));
//		memset((char *)grIsActive, 0, NUMGR*sizeof(bool));
////		grSpecInd.clear();
//
//		numSpecGR=0;
//		numActiveGR=0;
//		for(int j=i; j<=i; j++)//i-TEMPMETSLIDINGW+1
//		{
//			if(j<0)
//			{
//				continue;
//			}
//			for(int k=0; k<NUMGR; k++)
//			{
//				if(grTempSpPeakVal[k]*grTotalSpikes[k]>=numTrials*2)
//				{
//					grIsActive[k]=true;
//					numActiveGR++;
//
//					if(grTempSpPeakBin[k]==j)
//					{
//						grIsSpecific[k]=true;
//						numSpecGR++;
//					}
//				}
//			}
//		}
//
//		specGRSpSum=0;
//		activeGRSpSum=0;
//		totalGRSpSum=0;
//		specGRActSum=0;
//		activeGRActSum=0;
//		totalGRActSum=0;
////		grSpSumTemp=0;
////		cout<<grSpSum<<" "<<specGRSpSum<<" "<<specAvg<<endl;
//		for(int j=0; j<NUMGR; j++)
//		{
//			if(grIsSpecific[j])
//			{
//				specGRActSum=specGRActSum+grTempSpecificity[j][i]*grTotalSpikes[j];
////				cout<<endl<<grTempSpecificity[j][i]<<" "<<grTotalSpikes[j]<<" "<<specGRSpSum<<" "<<grSpSum<<" "<<grSpSumTemp<<endl;
//				specGRSpSum=specGRSpSum+grTempSpPeakVal[j];
//			}
//
//			if(grIsActive[j])
//			{
//				activeGRActSum=activeGRActSum+grTempSpecificity[j][i]*grTotalSpikes[j];
//				activeGRSpSum=activeGRSpSum+grTempSpecificity[j][i];
//			}
////			if(grTempSpecificity[j][i]*grTotalSpikes[j]<0 || grTempSpecificity[j][i]*grTotalSpikes[j]>100000)
////			{
////				cout<<grSpSum<<" ";
////				cout<<grTempSpecificity[j][i]*grTotalSpikes[j]<<" "<<grTempSpecificity[j][i]<<" "<<grTotalSpikes[j]<<"|";
////			}
//
////			grSpSumTemp=grSpSum;
//			totalGRSpSum=totalGRSpSum+grTempSpecificity[j][i];
//			totalGRActSum=totalGRActSum+grTempSpecificity[j][i]*grTotalSpikes[j];
////			if(grSpSum<0 && grSpSumTemp>=0)
////			{
////				cout<<grSpSumTemp<<" "<<grSpSum<<" "<<grTempSpecificity[j][i]<<" "<<grTotalSpikes[j]<<" "<<grTempSpecificity[j][i]*grTotalSpikes[j]<<endl;
////			}
//		}
//
//		numGRActive[i]=numActiveGR;
//		numGRSpecific[i]=numSpecGR;
//
//		if(specGRSpSum>0)
//		{
//			specGRPopSpMean[i]=specGRSpSum/((float)numSpecGR);
//			specGRPopActMean[i]=specGRActSum/((float)numSpecGR);
//		}
//		if(activeGRSpSum>0)
//		{
//			activeGRPopSpMean[i]=activeGRSpSum/((float)numActiveGR);
//			activeGRPopActMean[i]=activeGRActSum/((float)numActiveGR);
//
//			spActGRPopActR[i]=((float)specGRActSum)/((float)activeGRActSum);
//		}
//		if(totalGRSpSum>0)
//		{
//			totalGRPopSpMean[i]=totalGRSpSum/((float)NUMGR);
//			totalGRPopActMean[i]=totalGRActSum/((float)NUMGR);
//
//			spTotGRPopActR[i]=((float)specGRActSum)/((float)totalGRActSum);
//			actTotGRPopActR[i]=((float)activeGRActSum)/((float)totalGRActSum);
//		}
//	}
//}

//void MainW::calcGRPlastTempMetric(ofstream &outfile)
//{
//	initGRPlastTempVars();
//
////	for(int i=0; i<NUMBINS; i+=10)
//	for(int i=calcTempMetricBinN; i<=calcTempMetricBinN; i++)
//	{
//		double maxLTDBinDiff;
//
//		double lastLTDBinDiff;
//		double lastLTPBinDiff;
//
////		int startT;
////		startT=time(NULL);
//		cout<<i<<endl;
//
//		maxLTDBinDiff=0;
//
//		lastLTDBinDiff=0;
//		lastLTPBinDiff=0;
//
////		outfile<<i<<endl;
////		cout<<"initializing LTD"<<endl;
////		for(int j=0; j<300; j++)
////		{
////			cout<<j<<" ";
////			calcGRLTDSynWeight(i, 1);
////			calcGRPlastPopAct(i);
////			calcGRPlastPopActDiff(i);
////			for(int k=0; k<NUMBINS; k++)
////			{
////				outfile<<grPopActDiffPlast[i][k]<<" ";
////			}
////			outfile<<endl;
////		}
//		calcGRLTDSynWeight(i, 1);
////		cout<<endl;
//		calcGRPlastPopAct(i);
//		maxLTDBinDiff=calcGRPlastPopActDiff(i);
//		lastLTDBinDiff=maxLTDBinDiff;
//
////		for(int j=0; j<NUMBINS; j++)
////		{
////			outfile<<grPopActDiffPlast[i][j]<<" ";
////		}
////		outfile<<endl;
////		cout<<"first LTD initialized, max val: "<<maxLTDBinDiff<<endl;
//
//		for(int j=0; j<200; j++)//j>=0; j++)//<500; j++)
//		{
//			double curLTDBinDiff;
//			double curLTPBinDiff;
//
//			calcGRLTPSynWeight(i, maxLTDBinDiff);
//
////			cout<<j<<endl;
//			calcGRPlastPopAct(i);
//			curLTPBinDiff=calcGRPlastPopActDiff(i);
////			for(int k=0; k<NUMBINS; k++)
////			{
////				outfile<<grPopActDiffPlast[i][k]<<" ";
////			}
////			outfile<<endl;
//
//			calcGRLTDSynWeight(i, (curLTPBinDiff<maxLTDBinDiff)*(1-(curLTPBinDiff/maxLTDBinDiff)));
//
//			calcGRPlastPopAct(i);
//			curLTDBinDiff=calcGRPlastPopActDiff(i);
////			for(int k=0; k<NUMBINS; k++)
////			{
////				outfile<<grPopActDiffPlast[i][k]<<" ";
////			}
////			outfile<<endl;
//
////			if(fabs((lastLTDBinDiff-curLTDBinDiff)/maxLTDBinDiff)<0.0001)
////			{
////				cout<<j;
////				break;
////			}
//			lastLTPBinDiff=curLTPBinDiff;
//			lastLTDBinDiff=curLTDBinDiff;
//		}
////		cout<<"time for bin: "<<time(NULL)-startT<<endl;
//
////		for(int j=0; j<NUMGR; j++)
////		{
////			if(grWeightsPlast[i][j]<0 || grWeightsPlast[i][j]>GRSYNWEIGHTMAX)
////			{
////				cout<<j<<" "<<grWeightsPlast[i][j]<<endl;
////			}
////		}
//	}
//}

//void MainW::calcGRPlastTempMetricPC(ofstream &outfile)
//{
//	for(int i=calcTempMetricBinN; i<=calcTempMetricBinN; i++)
//	{
//		cout<<i<<endl;
//
//#pragma omp parallel for schedule(static)
//		for(int j=0; j<NUMPC; j++)
//		{
//			double maxLTDBinDiff;
//
//			double lastLTDBinDiff;
//			double lastLTPBinDiff;
//
//			maxLTDBinDiff=0;
//
//			lastLTDBinDiff=0;
//			lastLTPBinDiff=0;
//
//			calcGRLTDSynWeightPC(i, 1, j);
////			cout<<"here1"<<endl;
//
//			calcGRPlastPopActPC(i, j);
////			cout<<"here2"<<endl;
//			maxLTDBinDiff=calcGRPlastPopActDiffPC(i, j);
////			cout<<"here3"<<endl;
//			lastLTDBinDiff=maxLTDBinDiff;
//
//			for(int k=0; k<200; k++)
//			{
//				double curLTDBinDiff;
//				double curLTPBinDiff;
//
//				calcGRLTPSynWeightPC(i, maxLTDBinDiff, j);
////				cout<<k<<" here4"<<endl;
//
//				calcGRPlastPopActPC(i, j);
////				cout<<k<<" here5"<<endl;
//				curLTPBinDiff=calcGRPlastPopActDiffPC(i, j);
////				cout<<k<<" here6"<<endl;
//
//				calcGRLTDSynWeightPC(i, (curLTPBinDiff<maxLTDBinDiff)*(1-(curLTPBinDiff/maxLTDBinDiff)), j);
////				cout<<k<<" here7"<<endl;
//
//				calcGRPlastPopActPC(i, j);
////				cout<<k<<" here8"<<endl;
//				curLTDBinDiff=calcGRPlastPopActDiffPC(i, j);
////				cout<<k<<" here9"<<endl;
//
//				lastLTPBinDiff=curLTPBinDiff;
//				lastLTDBinDiff=curLTDBinDiff;
//			}
//		}
//	}
//}

//void MainW::initGRPlastTempVars()
//{
//	for(int i=0; i<NUMBINS; i++)
//	{
//		for(int j=0; j<NUMGR; j++)
//		{
//			grWeightsPlast[i][j]=1;
//			grWeightsPlastPC[i][j]=1;
//		}
//	}
//}

//void MainW::calcGRLTDSynWeight(int binN, float scale)
//{
//	if(binN<0 || binN>=NUMBINS)
//	{
//		return;
//	}
//
//	if(scale<=0)
//	{
//		return;
//	}
//
//#pragma omp parallel for schedule(static)
//	for(int i=0; i<NUMGR; i++)
//	{
//		float synWeight;
//		synWeight=grWeightsPlast[binN][i];
//
//		for(int j=binN-TEMPMETSLIDINGW+1; j<=binN-(TEMPMETSLIDINGW/2); j++)
//		{
////			float spikesPerTrial;
//			if(j<0)
//			{
//				continue;
//			}
//
////			spikesPerTrial=pshGRTrans[i][j]/((float)numTrials);
//
//			synWeight=synWeight-(ratesGRTrans[i][j]*LTDSTEP*scale);//spikesPerTrial
//			synWeight=(synWeight>0)*synWeight;
//		}
//		grWeightsPlast[binN][i]=synWeight;
//	}
//}

//void MainW::calcGRLTDSynWeightPC(int binN, float scale, int pcN)
//{
//	if(binN<0 || binN>=NUMBINS)
//	{
//		return;
//	}
//	if(scale<=0)
//	{
//		return;
//	}
//	if(pcN<0 || pcN>=NUMPC)
//	{
//		return;
//	}
//
//	for(int i=pcN*(NUMGR/NUMPC); i<(pcN+1)*(NUMGR/NUMPC); i++)
//	{
//		float synWeight;
//		synWeight=grWeightsPlastPC[binN][i];
//
//		for(int j=binN-TEMPMETSLIDINGW+1; j<=binN-(TEMPMETSLIDINGW/2); j++)
//		{
////			float spikesPerTrial;
//			if(j<0)
//			{
//				continue;
//			}
//
////			spikesPerTrial=pshGRTrans[i][j]/((float)numTrials);
//
//			synWeight=synWeight-(ratesGRTrans[i][j]*LTDSTEP*scale);//spikesPerTrial
//			synWeight=(synWeight>0)*synWeight;
//		}
//		grWeightsPlastPC[binN][i]=synWeight;
//
//	}
//}

//void MainW::calcGRLTPSynWeight(int binN, double maxBinLTDDiff)
//{
//	double synWeightScale[NUMBINS];
//
//	if(binN<0 || binN>=NUMBINS)
//	{
//		return;
//	}
//
//	for(int i=0; i<NUMBINS; i++)
//	{
//		synWeightScale[i]=grPopActDiffPlast[binN][i]/maxBinLTDDiff;
//		synWeightScale[i]=(synWeightScale[i]>0)*synWeightScale[i];
////		synWeightScale[i]=1;
////		cout<<synWeightScale[i]<<" ";
//	}
////	cout<<endl;
//
//#pragma omp parallel for schedule(static)
//	for(int i=0; i<NUMGR; i++)
//	{
//		float synWeight;
//		synWeight=grWeightsPlast[binN][i];
//
////		cout<<i<<" "<<synWeight<<endl;
//		for(int j=0; j<NUMBINS; j++)
//		{
////			float spikesPerTrial;
//			if(j>=binN-TEMPMETSLIDINGW+1 && j<=binN-(TEMPMETSLIDINGW/2))
//			{
//				continue;
//			}
//
////			spikesPerTrial=pshGRTrans[i][j]/((float)numTrials);
//
//			synWeight=synWeight+(ratesGRTrans[i][j]*LTPSTEP*synWeightScale[j]);//spikesPerTrial
//			synWeight=(synWeight<GRSYNWEIGHTMAX)*synWeight+(!(synWeight<GRSYNWEIGHTMAX))*GRSYNWEIGHTMAX;
//		}
//
//		grWeightsPlast[binN][i]=synWeight;
//	}
//}

//void MainW::calcGRLTPSynWeightPC(int binN, double maxBinLTDDiff, int pcN)
//{
//	double synWeightScale[NUMBINS];
//
//	if(binN<0 || binN>=NUMBINS)
//	{
//		return;
//	}
//	if(pcN<0 || pcN>=NUMPC)
//	{
//		return;
//	}
//
//	for(int i=0; i<NUMBINS; i++)
//	{
//		synWeightScale[i]=grPopActDiffPlastPC[binN][pcN][i]/maxBinLTDDiff;
//		synWeightScale[i]=(synWeightScale[i]>0)*synWeightScale[i];
//	}
//
//	for(int i=pcN*(NUMGR/NUMPC); i<(pcN+1)*(NUMGR/NUMPC); i++)
//	{
//		float synWeight;
//		synWeight=grWeightsPlastPC[binN][i];
//
////		cout<<i<<" "<<synWeight<<endl;
//		for(int j=0; j<NUMBINS; j++)
//		{
////			float spikesPerTrial;
//			if(j>=binN-TEMPMETSLIDINGW+1 && j<=binN-(TEMPMETSLIDINGW/2))
//			{
//				continue;
//			}
//
////			spikesPerTrial=pshGRTrans[i][j]/((float)numTrials);
//
//			synWeight=synWeight+(ratesGRTrans[i][j]*LTPSTEP*synWeightScale[j]);//spikesPerTrial
//			synWeight=(synWeight<GRSYNWEIGHTMAX)*synWeight+(!(synWeight<GRSYNWEIGHTMAX))*GRSYNWEIGHTMAX;
//		}
//
//		grWeightsPlastPC[binN][i]=synWeight;
//	}
//}

//void MainW::calcGRPlastPopAct(int binN)
//{
//	if(binN<0 || binN>=NUMBINS)
//	{
//		return;
//	}
//
//#pragma omp parallel for schedule(static)
//	for(int i=0; i<NUMBINS; i++)
//	{
//		double binActSum;
//
//		binActSum=0;
//		for(int j=0; j<NUMGR; j++)
//		{
//			double spikes;
//
//			spikes=grWeightsPlast[binN][j]*pshGR[i][j];
//			binActSum=binActSum+spikes;
//		}
//
//		grPopActPlast[binN][i]=binActSum/numTrials;
//	}
//}

//void MainW::calcGRPlastPopActPC(int binN, int pcN)
//{
//	if(binN<0 || binN>=NUMBINS)
//	{
//		return;
//	}
//	if(pcN<0 || pcN>=NUMPC)
//	{
//		return;
//	}
//
////	cout<<binN<<" "<<pcN<<endl;
//
//	for(int i=0; i<NUMBINS; i++)
//	{
//		double binActSum;
//
//		binActSum=0;
//		for(int j=pcN*(NUMGR/NUMPC); j<(pcN+1)*(NUMGR/NUMPC); j++)
//		{
//			double spikes;
//
////			cout<<j<<" ";
////			cout.flush();
//			spikes=grWeightsPlastPC[binN][j]*pshGR[i][j];
////			cout<<spikes<<" ";
////			cout.flush();
//			binActSum=binActSum+spikes;
////			cout<<binActSum<<" "<<endl;
//
//		}
////		cout<<"herew"<<endl;
//		grPopActPlastPC[binN][pcN][i]=binActSum/numTrials;
////		cout<<"herewdone"<<endl;
//	}
//}

//double MainW::calcGRPlastPopActDiff(int binN)
//{
//	if(binN<0 || binN>=NUMBINS)
//	{
//		return 0;
//	}
//#pragma omp parallel for schedule(static)
//	for(int i=0; i<NUMBINS; i++)
//	{
//		grPopActDiffPlast[binN][i]=grBinTotalSpikes[i]-grPopActPlast[binN][i];
//	}
//	return grPopActDiffPlast[binN][binN];
//}
//
//double MainW::calcGRPlastPopActDiffPC(int binN, int pcN)
//{
//	if(binN<0 || binN>=NUMBINS)
//	{
//		return 0;
//	}
//	if(pcN<0 || pcN>=NUMPC)
//	{
//		return 0;
//	}
//	for(int i=0; i<NUMBINS; i++)
//	{
//		grPopActDiffPlastPC[binN][pcN][i]=grBinTotalSpikesPC[pcN][i]-grPopActPlastPC[binN][pcN][i];
//	}
//	return grPopActDiffPlastPC[binN][pcN][binN];
//}

//void MainW::calcGRPlastPopActDiffSum(int binN)
//{
//	double sum;
//	sum=0;
//	for(int i=0; i<NUMBINS; i++)
//	{
//		sum=sum+grPopActDiffPlast[binN][i];
//	}
//
//	grPopActDiffSumPlast[binN]=sum;
//}
//
//void MainW::calcGRLTDPopSpec(int binN)
//{
//	double actSum;
//	actSum=0;
//	for(int i=binN-TEMPMETSLIDINGW+1; i<=binN; i++)
//	{
//		if(i<0)
//		{
//			continue;
//		}
//
//		actSum=actSum+grPopActDiffPlast[binN][i];
//	}
//
//	grPopActSpecPlast[binN]=actSum/grPopActDiffSumPlast[binN];
//}
//
//void MainW::calcGRLTDPopAmp(int binN)
//{
//	grPopActAmpPlast[binN]=grPopActDiffPlast[binN][binN]/((double)grBinTotalSpikes[binN]);
//}
