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
	ui.grDispStartNum->setMaximum(1023);

	this->setAttribute(Qt::WA_DeleteOnClose);

	app=a;

	grTotalCalced=false;
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
	cout<<"loading non cell type specific variables..."<<endl;
	infile.read((char *)&numTrials, sizeof(unsigned int));
	cout<<"number of trials: "<<numTrials<<endl;

	cout<<"loading MF PSH..."<<endl;
	infile.read((char *)pshMF, NUMBINS*NUMMF*sizeof(unsigned short));
	infile.read((char *)&pshMFMax, sizeof(unsigned short));

	cout<<"loading GO PSH..."<<endl;
	infile.read((char *)pshGO, NUMBINS*NUMGO*sizeof(unsigned short));
	infile.read((char *)&pshGOMax, sizeof(unsigned short));

	cout<<"loading GR PSH..."<<endl;
	infile.read((char *)pshGR, NUMBINS*NUMGR*sizeof(unsigned short));
	infile.read((char *)&pshGRMax, sizeof(unsigned short));


	pshValGR.clear();
	for(int i=0; i<NUMGR; i++)
	{
		for(int j=0; j<NUMBINS; j++)
		{
			pshGRTrans[i][j]=pshGR[j][i];
			if(pshGR[j][i]>numTrials/3)//pshGRMax/5)//10
			{
				vector<unsigned short> tempRow(NUMBINS);
				for(int k=0; k<NUMBINS; k++)
				{
					tempRow[k]=pshGR[k][i];
				}

				pshValGR.push_back(tempRow);
				break;
			}
		}
	}

	grTotalCalced=false;
	cout<<"done!"<<endl;
	infile.close();
}

void MainW::calcTempMetrics()
{
	ofstream outfile;
	QString fileName;

	fileName=QFileDialog::getOpenFileName(this, "Please specify where to save the data", "/", "");
	outfile.open(fileName.toStdString().c_str(), ios::out);

	calcGRTempSpecific();
	calcGRPopTempMetric();

	for(int i=0; i<NUMBINS; i++)
	{
		outfile<<grPopSpecMean<<" "<<grPopSpecSR<<endl;
	}
	outfile.close();
}


void MainW::calcGRTotalSpikes()
{
	if(grTotalCalced)
	{
		return;
	}

	for(int i=0; i<NUMGR; i++)
	{
		grTotalSpikes[i]=0;
	}
	for(int i=0; i<NUMBINS; i++)
	{
		for(int j=0; j<NUMGR; j++)
		{
			grTotalSpikes[j]=grTotalSpikes[j]+pshGR[i][j];
		}
	}

	grTotalCalced=true;
}

void MainW::calcGRTempSpecific()
{
	for(int i=0; i<NUMGR; i++)
	{
		short peakBin;
		float peakVal;

		PeakBin=-1;
		peakVal=0;
		for(int j=0; j<NUMBINS; j++)
		{
			int tempSum;
			tempSum=0;
			for(int k=j-TEMPMETSLIDINGW+1; k<=j; k++)
			{
				if(k<0)
				{
					continue;
				}
				tempSum=tempSum+pshGRTrans[i][k];
			}
			grTempSpecificity[i][j]=((float)tempSum)/grTotalSpikes[i];
			if(grTempSpecificity[i][j]>peakVal)
			{
				peakVal=grTempSpecificity[i][j];
				peakBin=j;
			}
		}

		grTempSpPeakBin[i]=peakBin;
		grTempSpPeakVal[i]=peakVal;
	}
}

void MainW::calcGRPopTempMetric()
{
	for(int i=0; i<NUMBINS; i++)
	{
//		vector<int> grSpecInd;
		bool grIsSpecific[NUMGR];
		int numSpecGR;
		float specAvg;
		unsigned int specGRSpSum;
		unsigned int grSpSum;

		memset((char *)grIsSpecific, 0, NUMGR*sizeof(bool));
//		grSpecInd.clear();

		numSpecGR=0;
		for(int j=i-TEMPMETSLIDINGW+1; j<=i; j++)
		{
			if(j<0)
			{
				continue;
			}
			for(int k=0; k<NUMGR; k++)
			{
				if(grTempSpPeakBin[k]==j && grTempSpPeakVal[k]*grTotalSpikes[k]>=numTrials)
				{
//					grSpecInd.push_back(grTempSpPeakBin);
					grIsSpecific[k]=true;
					numSpecGR++;
				}
			}
		}

		specAvg=0;
//		for(int j=0; j<grSpecVecSize; j++)
//		{
//			specAvg=specAvg+grTempSpPeakVal[grSpecInd[j]];
//		}
//		if(specAvg>0)
//		{
//			specAvg=specAvg/grSpecVecSize;
//		}
//
//		grPopSpecMean[i]=specAvg;

		specGRSpSum=0;
		grSpSum=0;
		for(int j=0; j<NUMGR; j++)
		{
			if(grIsSpecific[j])
			{
				specGRSpSum=specGRSpSum+grTempSpecificity[j][i]*grTotalSpikes[j];
				specAvg=specAvg+grTempSpPeakVal[j];
			}

			grSpSum=grSpSum+grTempSpecificity[j][i]*grTotalSpikes[j];
		}

		if(specAvg>0)
		{
			grPopSpecMean[i]=specAvg/numSpecGR;
		}

		if(grSpSum>0)
		{
			grPopSpecSR[i]=specGRSpSum/grSpSum;
		}
		else
		{
			grPopSpecSR[i]=0;
		}

	}
}
