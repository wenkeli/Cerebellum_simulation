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
	bool dispGR[NUMGR];

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


	pshActiveGR.clear();
	memset((char *)dispGR, false, NUMGR*sizeof(bool));
	for(int i=0; i<NUMGR; i++)
	{
		for(int j=0; j<NUMBINS; j++)
		{
			pshGRTrans[i][j]=pshGR[j][i];
			if(pshGR[j][i]>numTrials/3 && !dispGR[i])//pshGRMax/5)//10
			{
				vector<unsigned short> tempRow(NUMBINS);
				for(int k=0; k<NUMBINS; k++)
				{
					tempRow[k]=pshGR[k][i];
				}

				pshActiveGR.push_back(tempRow);
				dispGR[i]=true;
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

	cout<<"Calculating total spikes of individual cells"<<endl;
	calcGRTotalSpikes();
	cout<<"calculating individual temporal specificity"<<endl;
	calcGRTempSpecific();
	cout<<"calculating population metrics"<<endl;
	calcGRPopTempMetric();
	cout<<"writing results"<<endl;

	outfile<<"specGRSpM activeGRSpM totalGRSpM specGRActM activeGRActM totalGRActM spTotGRActR spActGRActR actTotGRActR"<<endl;
	for(int i=0; i<NUMBINS; i++)
	{
		outfile<<specGRPopSpMean[i]<<" "<<activeGRPopSpMean[i]<<" "<<totalGRPopSpMean[i]<<
				" "<<specGRPopActMean[i]<<" "<<activeGRPopActMean[i]<<" "<<totalGRPopActMean[i]<<
				" "<<spTotGRPopActR[i]<<" "<<spActGRPopActR[i]<<" "<<actTotGRPopActR[i]<<endl;
	}
	outfile.close();
	cout<<"done!"<<endl;
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
		grBinTotalSpikes[i]=0;

		for(int j=0; j<NUMGR; j++)
		{
			grTotalSpikes[j]=grTotalSpikes[j]+pshGR[i][j];

			grBinTotalSpikes[i]=grBinTotalSpikes[i]+pshGR[i][j];
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

		peakBin=-1;
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
//				cout<<"k="<<k<<" tempSum="<<tempSum<<endl;
			}
			if(grTotalSpikes[i]>0)
			{
				grTempSpecificity[i][j]=((float)tempSum)/((float)grTotalSpikes[i]);
			}
			else
			{
//				cout<<grTotalSpikes[i]<<" "<<tempSum<<" "<<endl;
				grTempSpecificity[i][j]=0;
			}
//			cout<<grTempSpecificity[i][j]<<endl;
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
		bool grIsActive[NUMGR];
		int numSpecGR;
		int numActiveGR;
		float specGRSpSum;
		float activeGRSpSum;
		float totalGRSpSum;
		unsigned int specGRActSum;
		unsigned int activeGRActSum;
		unsigned int totalGRActSum;
//		int grSpSumTemp;

		memset((char *)grIsSpecific, 0, NUMGR*sizeof(bool));
		memset((char *)grIsActive, 0, NUMGR*sizeof(bool));
//		grSpecInd.clear();

		numSpecGR=0;
		numActiveGR=0;
		for(int j=i; j<=i; j++)//i-TEMPMETSLIDINGW+1
		{
			if(j<0)
			{
				continue;
			}
			for(int k=0; k<NUMGR; k++)
			{
				if(grTempSpPeakVal[k]*grTotalSpikes[k]>=numTrials*2)
				{
					grIsActive[k]=true;
					numActiveGR++;

					if(grTempSpPeakBin[k]==j)
					{
						grIsSpecific[k]=true;
						numSpecGR++;
					}
				}
			}
		}

		specGRSpSum=0;
		activeGRSpSum=0;
		totalGRSpSum=0;
		specGRActSum=0;
		activeGRActSum=0;
		totalGRActSum=0;
//		grSpSumTemp=0;
//		cout<<grSpSum<<" "<<specGRSpSum<<" "<<specAvg<<endl;
		for(int j=0; j<NUMGR; j++)
		{
			if(grIsSpecific[j])
			{
				specGRActSum=specGRActSum+grTempSpecificity[j][i]*grTotalSpikes[j];
//				cout<<endl<<grTempSpecificity[j][i]<<" "<<grTotalSpikes[j]<<" "<<specGRSpSum<<" "<<grSpSum<<" "<<grSpSumTemp<<endl;
				specGRSpSum=specGRSpSum+grTempSpPeakVal[j];
			}

			if(grIsActive[j])
			{
				activeGRActSum=activeGRActSum+grTempSpecificity[j][i]*grTotalSpikes[j];
				activeGRSpSum=activeGRSpSum+grTempSpecificity[j][i];
			}
//			if(grTempSpecificity[j][i]*grTotalSpikes[j]<0 || grTempSpecificity[j][i]*grTotalSpikes[j]>100000)
//			{
//				cout<<grSpSum<<" ";
//				cout<<grTempSpecificity[j][i]*grTotalSpikes[j]<<" "<<grTempSpecificity[j][i]<<" "<<grTotalSpikes[j]<<"|";
//			}

//			grSpSumTemp=grSpSum;
			totalGRSpSum=totalGRSpSum+grTempSpecificity[j][i];
			totalGRActSum=totalGRActSum+grTempSpecificity[j][i]*grTotalSpikes[j];
//			if(grSpSum<0 && grSpSumTemp>=0)
//			{
//				cout<<grSpSumTemp<<" "<<grSpSum<<" "<<grTempSpecificity[j][i]<<" "<<grTotalSpikes[j]<<" "<<grTempSpecificity[j][i]*grTotalSpikes[j]<<endl;
//			}
		}

		numGRActive[i]=numActiveGR;
		numGRSpecific[i]=numSpecGR;

		if(specGRSpSum>0)
		{
			specGRPopSpMean[i]=specGRSpSum/((float)numSpecGR);
			specGRPopActMean[i]=specGRActSum/((float)numSpecGR);
		}
		if(activeGRSpSum>0)
		{
			activeGRPopSpMean[i]=activeGRSpSum/((float)numActiveGR);
			activeGRPopActMean[i]=activeGRActSum/((float)numActiveGR);

			spActGRPopActR[i]=((float)specGRActSum)/((float)activeGRActSum);
		}
		if(totalGRSpSum>0)
		{
			totalGRPopSpMean[i]=totalGRSpSum/((float)NUMGR);
			totalGRPopActMean[i]=totalGRActSum/((float)NUMGR);

			spTotGRPopActR[i]=((float)specGRActSum)/((float)totalGRActSum);
			actTotGRPopActR[i]=((float)activeGRActSum)/((float)totalGRActSum);
		}
	}
}
