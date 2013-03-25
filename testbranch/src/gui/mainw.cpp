#include "../../includes/gui/mainw.h"
#include "../../includes/gui/moc/moc_mainw.h"

using namespace std;

MainW::MainW(QApplication *application, QWidget *parent)
    : QMainWindow(parent)
{
	vector<unsigned int> gXs;
	vector<unsigned int> gYs;
	vector<unsigned int> dispSizes;
	vector<QColor> dispColors;

	vector<unsigned int> cts;

	fstream conPFile;
	fstream actPFile;

	QStringList args;

	ui.setupUi(this);

	app=application;

	args=app->arguments();

	conPFile.open(args[1].toStdString().c_str(), ios::in);
	actPFile.open(args[2].toStdString().c_str(), ios::in);

	simState=new CBMState(actPFile, conPFile, 1);

	innetCS=simState->getInnetConState();
	conP=simState->getConnectivityParams();

	conP->showParams(cout);

	gXs.push_back(conP->getGLX());
	gXs.push_back(conP->getGOX());
	gXs.push_back(conP->getGRX());

	gYs.push_back(conP->getGLY());
	gYs.push_back(conP->getGOY());
	gYs.push_back(conP->getGRY());

	dispSizes.push_back(1);
	dispSizes.push_back(1);
	dispSizes.push_back(1);

	dispColors.push_back(Qt::white);
	dispColors.push_back(Qt::red);
	dispColors.push_back(Qt::green);

	conView=new ConnectivityView(gXs, gYs, dispSizes,
			dispColors, conP->getGRX(), conP->getGRY(), Qt::black, "", this);

	conDispCTNames.push_back("GO");
	conDispCTNames.push_back("GO");
	conDispCTNames.push_back("MF");
	conDispCTNames.push_back("GO");
	conDispCTNames.push_back("GL");
	conDispCTNames.push_back("GO");
	conDispCTNames.push_back("MF");

	conDispCTMaxNs.push_back(conP->getNumGO());
	conDispCTMaxNs.push_back(conP->getNumGO());
	conDispCTMaxNs.push_back(conP->getNumMF());
	conDispCTMaxNs.push_back(conP->getNumGO());
	conDispCTMaxNs.push_back(conP->getNumGL());
	conDispCTMaxNs.push_back(conP->getNumGO());
	conDispCTMaxNs.push_back(conP->getNumMF());

	cts.resize(2);

	cts[0]=1;
	cts[1]=2;
	conDispCTs.push_back(cts);
	conDispCTs.push_back(cts);
	cts.resize(1);
	cts[0]=2;
	conDispCTs.push_back(cts);
	cts.resize(2);
	cts[0]=1;
	cts[1]=1;
	conDispCTs.push_back(cts);
	cts[0]=0;
	cts[1]=2;
	conDispCTs.push_back(cts);
	cts[0]=1;
	cts[1]=0;
	conDispCTs.push_back(cts);
	cts.resize(1);
	cts[0]=0;
	conDispCTs.push_back(cts);

	updateConCellT(0);
	conDispCellN=ui.cellNBox->value();
	{
		fstream dataIn;
		fstream rasterOut;
		stringstream fileName;
		ISpikeRaster *raster;
		vector<int> spikeTimes;
		vector<string> cellNames;

		cellNames.push_back("go");
		cellNames.push_back("bc");
		cellNames.push_back("sc");
		cellNames.push_back("pc");

//		dataIn.open("dataOut", ios::in|ios::binary);
//
//		data=new ECTrialsData(dataIn);
//
//		dataIn.close();

//		for(int i=0; i<cellNames.size(); i++)
//		{
//			raster=data->getRaster(cellNames[i]);
//
//			for(int j=0; j<raster->getNumCells(); j++)
//			{
//				vector<vector<int> > spikeTimes;
//				ct_int32_t numTrials;
//				ct_int32_t numSpikes;
//
//				spikeTimes=raster->getCellSpikeTimes(j, 0);
//
//				fileName.str("");
//				fileName<<cellNames[i]<<"_"<<j<<".raster";
//
//				rasterOut.open(fileName.str().c_str(), ios::out|ios::binary);
//
//				numTrials=spikeTimes.size();
//
//				rasterOut.write((char *)&numTrials, sizeof(ct_int32_t));
//
//				for(int k=0; k<numTrials; k++)
//				{
//					numSpikes=spikeTimes[k].size();
//
//					rasterOut.write((char *)&numSpikes, sizeof(ct_int32_t));
//					for(int m=0; m<numSpikes; m++)
//					{
//						ct_int32_t st;
//						st=spikeTimes[k][m];
//
//						rasterOut.write((char *)&st, sizeof(ct_int32_t));
//					}
//				}
//
//				rasterOut.close();
//			}
//		}
	}
}

MainW::~MainW()
{
	delete simState;
	delete conView;
}


void MainW::showConnection()
{
	vector<vector<unsigned int> > cellInds;
	vector<unsigned int> srcInd;

	srcInd.push_back(conDispCellN);

	switch(conDispCellT)
	{
	case 0:
		cellInds.push_back(srcInd);
		cellInds.push_back(innetCS->getpGOfromGRtoGOCon(conDispCellN));
		break;
	case 1:
		cellInds.push_back(srcInd);
		cellInds.push_back(innetCS->getpGOfromGOtoGRCon(conDispCellN));
		break;
	case 2:
		cellInds.push_back(innetCS->getpMFfromMFtoGRCon(conDispCellN));
		break;
	case 3:
		cellInds.push_back(srcInd);
		cellInds.push_back(innetCS->getpGOOutGOGOCon(conDispCellN));
		break;
	case 4:
		cellInds.push_back(srcInd);
		cellInds.push_back(innetCS->getpGLfromGLtoGRCon(conDispCellN));
		break;
	case 5:
		cellInds.push_back(srcInd);
		cellInds.push_back(innetCS->getpGOfromGOtoGLCon(conDispCellN));
		break;
	default:
		cellInds.push_back(innetCS->getpMFfromMFtoGLCon(conDispCellN));
		break;
	}

	conView->updateDisp(cellInds, conDispCTs[conDispCellT]);
}

void MainW::updateConCellN(int cellNum)
{
	conDispCellN=cellNum;
	showConnection();
}

void MainW::updateConCellT(int cellType)
{
	conDispCellT=cellType;

	cout<<"cellT #"<<conDispCellT<<endl;

	ui.cellNBox->setMinimum(0);
	ui.cellNBox->setMaximum(conDispCTMaxNs[conDispCellT]-1);

	ui.conCellNLabel->setText(conDispCTNames[conDispCellT]);
}
