/*
 * innet.cpp
 *
 *  Created on: Jun 21, 2011
 *      Author: consciousness
 */

#include "../../includes/innetmodules/innet.h"
#include "../../includes/globalvars.h"

InNet::InNet(const bool *actInMF)
{
	apMF=actInMF;

	//connectivity variables
	memset(numGROutPerMF, 0, numMF*sizeof(short));
	memset(numGOOutPerMF, 0, numMF*sizeof(char));
	memset(numGROutPerGO, 0, numGO*sizeof(short));
	memset(numGOOutPerGR, 0, numGR*sizeof(int));
	memset(numGOInPerGR, 0, numGR*sizeof(int));
	memset(numMFInPerGR, 0, numGR*sizeof(int));
	memset(numMFInPerGO, 0, numGO*sizeof(int));
	for(int i=0; i<numMF; i++)
	{
		for(int j=0; j<maxNumGROutPerMF; j++)
		{
			mfConMFOutGR[i][j]=numGR;
		}
		for(int j=0; j<maxNumGOOutPerMF; j++)
		{
			mfConMFOutGO[i][j]=numGO;
		}
	}
	for(int i=0; i<numGO; i++)
	{
		for(int j=0; j<maxNumGROutPerGO; j++)
		{
			goConGOOutGR[i][j]=0;
		}
	}
	for(int i=0; i<maxNumGOOutPerGR; i++)
	{
		for(int j=0; j<numGR; j++)
		{
			grConGROutGO[i][j]=numGO;
		}
	}
	for(int i=0; i<maxNumInPerGR; i++)
	{
		for(int j=0; j<numGR; j++)
		{
			grConGOOutGR[i][j]=numGO;
			grConMFOutGR[i][j]=numMF;
		}
	}
	for(int i=0; i<maxNumMFInPerGO; i++)
	{
		for(int j=0; j<numGO; j++)
		{
			goConMFOutGO[i][j]=numMF;
		}
	}
	for(int i=0; i<numGL; i++)
	{
		glomeruli[i].hasGODen=false;
		glomeruli[i].hasGOAx=false;
		glomeruli[i].hasMF=false;
		glomeruli[i].goDenInd=numGO;
		glomeruli[i].goAxInd=numGO;
		glomeruli[i].mfInd=numMF;
		glomeruli[i].numGRDen=0;
		for(int j=0; j<maxNumGRDenPerGL; j++)
		{
			glomeruli[i].grDenInds[j]=numGR;
		}
	}
	for(int i=0; i<numGR; i++)
	{
		for(int j=0; j<maxNumInPerGR; j++)
		{
			grConGRInGL[i][j]=numGL;
		}
	}
	for(int i=0; i<numGO; i++)
	{
		for(int j=0; j<maxNumGLInPerGO; j++)
		{
			goConGOInGL[i][j]=numGL;
		}
		for(int j=0; j<maxNumGLOutPerGO; j++)
		{
			goConGOOutGL[i][j]=numGL;
		}
	}
	for(int i=0; i<numMF; i++)
	{
		for(int j=0; j<numGLOutPerMF; j++)
		{
			mfConMFOutGL[i][j]=numGL;
		}
	}
	//end connectivity variables

	//mossy fiber variables
	memset(histMF, false, numMF*sizeof(bool));
	memset(apBufMF, 0, numMF*sizeof(unsigned int));
	//end mossy fiber variables

	//granule cell variables
	eLeakGR=-70.0;
	eGOGR=-80.0;
	eMFGR=0;
	threshMaxGR=-20.0;
	threshBaseGR=-42.0;
	gEIncGR=0.02*0.22;//*2;//1.4;//1;
	gIIncGR=0.022*4/3;//0.2*0.22*2;//2;//1;//1.2;//1.7;//2;
	gEDecayTGR=55.0;
	gEDecayGR=exp(-TIMESTEP/gEDecayTGR);
	gIDecayTGR=50.0;
	gIDecayGR=exp(-TIMESTEP/gIDecayTGR);
	threshDecayTGR=3.0;
	threshDecayGR=1-exp(-TIMESTEP/threshDecayTGR);
	gLeakGR=0.1/(6-TIMESTEP);

	//golgi cell variables
	eLeakGO=-70.0;
	eMGluRGO=-96.0;
	threshMaxGO=-10.0;
	threshBaseGO=-33.0;
	gMFIncGO=0.07*0.28/32;//64;
	gGRIncGO=0.02*0.28/8;//11;//14;//40;//20;//9;//14;
	gMGluRScaleGO=0; //0.000015;
	gMGluRIncScaleGO=0.7;
	mGluRScaleGO=0.1;
	gluScaleGO=.01;
	gLeakGO=0.02/(6-TIMESTEP);
	gMFDecayTGO=4.5;
	gMFDecayGO=exp(-TIMESTEP/gMFDecayTGO);
	gGRDecayTGO=4.5;
	gGRDecayGO=exp(-TIMESTEP/gGRDecayTGO);
	mGluRDecayGO=0.98;
	gMGluRIncDecayGO=0.98;
	gMGluRDecayGO=0.98;
	gluDecayGO=0.98;
	threshDecayTGO=20.0;
	threshDecayGO=1-exp(-TIMESTEP/threshDecayTGO);
	for(int i=0; i<numGO; i++)
	{
		vGO[i]=eLeakGO-10+randGen->Random()*20;
		threshGO[i]=threshBaseGO-10+randGen->Random()*20;

		gMFGO[i]=0;
		gGRGO[i]=0;
		gMGluRGO[i]=0;
		gMGluRIncGO[i]=0;
		mGluRGO[i]=0;
		gluGO[i]=0;
	}
	memset(apGO, false, numGO*sizeof(bool));
	memset(apBufGO, 0, numGO*sizeof(unsigned int));
	memset(inputMFGO, 0, numGO*sizeof(unsigned short));
	//end golgi cell variables

	//stellate cell variables
	eLeakSC=-60.0;
	gLeakSC=0.2/(6-TIMESTEP);
	gPFDecayTSC=4.15;
	gPFDecaySC=exp(-TIMESTEP/gPFDecayTSC);
	threshMaxSC=0;
	threshBaseSC=-50.0;
	threshDecayTSC=22.0;
	threshDecaySC=1-exp(-TIMESTEP/threshDecayTSC);
	pfIncSC=0.0011;//0.00135;
	for(int i=0; i<numSC; i++)
	{
		vSC[i]=eLeakSC;
		threshSC[i]=threshBaseSC;
		gPFSC[i]=0;
	}
	memset(apSC, false, numSC*sizeof(bool));
	memset(apBufSC, 0, numSC*sizeof(unsigned int));
	//end stellate cell variables

	connectNetwork();

	initCUDA();
}

InNet::InNet(ifstream &infile, const bool *actInMF)
{
	apMF=actInMF;

	infile.read((char *)glomeruli, numGL*sizeof(Glomerulus));
	infile.read((char *)grConGRInGL, numGR*maxNumInPerGR*sizeof(unsigned int));
	infile.read((char *)goConGOInGL, numGO*maxNumGLInPerGO*sizeof(unsigned int));
	infile.read((char *)goConGOOutGL, numGO*maxNumGLOutPerGO*sizeof(unsigned int));
	infile.read((char *)mfConMFOutGL, numMF*numGLOutPerMF*sizeof(unsigned int));

	infile.read((char *)histMF, numMF*sizeof(bool));
	infile.read((char *)apBufMF, numMF*sizeof(unsigned int));
	infile.read((char *)numGROutPerMF, numMF*sizeof(short));
	infile.read((char *)mfConMFOutGR, numMF*maxNumGROutPerMF*sizeof(unsigned int));
	infile.read((char *)numGOOutPerMF, numMF*sizeof(char));
	infile.read((char *)mfConMFOutGO, numMF*maxNumGOOutPerMF*sizeof(unsigned int));

	eLeakGO=-70;
	eMGluRGO=-96;
	threshMaxGO=-10;
	threshBaseGO=-33;
	gMFIncGO=0.07*0.28/32;//64;
	gGRIncGO=0.02*0.28/8;//11;//14;//40;//20;//9;//14;
	gMGluRScaleGO=0; //0.000015;
	gMGluRIncScaleGO=0.7;
	mGluRScaleGO=0.1;
	gluScaleGO=.01;
	gLeakGO=0.02/(6-TIMESTEP);
	gMFDecayTGO=4.5;
	gMFDecayGO=exp(-TIMESTEP/gMFDecayTGO);
	gGRDecayTGO=4.5;
	gGRDecayGO=exp(-TIMESTEP/gGRDecayTGO);
	mGluRDecayGO=0.98;
	gMGluRIncDecayGO=0.98;
	gMGluRDecayGO=0.98;
	gluDecayGO=0.98;
	threshDecayTGO=20;
	threshDecayGO=1-exp(-TIMESTEP/threshDecayTGO);
	infile.read((char *)vGO, numGO*sizeof(float));
	infile.read((char *)threshGO, numGO*sizeof(float));
	infile.read((char *)apGO, numGO*sizeof(bool));
	infile.read((char *)apBufGO, numGO*sizeof(unsigned int));
	infile.read((char *)inputMFGO, numGO*sizeof(unsigned short));
	infile.read((char *)gMFGO, numGO*sizeof(float));
	infile.read((char *)gGRGO, numGO*sizeof(float));
	infile.read((char *)gMGluRGO, numGO*sizeof(float));
	infile.read((char *)gMGluRIncGO, numGO*sizeof(float));
	infile.read((char *)mGluRGO, numGO*sizeof(float));
	infile.read((char *)gluGO, numGO*sizeof(float));
	infile.read((char *)numMFInPerGO, numGO*sizeof(int));
	infile.read((char *)goConMFOutGO, numGO*maxNumMFInPerGO*sizeof(unsigned int));
	infile.read((char *)numGROutPerGO, numGO*sizeof(int));
	infile.read((char *)goConGOOutGR, numGO*maxNumGROutPerGO*sizeof(unsigned int));

	eLeakGR=-70;
	eGOGR=-80;
	eMFGR=0;
	threshMaxGR=-20;
	threshBaseGR=-42;
	gEIncGR=0.02*0.22;//*2;//1.4;//1;
	gIIncGR=0.022*4/3;//0.2*0.22*2;//2;//1;//1.2;//1.7;//2;
	gEDecayTGR=55;
	gEDecayGR=exp(-TIMESTEP/gEDecayTGR);
	gIDecayTGR=50;
	gIDecayGR=exp(-TIMESTEP/gIDecayTGR);
	threshDecayTGR=3;
	threshDecayGR=1-exp(-TIMESTEP/threshDecayTGR);
	gLeakGR=0.1/(6-TIMESTEP);
	infile.read((char *)delayGOMasksGR, maxNumGOOutPerGR*numGR*sizeof(unsigned int));
	infile.read((char *)delayBCPCSCMaskGR, numGR*sizeof(unsigned int));
	infile.read((char *)numGOOutPerGR, numGR*sizeof(int));
	infile.read((char *)grConGROutGO, maxNumGOOutPerGR*numGR*sizeof(unsigned int));
	infile.read((char *)numGOInPerGR, numGR*sizeof(int));
	infile.read((char *)grConGOOutGR, maxNumInPerGR*numGR*sizeof(unsigned int));
	infile.read((char *)numMFInPerGR, numGR*sizeof(int));
	infile.read((char *)grConMFOutGR, maxNumInPerGR*numGR*sizeof(unsigned int));

	eLeakSC=-60.0;
	gLeakSC=0.2/(6-TIMESTEP);
	gPFDecayTSC=4.15;
	gPFDecaySC=exp(-TIMESTEP/gPFDecayTSC);
	threshMaxSC=0;
	threshBaseSC=-50;
	threshDecayTSC=22;
	threshDecaySC=1-exp(-TIMESTEP/threshDecayTSC);
	pfIncSC=0.0011;//0.00135;
	infile.read((char *)gPFSC, numSC*sizeof(float));
	infile.read((char *)threshSC, numSC*sizeof(float));
	infile.read((char *)vSC, numSC*sizeof(float));
	infile.read((char *)apSC, numSC*sizeof(bool));
	infile.read((char *)apBufSC, numSC*sizeof(unsigned int));

	cout<<"input net read: "<<" "<<vGO[0]<<" "<<vSC[numSC-1]<<endl;

	initCUDA();
}

InNet::~InNet()
{
	//gpu variables
	//MF variables
	cudaFreeHost(apMFH);
	cudaFree(apMFGPU);

	//GR variables
	cudaFreeHost(outputGRH);
	cudaFree(outputGRGPU);
	cudaFree(vGRGPU);
	cudaFree(gKCaGRGPU);
	cudaFree(gEGRGPU);
	cudaFree(gEGRSumGPU);
	cudaFree(gIGRGPU);
	cudaFree(gIGRSumGPU);
	cudaFree(apBufGRGPU);
	cudaFree(threshGRGPU);
	cudaFree(delayGOMasksGRGPU);
	cudaFree(delayBCPCSCMaskGRGPU);
	cudaFree(numGOOutPerGRGPU);
	cudaFree(grConGROutGOGPU);
	cudaFree(numGOInPerGRGPU);
	cudaFree(grConGOOutGRGPU);
	cudaFree(numMFInPerGRGPU);
	cudaFree(grConMFOutGRGPU);
	cudaFree(historyGRGPU);

	//GO variables
	cudaFreeHost(apGOH);
	cudaFree(apGOGPU);
	cudaFree(grInputGOGPU);
	cudaFree(grInputGOSumGPU);

	//BC variables
	cudaFreeHost(inputSumPFBCH);
	cudaFree(inputPFBCGPU);
	cudaFree(inputSumPFBCGPU);

	//SC variables
	cudaFreeHost(inputSumPFSCH);
	cudaFree(inputPFSCGPU);
	cudaFree(inputSumPFSCGPU);
	//end gpu variables
}

void InNet::exportState(ofstream &outfile)
{
	outfile.write((char *)glomeruli, numGL*sizeof(Glomerulus));
	outfile.write((char *)grConGRInGL, numGR*maxNumInPerGR*sizeof(unsigned int));
	outfile.write((char *)goConGOInGL, numGO*maxNumGLInPerGO*sizeof(unsigned int));
	outfile.write((char *)goConGOOutGL, numGO*maxNumGLOutPerGO*sizeof(unsigned int));
	outfile.write((char *)mfConMFOutGL, numMF*numGLOutPerMF*sizeof(unsigned int));

	outfile.write((char *)histMF, numMF*sizeof(bool));
	outfile.write((char *)apBufMF, numMF*sizeof(unsigned int));
	outfile.write((char *)numGROutPerMF, numMF*sizeof(short));
	outfile.write((char *)mfConMFOutGR, numMF*maxNumGROutPerMF*sizeof(unsigned int));
	outfile.write((char *)numGOOutPerMF, numMF*sizeof(char));
	outfile.write((char *)mfConMFOutGO, numMF*maxNumGOOutPerMF*sizeof(unsigned int));

	outfile.write((char *)vGO, numGO*sizeof(float));
	outfile.write((char *)threshGO, numGO*sizeof(float));
	outfile.write((char *)apGO, numGO*sizeof(bool));
	outfile.write((char *)apBufGO, numGO*sizeof(unsigned int));
	outfile.write((char *)inputMFGO, numGO*sizeof(unsigned short));
	outfile.write((char *)gMFGO, numGO*sizeof(float));
	outfile.write((char *)gGRGO, numGO*sizeof(float));
	outfile.write((char *)gMGluRGO, numGO*sizeof(float));
	outfile.write((char *)gMGluRIncGO, numGO*sizeof(float));
	outfile.write((char *)mGluRGO, numGO*sizeof(float));
	outfile.write((char *)gluGO, numGO*sizeof(float));
	outfile.write((char *)numMFInPerGO, numGO*sizeof(int));
	outfile.write((char *)goConMFOutGO, numGO*maxNumMFInPerGO*sizeof(unsigned int));
	outfile.write((char *)numGROutPerGO, numGO*sizeof(int));
	outfile.write((char *)goConGOOutGR, numGO*maxNumGROutPerGO*sizeof(unsigned int));

	outfile.write((char *)delayGOMasksGR, maxNumGOOutPerGR*numGR*sizeof(unsigned int));
	outfile.write((char *)delayBCPCSCMaskGR, numGR*sizeof(unsigned int));
	outfile.write((char *)numGOOutPerGR, numGR*sizeof(int));
	outfile.write((char *)grConGROutGO, maxNumGOOutPerGR*numGR*sizeof(unsigned int));
	outfile.write((char *)numGOInPerGR, numGR*sizeof(int));
	outfile.write((char *)grConGOOutGR, maxNumInPerGR*numGR*sizeof(unsigned int));
	outfile.write((char *)numMFInPerGR, numGR*sizeof(int));
	outfile.write((char *)grConMFOutGR, maxNumInPerGR*numGR*sizeof(unsigned int));

	outfile.write((char *)gPFSC, numSC*sizeof(float));
	outfile.write((char *)threshSC, numSC*sizeof(float));
	outfile.write((char *)vSC, numSC*sizeof(float));
	outfile.write((char *)apSC, numSC*sizeof(bool));
	outfile.write((char *)apBufSC, numSC*sizeof(unsigned int));
}

void InNet::exportActGRDisp(vector<bool> &apRaster, int numCells)
{
	cudaMemcpy(outputGRH, outputGRGPU, numCells*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	for(int i=0; i<numCells; i++)
	{
		apRaster[i]=(bool)(outputGRH[i]);
	}
}
void InNet::exportActGODisp(vector<bool> &apRaster, int numCells)
{
	for(int i=0; i<numCells; i++)
	{
		apRaster[i]=apGO[i];
	}
}
void InNet::exportActSCDisp(SCBCPCActs &sbp)
{
	for(int i=0; i<numSC; i++)
	{
		sbp.apSC[i]=apSC[i];
	}
}

void InNet::initCUDA()
{
	cudaError_t error;
	cout<<"input network cuda init..."<<endl;
	initMFCUDA();
	error=cudaGetLastError();
	cout<<"CUDA MF init: "<<cudaGetErrorString(error)<<endl;
	initGRCUDA();
	error=cudaGetLastError();
	cout<<"CUDA gr init: "<<cudaGetErrorString(error)<<endl;
	initGOCUDA();
	error=cudaGetLastError();
	cout<<"CUDA go init: "<<cudaGetErrorString(error)<<endl;
	initBCCUDA();
	error=cudaGetLastError();
	cout<<"CUDA bc init: "<<cudaGetErrorString(error)<<endl;
	initSCCUDA();
	error=cudaGetLastError();
	cout<<"CUDA sc init: "<<cudaGetErrorString(error)<<endl;

}
void InNet::initMFCUDA()
{
	cudaMallocHost((void **)&apMFH, numMF*sizeof(unsigned int));
	cudaMalloc((void **)&apMFGPU, numMF*sizeof(unsigned int));
}
void InNet::initGRCUDA()
{
	cudaMallocHost((void **)&outputGRH, numGR*sizeof(unsigned char));

	//allocate memory for GPU
	cudaMalloc((void **)&outputGRGPU, numGR*sizeof(unsigned char));

	cudaMalloc((void **)&vGRGPU, numGR*sizeof(float));
	cudaMalloc((void **)&gKCaGRGPU, numGR*sizeof(float));
	cudaMallocPitch((void **)&gEGRGPU, (size_t *)&gEGRGPUP,
			numGR*sizeof(float), maxNumInPerGR);
	cudaMalloc((void **)&gEGRSumGPU, numGR*sizeof(float));
	cudaMallocPitch((void **)&gIGRGPU, (size_t *)&gIGRGPUP,
			numGR*sizeof(float), maxNumInPerGR);
	cudaMalloc((void **)&gIGRSumGPU, numGR*sizeof(float));

	cudaMalloc((void **)&apBufGRGPU, numGR*sizeof(unsigned int));
	cudaMalloc((void **)&threshGRGPU, numGR*sizeof(float));

	//variables for conduction delays
	cudaMalloc((void **)&delayBCPCSCMaskGRGPU, numGR*sizeof(unsigned int));
	cudaMallocPitch((void **)&delayGOMasksGRGPU, (size_t *)&delayGOMasksGRGPUP,
			numGR*sizeof(unsigned int), maxNumGOOutPerGR);
	//end conduction delay

	//connectivity
	cudaMallocPitch((void **)&grConGROutGOGPU, (size_t *)&grConGROutGOGPUP,
				numGR*sizeof(unsigned int), maxNumGOOutPerGR);
	cudaMalloc((void **)&numGOOutPerGRGPU, numGR*sizeof(int));

	cudaMallocPitch((void **)&grConGOOutGRGPU, (size_t *)&grConGOOutGRGPUP,
				numGR*sizeof(unsigned int), maxNumInPerGR);
	cudaMalloc((void **)&numGOInPerGRGPU, numGR*sizeof(int));

	cudaMallocPitch((void **)&grConMFOutGRGPU, (size_t *)&grConMFOutGRGPUP,
				numGR*sizeof(unsigned int), maxNumInPerGR);
	cudaMalloc((void **)&numMFInPerGRGPU, numGR*sizeof(int));
	//end connectivity

	cudaMalloc((void **)&historyGRGPU, numGR*sizeof(unsigned long));
	//end GPU memory allocation


	//initialize GR GPU variables
	{
		//temp variables to assign values to for copying over to the GPU
		float vGRIni[numGR];
		float gKCaGRIni[numGR];

		float gEGRIni[maxNumInPerGR][numGR];
		float gEGRSumIni[numGR];
		float gIGRIni[maxNumInPerGR][numGR];
		float gIGRSumIni[numGR];

		unsigned int apBufGRIni[numGR];
		float threshGRIni[numGR];

		unsigned long historyGRIni[numGR];

		//initialize values to copy over to GPU
		for(int i=0; i<numGR; i++)
		{
			vGRIni[i]=eLeakGR;//-10+randGen.Random()*20;
			gKCaGRIni[i]=0;

			for(int j=0; j<maxNumInPerGR; j++)
			{
				gEGRIni[j][i]=0;
				gIGRIni[j][i]=0;
			}
			gEGRSumIni[i]=0;
			gIGRSumIni[i]=0;

			apBufGRIni[i]=0;
			threshGRIni[i]=threshBaseGR;

			historyGRIni[i]=0;
		}
		//end value initialization
		cudaMemcpy(vGRGPU, vGRIni, numGR*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(gKCaGRGPU, gKCaGRIni, numGR*sizeof(float), cudaMemcpyHostToDevice);

		cudaMemcpy2D(gEGRGPU, gEGRGPUP,
				gEGRIni, numGR*sizeof(float),
				numGR*sizeof(float), maxNumInPerGR, cudaMemcpyHostToDevice);
		cudaMemcpy(gEGRSumGPU, gEGRSumIni, numGR*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy2D(gIGRGPU, gIGRGPUP,
				gIGRIni, numGR*sizeof(float),
				numGR*sizeof(float), maxNumInPerGR, cudaMemcpyHostToDevice);
		cudaMemcpy(gIGRSumGPU, gIGRSumIni, numGR*sizeof(float), cudaMemcpyHostToDevice);

		cudaMemcpy(apBufGRGPU, apBufGRIni, numGR*sizeof(unsigned int), cudaMemcpyHostToDevice);

		cudaMemcpy(threshGRGPU, threshGRIni, numGR*sizeof(float), cudaMemcpyHostToDevice);

		cudaMemcpy2D(delayGOMasksGRGPU, delayGOMasksGRGPUP,
				delayGOMasksGR, numGR*sizeof(unsigned int),
				numGR*sizeof(unsigned int), maxNumGOOutPerGR, cudaMemcpyHostToDevice);

		cudaMemcpy(delayBCPCSCMaskGRGPU, delayBCPCSCMaskGR,
				numGR*sizeof(unsigned int), cudaMemcpyHostToDevice);

		cudaMemcpy2D(grConGROutGOGPU, grConGROutGOGPUP,
				grConGROutGO, numGR*sizeof(unsigned int),
				numGR*sizeof(unsigned int), maxNumGOOutPerGR, cudaMemcpyHostToDevice);
		cudaMemcpy(numGOOutPerGRGPU, numGOOutPerGR, numGR*sizeof(int), cudaMemcpyHostToDevice);

		cudaMemcpy2D(grConGOOutGRGPU, grConGOOutGRGPUP,
				grConGOOutGR, numGR*sizeof(unsigned int),
				numGR*sizeof(unsigned int), maxNumInPerGR, cudaMemcpyHostToDevice);
		cudaMemcpy(numGOInPerGRGPU, numGOInPerGR, numGR*sizeof(int), cudaMemcpyHostToDevice);

		cudaMemcpy2D(grConMFOutGRGPU, grConMFOutGRGPUP,
				grConMFOutGR, numGR*sizeof(unsigned int),
				numGR*sizeof(unsigned int), maxNumInPerGR, cudaMemcpyHostToDevice);
		cudaMemcpy(numMFInPerGRGPU, numMFInPerGR, numGR*sizeof(int), cudaMemcpyHostToDevice);



		cudaMemcpy(historyGRGPU, historyGRIni, numGR*sizeof(unsigned long), cudaMemcpyHostToDevice);
		//end copying to GPU
	}
}
void InNet::initGOCUDA()
{
	cudaMallocHost((void **)&apGOH, numGO*sizeof(unsigned int));

	cudaMallocHost((void **)&grInputGOSumH, numGO*sizeof(unsigned int));

	//allocate gpu memory
	cudaMalloc((void **)&apGOGPU, numGO*sizeof(unsigned int));

	cudaMallocPitch((void **)&grInputGOGPU, (size_t *)&grInputGOGPUP,
			numGO*sizeof(unsigned int), 1024);
	cudaMalloc((void **)&grInputGOSumGPU, numGO*sizeof(unsigned int));

	{
		unsigned int apGOIni[numGO];
		unsigned int grInputGOSumIni[numGO];

		//initialize host memory
		for(int i=0; i<numGO; i++)
		{
			grInputGOSumH[i]=0;
			apGOH[i]=0;
			apGOIni[i]=0;
			grInputGOSumIni[i]=0;
		}

		cudaMemcpy(apGOGPU, apGOIni, numGO*sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaMemcpy(grInputGOSumGPU, grInputGOSumIni, numGO*sizeof(unsigned int), cudaMemcpyHostToDevice);
	}
}
void InNet::initBCCUDA()
{
	//allocate host memory
	cudaMallocHost((void **)&inputSumPFBCH, numBC*sizeof(unsigned int));

	//allocate GPU memory
	cudaMallocPitch((void **)&inputPFBCGPU, (size_t *)&inputPFBCGPUP,
			numPFInPerBC*sizeof(unsigned int), numBC);
	cudaMalloc((void **)&inputSumPFBCGPU, numBC*sizeof(unsigned int));
	//end GPU allocation

	//initialize host variables
	for(int i=0; i<numBC; i++)
	{
		inputSumPFBCH[i]=0;
	}
}
void InNet::initSCCUDA()
{
	//allocate host memory
	cudaMallocHost((void **)&inputSumPFSCH, numSC*sizeof(unsigned int));

	//allocate GPU memory
	cudaMallocPitch((void **)&inputPFSCGPU, (size_t *)&inputPFSCGPUP,
			numPFInPerSC*sizeof(unsigned int), numSC);

	cudaMalloc((void **)&inputSumPFSCGPU, numSC*sizeof(unsigned int));
	//end GPU allocation

	//initialize host variables
	for(int i=0; i<numSC; i++)
	{
		inputSumPFSCH[i]=0;
	}
}

void InNet::connectNetwork()
{
	stringstream output;
	output.str("");
	assignGRGL(output);
	cout<<output.str()<<endl;

	output.str("");
	assignGOGL(output);
	cout<<output.str()<<endl;

	output.str("");
	assignMFGL(output);
	cout<<output.str()<<endl;

	output.str("");
	translateMFGL(output);
	cout<<output.str()<<endl;

	output.str("");
	translateGOGL(output);
	cout<<output.str()<<endl;

	output.str("");
	assignGRGO(output, maxNumGRInPerGO);
	cout<<output.str()<<endl;

	output.str("");
	assignGRDelays(output);
	cout<<output.str()<<endl;
}
void InNet::assignGRGL(stringstream &statusOut)
{
	float scaleGRGLX;
	float scaleGRGLY;

	scaleGRGLX=(float)grX/glX;
	scaleGRGLY=(float)grY/glY;

	for(char i=0; i<maxNumInPerGR; i++)
	{
		int numConnectedGR;
		bool  grConnected[numGR];

		numConnectedGR=0;
		memset(grConnected, false, numGR*sizeof(bool));

		while(numConnectedGR<numGR)
		{
			int grInd;
			int grPosX;
			int grPosY;
			int tempGRDenSpanGLX;
			int tempGRDenSpanGLY;
			int numGRPerGLLim;
			int attempts;
			bool complete;

			grInd=randGen->IRandom(0, numGR-1);

			if(grConnected[grInd])
			{
				continue;
			}

			grConnected[grInd]=true;
			numConnectedGR++;

			grPosX=grInd%grX;
			grPosY=(int)(grInd/grX);
			tempGRDenSpanGLX=grGLDenSpanGLX;
			tempGRDenSpanGLY=grGLDenSpanGLY;

			numGRPerGLLim=normNumGRDenPerGL;

			complete=false;
			for(attempts=0; attempts<60000; attempts++)
			{
				int tempGLPosX, tempGLPosY;
				int derivedGLIndex;
				bool unique;

				if(attempts==2000)
				{
					tempGRDenSpanGLX=tempGRDenSpanGLX*2;
					tempGRDenSpanGLY=tempGRDenSpanGLY*2;
				}
				if(attempts==20000)
				{
					numGRPerGLLim=maxNumGRDenPerGL;
				}

				tempGLPosX=(int)(grPosX/scaleGRGLX);
				tempGLPosY=(int)(grPosY/scaleGRGLY);

				tempGLPosX+=randGen->IRandom(-tempGRDenSpanGLX/2, tempGRDenSpanGLX/2);
				tempGLPosY+=randGen->IRandom(-tempGRDenSpanGLY/2, tempGRDenSpanGLY/2);

				tempGLPosX=(tempGLPosX%glX+glX)%glX;
				tempGLPosY=(tempGLPosY%glY+glY)%glY;

				derivedGLIndex=tempGLPosY*glX+tempGLPosX;

				unique=true;
				for(int j=0; j<i; j++)
				{
					if(derivedGLIndex==grConGRInGL[grInd][j])
					{
						unique=false;
						break;
					}
				}
				if(!unique)
				{
					continue;
				}

				if(glomeruli[derivedGLIndex].numGRDen<numGRPerGLLim)
				{
					glomeruli[derivedGLIndex].grDenInds[glomeruli[derivedGLIndex].numGRDen]=grInd*4+i;
					glomeruli[derivedGLIndex].numGRDen++;
					grConGRInGL[grInd][i]=derivedGLIndex;
					complete=true;
					break;
				}
			}

			if(attempts>=60000 && !complete)
			{
				statusOut<<"incomplete GR to GL assignment for GR#"<<grInd<<endl;
			}
		}
	}

	statusOut<<"granule cells assigned to glomeruli."<<endl;
}
void InNet::assignGOGL(stringstream &statusOut)
{
	float scaleGLGOX;
	float scaleGLGOY;

	int numConnectedGO;
	bool goConnected[numGO];

	scaleGLGOX=(float)glX/goX;
	scaleGLGOY=(float)glY/goY;

	for(int i=0; i<maxNumGLInPerGO; i++)
	{
		numConnectedGO=0;
		memset(goConnected, false, numGO*sizeof(bool));

		while(numConnectedGO<numGO)
		{
			int goInd;
			int goPosX;
			int goPosY;
			int tempGODenSpanGLX;
			int tempGODenSpanGLY;
			int attempts;

			goInd=randGen->IRandomX(0, numGO-1);

			if(goConnected[goInd])
			{
				continue;
			}

			goConnected[goInd]=true;
			numConnectedGO++;

			goPosX=goInd%goX;
			goPosY=(int)(goInd/goX);

			tempGODenSpanGLX=goGLDenSpanGLX;
			tempGODenSpanGLY=goGLDenSpanGLY;

			for(attempts=0; attempts<1000000000; attempts++)
			{
				int tempGLPosX, tempGLPosY;
				int derivedGLIndex;

				if(attempts==50000)
				{
					tempGODenSpanGLX=tempGODenSpanGLX*2;
					tempGODenSpanGLY=tempGODenSpanGLY*2;
				}

				tempGLPosX=(int)(goPosX*scaleGLGOX+scaleGLGOX/2);
				tempGLPosY=(int)(goPosY*scaleGLGOY+scaleGLGOY/2);

				tempGLPosX+=randGen->IRandom(-tempGODenSpanGLX/2, tempGODenSpanGLX/2);
				tempGLPosY+=randGen->IRandom(-tempGODenSpanGLY/2, tempGODenSpanGLY/2);

				tempGLPosX=(tempGLPosX%glX+glX)%glX;
				tempGLPosY=(tempGLPosY%glY+glY)%glY;

				derivedGLIndex=tempGLPosY*glX+tempGLPosX;

				if(glomeruli[derivedGLIndex].hasGODen)
				{
					continue;
				}

				glomeruli[derivedGLIndex].hasGODen=true;
				glomeruli[derivedGLIndex].goDenInd=goInd;
				goConGOInGL[goInd][i]=derivedGLIndex;
				break;
			}
		}
	}
	statusOut<<"golgi cell dendrites assigned to glomeruli"<<endl;

	for(int i=0; i<maxNumGLOutPerGO; i++)
	{
		numConnectedGO=0;
		memset(goConnected, false, numGO*sizeof(bool));

		while(numConnectedGO<numGO)
		{
			int goInd;
			int goPosX;
			int goPosY;
			int tempGOAxSpanGLX;
			int tempGOAxSpanGLY;
			int attempts;

			goInd=randGen->IRandomX(0, numGO-1);

			if(goConnected[goInd])
			{
				continue;
			}

			goConnected[goInd]=true;
			numConnectedGO++;

			goPosX=goInd%goX;
			goPosY=(int)(goInd/goX);

			tempGOAxSpanGLX=goGLAxSpanGLX;
			tempGOAxSpanGLY=goGLAxSpanGLY;

			for(attempts=0; attempts<1000000000; attempts++)
			{
				int tempGLPosX, tempGLPosY;
				int derivedGLIndex;
				bool unique;

				if(attempts==50000)
				{
					tempGOAxSpanGLX=tempGOAxSpanGLX*2;
					tempGOAxSpanGLY=tempGOAxSpanGLY*2;
				}

				tempGLPosX=(int)(goPosX*scaleGLGOX+scaleGLGOX/2);
				tempGLPosY=(int)(goPosY*scaleGLGOY+scaleGLGOY/2);

				tempGLPosX+=randGen->IRandom(-tempGOAxSpanGLX/2, tempGOAxSpanGLX/2);
				tempGLPosY+=randGen->IRandom(-tempGOAxSpanGLY/2, tempGOAxSpanGLY/2);

				tempGLPosX=(tempGLPosX%glX+glX)%glX;
				tempGLPosY=(tempGLPosY%glY+glY)%glY;

				derivedGLIndex=tempGLPosY*glX+tempGLPosX;

				if(glomeruli[derivedGLIndex].hasGOAx)
				{
					continue;
				}

				glomeruli[derivedGLIndex].hasGOAx=true;
				glomeruli[derivedGLIndex].goAxInd=goInd;
				goConGOOutGL[goInd][i]=derivedGLIndex;
				break;
			}
		}
	}
	statusOut<<"golgi cell axons assigned to glomeruli"<<endl;
}
void InNet::assignMFGL(stringstream &statusOut)
{
	int lastMFSynCount;
	for(int i=0; i<numMF-1; i++)
	{
		for(int j=0; j<numGLOutPerMF; j++)
		{
			int glIndex;
			while(true)
			{
				glIndex=randGen->IRandom(0, numGL-1);
				if(!glomeruli[glIndex].hasMF)
				{
					glomeruli[glIndex].mfInd=i;
					glomeruli[glIndex].hasMF=true;
					mfConMFOutGL[i][j]=glIndex;
					break;
				}
			}
		}
	}

	lastMFSynCount=0;
	for(int i=0; i<numGL; i++)
	{
		if(!glomeruli[i].hasMF)
		{
			glomeruli[i].hasMF=true;
			glomeruli[i].mfInd=numMF-1;
			mfConMFOutGL[numMF-1][lastMFSynCount]=i;
			lastMFSynCount++;
		}
	}
	statusOut<<"mossy fibers assigned to glomeruli"<<endl;
}
void InNet::translateMFGL(stringstream &statusOut)
{
	for(int i=0; i<numMF; i++)
	{
		numGROutPerMF[i]=0;
		numGOOutPerMF[i]=0;
		for(int j=0; j<numGLOutPerMF; j++)
		{
			int glIndex;
			glIndex=mfConMFOutGL[i][j];
			if(glomeruli[glIndex].hasGODen)
			{
				unsigned int goInd;

				goInd=glomeruli[glIndex].goDenInd;
				mfConMFOutGO[i][numGOOutPerMF[i]]=goInd;
				numGOOutPerMF[i]++;

				goConMFOutGO[goInd][numMFInPerGO[goInd]]=i;
				numMFInPerGO[goInd]++;
			}

			for(int k=0; k<glomeruli[glIndex].numGRDen; k++)
			{
				unsigned int grInd;

				grInd=glomeruli[glIndex].grDenInds[k];
				mfConMFOutGR[i][k+numGROutPerMF[i]]=grInd;

				grInd=grInd/maxNumInPerGR;
				grConMFOutGR[numMFInPerGR[grInd]][grInd]=i;
				numMFInPerGR[grInd]++;
			}
			numGROutPerMF[i]=numGROutPerMF[i]+glomeruli[glIndex].numGRDen;
		}
	}
	statusOut<<"mossy fiber to glomeruli connection translated to:"<<endl;
	statusOut<<"mossy fiber to golgi cell connection,"<<endl;
	statusOut<<"mossy fiber to granule cell connection."<<endl;
}
void InNet::translateGOGL(stringstream &statusOut)
{
	for(int i=0; i<numGO; i++)
	{
		numGROutPerGO[i]=0;
		for(int j=0; j<maxNumGLOutPerGO; j++)
		{
			int glIndex;
			glIndex=goConGOOutGL[i][j];
			for(int k=0; k<glomeruli[glIndex].numGRDen; k++)
			{
				unsigned int grInd;

				grInd=glomeruli[glIndex].grDenInds[k];
				goConGOOutGR[i][k+numGROutPerGO[i]]=grInd;

				grInd=grInd/maxNumInPerGR;
				grConGOOutGR[numGOInPerGR[grInd]][grInd]=i;
				numGOInPerGR[grInd]++;
			}
			numGROutPerGO[i]=numGROutPerGO[i]+glomeruli[glIndex].numGRDen;
		}
	}
	statusOut<<"golgi to glomeruli connection translated to golgi to granule cell connection."<<endl;
}
void InNet::assignGRGO(stringstream &statusOut, unsigned int nGRInPerGO)
{
	float scaleX=(float) GRX/GOX;
	float scaleY=(float) GRY/GOY;

	for(int i=0; i<nGRInPerGO; i++)
	{
		int numConnectedGO;
		bool goConnected[numGO];

		numConnectedGO=0;
		memset(goConnected, false, numGO*sizeof(bool));

		while(numConnectedGO<numGO)
		{
			int goInd=randGen->IRandom(0, numGO-1);
			int goPosX;
			int goPosY;
			int tempGODenSpanX;
			int tempGODenSpanY;
			int attempts;
			bool complete;

			if(goConnected[goInd])
			{
				continue;
			}

			goConnected[goInd]=true;
			numConnectedGO++;

			//get golgi cell coordinates from the cell index
			goPosX=goInd%goX;
			goPosY=(int) goInd/goX;

			tempGODenSpanX=goPFDenSpanGRX;
			tempGODenSpanY=goPFDenSpanGRY;

			attempts;
			complete=false;
			for(attempts=0; attempts<50000; attempts++)
			{
				int tempGRPosX, tempGRPosY;
				double tempGRPosXf, tempGRPosYf; //to eliminate truncation errors

				int derivedGRIndex;

				//given a golgi cell coordinate, randomly find a granule cell coordinate within the denspan x and y of the golgi cell
				//% operations are to take care of wraparounds
				tempGRPosXf=(goPosX+0.5)*scaleX;
				tempGRPosYf=(goPosY+0.5)*scaleY;
				tempGRPosX=((int)lroundf(tempGRPosXf+(double)tempGODenSpanX*(randGen->Random()-0.5))%grX+grX)%grX;
				tempGRPosY=((int)lroundf(tempGRPosYf+(double)tempGODenSpanY*(randGen->Random()-0.5))%grY+grY)%grY;

				//get the granule cell index given a granule cell coordinate
				derivedGRIndex=tempGRPosY*grX+tempGRPosX;
				//if that granule cell is not saturated with synapses, make the connection
				if(numGOOutPerGR[derivedGRIndex]<maxNumGOOutPerGR)
				{
					grConGROutGO[numGOOutPerGR[derivedGRIndex]][derivedGRIndex]=goInd;
					numGOOutPerGR[derivedGRIndex]++;
					complete=true;
					break;
				}

				//if at the 5000th try still can't find an unsaturated granule cell, increase the den span to 10 less than
				//the granule cell grid, to increase the chance of making connection
				if(attempts==4999)
				{
					tempGODenSpanX=grX-10;
					tempGODenSpanY=grY-10;
				}
			}
			if(attempts>=50000 && !complete)
			{
				statusOut<<"incomplete GR to GO connections for GO#"<<goInd<<endl;
			}
		}
	}
	statusOut<<"granule to golgi synapses connected."<<endl;
}
void InNet::assignGRDelays(stringstream &statusOut)
{
	unsigned int delayMasks[8]={0x00000001, 0x00000002, 0x00000004, 0x00000008, 0x00000010, 0x00000020, 0x00000040, 0x00000080};

	//calculate delay masks for each granule cells
	for(int i=0; i<numGR; i++)
	{
		int grPosX;
		int grBCPCSCDist;


		//calculate x coordinate of GR position
		grPosX=i%grX;

		//calculate distance of GR to BC, PC, and SC, and assign time delay.
		grBCPCSCDist=abs(1024-grPosX); //todo: put constants as marcos in params
		delayBCPCSCMaskGR[i]=delayMasks[(int)grBCPCSCDist/147+1];//147+1];//[0]//256+1]; //todo: put consts as marcos

		for(int j=0; j<numGOOutPerGR[i]; j++)
		{
			int goPosX;
			int grGODist;
			int distTemp;
			//calculate x position of GO that the GR is outputting to
			goPosX=grConGROutGO[j][i]%goX;

			//convert from golgi coordinate to granule coordinates
			goPosX=(goPosX+0.5)*(grX/goX);

			//calculate distance between GR and GO
			grGODist=grX;
			distTemp=abs(grPosX-goPosX);
			grGODist=(distTemp<grGODist)*distTemp+(distTemp>=grGODist)*grGODist;
			distTemp=(grX-goPosX)+grPosX;
			grGODist=(distTemp<grGODist)*distTemp+(distTemp>=grGODist)*grGODist;
			distTemp=(grX-grPosX)+goPosX;
			grGODist=(distTemp<grGODist)*distTemp+(distTemp>=grGODist)*grGODist;
			if(grGODist>1024)
			{
				statusOut<<"error in calculating gr to go distances: "<<grGODist<<endl;
				grGODist=1023;
			}

			//calculate time delay based distance
			delayGOMasksGR[j][i]=delayMasks[(int)grGODist/147+1];//[0]//256+1]; //todo: put consts as marcos
		}
	}

	statusOut<<"conduction delays assigned for parallel fibers"<<endl;
}

void InNet::updateMFActivties()
{
	for(int i=0; i<numMF; i++)
	{
		histMF[i]=histMF[i] || apMF[i];
		apMFH[i]=apMF[i];
		apBufMF[i]=(apBufMF[i]<<1)|(apMF[i]*0x00000001);
	}
}
void InNet::calcGOActivities()
{
	for(int i=0; i<numGO; i++)
	{
		gMFGO[i]=inputMFGO[i]*gMFIncGO+gMFGO[i]*gMFDecayGO;
		gGRGO[i]=grInputGOSumH[i]*gGRIncGO+gGRGO[i]*gGRDecayGO;

		gluGO[i]=gluGO[i]*gluDecayGO+grInputGOSumH[i]*gluScaleGO*exp(-1.5*gluGO[i]);
		mGluRGO[i]=mGluRGO[i]*mGluRDecayGO+gluGO[i]*mGluRScaleGO*exp(-mGluRGO[i]);
		gMGluRIncGO[i]=gMGluRIncGO[i]*gMGluRIncDecayGO+mGluRGO[i]*gMGluRIncScaleGO*exp(-gMGluRIncGO[i]);
		gMGluRGO[i]=gMGluRGO[i]*gMGluRDecayGO+gMGluRIncGO[i]*gMGluRScaleGO;
		threshGO[i]=threshGO[i]+(threshBaseGO-threshGO[i])*threshDecayGO;
		vGO[i]=vGO[i]+(gLeakGO*(eLeakGO-vGO[i]))+(gMGluRGO[i]*(eMGluRGO-vGO[i]))-(gMFGO[i]+gGRGO[i])*vGO[i];

		apGO[i]=vGO[i]>threshGO[i];
		apGOH[i]=apGO[i];
		apBufGO[i]=(apBufGO[i]<<1)|(apGO[i]*0x00000001);

		threshGO[i]=apGO[i]*threshMaxGO+(!apGO[i])*threshGO[i];
	}

	memset(inputMFGO, 0, numGO*sizeof(unsigned short));
}
void InNet::calcSCActivities()
{
	for(int i=0; i<numSC; i++)
	{
		gPFSC[i]=gPFSC[i]+(inputSumPFSCH[i]*pfIncSC);
		gPFSC[i]=gPFSC[i]*gPFDecaySC;

		vSC[i]=vSC[i]+(gLeakSC*(eLeakSC-vSC[i]))-gPFSC[i]*vSC[i];

		apSC[i]=vSC[i]>threshSC[i];
		apBufSC[i]=(apBufSC[i]<<1)|(apSC[i]*0x00000001);

		threshSC[i]=threshSC[i]+threshDecaySC*(threshBaseSC-threshSC[i]);
		threshSC[i]=apSC[i]*threshMaxSC+(!apSC[i])*(threshSC[i]);
	}
}
void InNet::updateMFtoGOOut()
{
	for(int i=0; i<numMF; i++)
	{
		if(apMF[i])
		{
			for(int j=0; j<numGOOutPerMF[i]; j++)
			{
				inputMFGO[mfConMFOutGO[i][j]]++;
			}
		}
	}
}
void InNet::resetMFHist(short t)
{
	if(t%histBinWidthGR==0)
	{
		for(int i=0; i<numMF; i++)
		{
			histMF[i]=false;
		}
	}
}
void InNet::runGRActivitiesCUDA(cudaStream_t &st)
{
	callGRActKernel(st, 4096, 256, vGRGPU, gKCaGRGPU, threshGRGPU, apBufGRGPU, outputGRGPU,
			gEGRSumGPU, gIGRSumGPU, gLeakGR, eLeakGR, eGOGR,
			threshBaseGR, threshMaxGR, threshDecayGR);
}
void InNet::runSumPFBCCUDA(cudaStream_t &st)
{
	callSumPFKernel<unsigned int, 512, true, false>
		(st, inputPFBCGPU, inputPFBCGPUP, inputSumPFBCGPU, 1, numBC, 1, numPFInPerBC);
}
void InNet::runSumPFSCCUDA(cudaStream_t &st)
{
	callSumPFKernel<unsigned int, 512, true, false>
		(st, inputPFSCGPU, inputPFSCGPUP, inputSumPFSCGPU, 1, numSC, 1, numPFInPerSC);
}
void InNet::runSumGRGOOutCUDA(cudaStream_t &st)
{
	callSumGRGOOutKernel<1024, 2, 512>
		(st, grInputGOGPU, grInputGOGPUP, grInputGOSumGPU);
}
void InNet::cpyAPMFHosttoGPUCUDA(cudaStream_t &st)
{
	cudaMemcpyAsync(apMFGPU, apMFH, numMF*sizeof(unsigned int), cudaMemcpyHostToDevice, st);
}
void InNet::cpyAPGOHosttoGPUCUDA(cudaStream_t &st)
{
	cudaMemcpyAsync(apGOGPU, apGOH, numGO*sizeof(unsigned int), cudaMemcpyHostToDevice, st);
}
void InNet::runUpdateMFInGRCUDA(cudaStream_t &st)
{
	callUpdateInGRKernel<numMF, 1024, 1024>(st, apMFGPU, gEGRGPU, gEGRGPUP,
			grConMFOutGRGPU, grConMFOutGRGPUP, numMFInPerGRGPU, gEGRSumGPU, gEDecayGR, gEIncGR);
}
void InNet::runUpdateGOInGRCUDA(cudaStream_t &st)
{
	callUpdateInGRKernel<numGO, 1024, 1024>(st, apGOGPU, gIGRGPU, gIGRGPUP,
			grConGOOutGRGPU, grConGOOutGRGPUP, numGOInPerGRGPU, gIGRSumGPU, gIDecayGR, gIIncGR);
}
void InNet::runUpdatePFBCSCOutCUDA(cudaStream_t &st)
{
	callUpdatePFBCSCOutKernel<numPFInPerBC, numPFInPerSC, 1024, 1024>(st, apBufGRGPU, delayBCPCSCMaskGRGPU,
			inputPFBCGPU, inputPFBCGPUP, inputPFSCGPU, inputPFSCGPUP);
}
void InNet::runUpdateGROutGOCUDA(cudaStream_t &st)
{
	callUpdateGROutGOKernel<numGO, 1024, 1024>(st, apBufGRGPU, grInputGOGPU, grInputGOGPUP,
			delayGOMasksGRGPU, delayGOMasksGRGPUP,
			grConGROutGOGPU, grConGROutGOGPUP, numGOOutPerGRGPU);
}
void InNet::cpyPFBCSumGPUtoHostCUDA(cudaStream_t &st)
{
	cudaMemcpyAsync(inputSumPFBCH, inputSumPFBCGPU, numBC*sizeof(unsigned int), cudaMemcpyDeviceToHost, st);
}
void InNet::cpyPFSCSumGPUtoHostCUDA(cudaStream_t &st)
{
	cudaMemcpyAsync(inputSumPFSCH, inputSumPFSCGPU, numSC*sizeof(unsigned int), cudaMemcpyDeviceToHost, st);
}
void InNet::cpyGRGOSumGPUtoHostCUDA(cudaStream_t &st)
{
	cudaMemcpyAsync(grInputGOSumH, grInputGOSumGPU, numGO*sizeof(unsigned int), cudaMemcpyDeviceToHost, st);
}
void InNet::runUpdateGRHistoryCUDA(cudaStream_t &st, short t)
{
	if(t%histBinWidthGR==0)
	{
		callUpdateGRHistKernel<2048, 512>(st, apBufGRGPU, historyGRGPU);
	}
}
