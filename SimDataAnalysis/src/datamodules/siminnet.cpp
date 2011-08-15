/*
 * siminnet.cpp
 *
 *  Created on: Aug 15, 2011
 *      Author: consciousness
 */

#include "../../includes/datamodules/siminnet.h"

SimInNet::SimInNet(ifstream &infile)
{
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

	infile.read((char *)delayGOMasksGR, maxNumGOOutPerGR*numGR*sizeof(unsigned int));
	infile.read((char *)delayBCPCSCMaskGR, numGR*sizeof(unsigned int));
	infile.read((char *)numGOOutPerGR, numGR*sizeof(int));
	infile.read((char *)grConGROutGO, maxNumGOOutPerGR*numGR*sizeof(unsigned int));
	infile.read((char *)numGOInPerGR, numGR*sizeof(int));
	infile.read((char *)grConGOOutGR, maxNumInPerGR*numGR*sizeof(unsigned int));
	infile.read((char *)numMFInPerGR, numGR*sizeof(int));
	infile.read((char *)grConMFOutGR, maxNumInPerGR*numGR*sizeof(unsigned int));

	infile.read((char *)gPFSC, numSC*sizeof(float));
	infile.read((char *)threshSC, numSC*sizeof(float));
	infile.read((char *)vSC, numSC*sizeof(float));
	infile.read((char *)apSC, numSC*sizeof(bool));
	infile.read((char *)apBufSC, numSC*sizeof(unsigned int));
}
