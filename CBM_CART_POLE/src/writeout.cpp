/*
 * writeout.cpp
 *
 *  Created on: Mar 29, 2011
 *      Author: consciousness
 */

#include "../includes/writeout.h"

void writeSimOut()
{
	cout<<"exporting simulation state... ";
	cout.flush();
	simOut.seekp(0, ios_base::beg);

	accessConnLock.lock();
	simOut.write((char *)numMFtoGRN, (NUMMF+1)*sizeof(short));
	simOut.write((char *)numMFtoGON, (NUMMF+1)*sizeof(char));
	simOut.write((char *)numGOtoGRN, (NUMGO+1)*sizeof(short));
	simOut.write((char *)numSynGRtoGO, (NUMGR+1)*sizeof(char));

	simOut.write((char *)conMFtoGRN, (NUMMF+1)*NUMGRPERMFN*sizeof(int));
	simOut.write((char *)conMFtoGON, (NUMMF+1)*MFGOSYNPERMF*sizeof(short));
	simOut.write((char *)conGOtoGRN, (NUMGO+1)*NUMGROUTPERGON*sizeof(int));
	simOut.write((char *)conGRtoGO, (NUMGR+1)*GRGOSYNPERGR*sizeof(short));
	simOut.write((char *)conBCtoPC, NUMBC*BCPCSYNPERBC*sizeof(char));
	simOut.write((char *)conIOCouple, NUMIO*IOCOUPSYNPERIO*sizeof(char));
	simOut.write((char *)conPCtoNC, NUMPC*PCNCSYNPERPC*sizeof(char));

	simOut.write((char *)typeMFs, (NUMMF+1)*sizeof(char));
	simOut.write((char *)bgFreqContsMF, NUMCONTEXTS*(NUMMF+1)*sizeof(float));
	simOut.write((char *)incFreqMF, (NUMMF+1)*sizeof(float));
	simOut.write((char *)csStartMF, (NUMMF+1)*sizeof(short));
	simOut.write((char *)csEndMF, (NUMMF+1)*sizeof(short));
	accessConnLock.unlock();

	pfSynWeightPCLock.lock();
	simOut.write((char *)pfSynWeightPC, NUMGR*sizeof(float));
	pfSynWeightPCLock.unlock();

	simOut.flush();

	cout<<"done"<<endl;
}

void writePSHOut()
{
	cout<<"exporting PSH... ";
	cout.flush();
	pshOut.seekp(0, ios_base::beg);
	accessPSHLock.lock();
	for(int binN=0; binN<PSHNUMBINS; binN++)
	{
		for(int j=0; j<NUMGR; j++)
		{
			pshGRMax=(pshGR[binN][j]>pshGRMax)*pshGR[binN][j]+(!(pshGR[binN][j]>pshGRMax))*pshGRMax;
		}
		for(int j=0; j<NUMGO; j++)
		{
			pshGOMax=(pshGO[binN][j]>pshGOMax)*pshGO[binN][j]+(!(pshGO[binN][j]>pshGOMax))*pshGOMax;
		}
		for(int j=0; j<NUMMF; j++)
		{
			pshMFMax=(pshMF[binN][j]>pshMFMax)*pshMF[binN][j]+(!(pshMF[binN][j]>pshMFMax))*pshMFMax;
		}
		for(int j=0; j<NUMPC; j++)
		{
			pshPCMax=(pshPC[binN][j]>pshPCMax)*pshPC[binN][j]+(!(pshPC[binN][j]>pshPCMax))*pshPCMax;
		}
		for(int j=0; j<NUMBC; j++)
		{
			pshBCMax=(pshBC[binN][j]>pshBCMax)*pshBC[binN][j]+(!(pshBC[binN][j]>pshBCMax))*pshBCMax;
		}
		for(int j=0; j<NUMSC; j++)
		{
			pshSCMax=(pshSC[binN][j]>pshSCMax)*pshSC[binN][j]+(!(pshSC[binN][j]>pshSCMax))*pshSCMax;
		}
	}

	pshOut.write((char *)&numTrials, sizeof(unsigned int));
	pshOut.write((char *)pshMF, PSHNUMBINS*NUMMF*sizeof(unsigned short));
	pshOut.write((char *)&pshMFMax, sizeof(unsigned short));

	pshOut.write((char *)pshGO, PSHNUMBINS*NUMGO*sizeof(unsigned short));
	pshOut.write((char *)&pshGOMax, sizeof(unsigned short));

	pshOut.write((char *)pshGR, PSHNUMBINS*NUMGR*sizeof(unsigned short));
	pshOut.write((char *)&pshGRMax, sizeof(unsigned short));

	pshOut.write((char *)pshPC, PSHNUMBINS*NUMPC*sizeof(unsigned int));
	pshOut.write((char *)&pshPCMax, sizeof(unsigned int));

	pshOut.write((char *)pshBC, PSHNUMBINS*NUMBC*sizeof(unsigned int));
	pshOut.write((char *)&pshBCMax, sizeof(unsigned int));

	pshOut.write((char *)pshSC, PSHNUMBINS*NUMSC*sizeof(unsigned int));
	pshOut.write((char *)&pshSCMax, sizeof(unsigned int));
	accessPSHLock.unlock();
	pshOut.flush();

	cout<<"done!"<<endl;
}
