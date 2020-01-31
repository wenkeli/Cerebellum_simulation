/*
 * readin.cpp
 *
 *  Created on: Mar 29, 2011
 *      Author: consciousness
 */

#include "../includes/readin.h"

void readSimIn()
{
	simIn.seekg(0, ios_base::beg);

	accessConnLock.lock();
	simIn.read((char *)numMFtoGRN, (NUMMF+1)*sizeof(short));
	simIn.read((char *)numMFtoGON, (NUMMF+1)*sizeof(char));
	simIn.read((char *)numGOtoGRN, (NUMGO+1)*sizeof(short));
	simIn.read((char *)numSynGRtoGO, (NUMGR+1)*sizeof(char));

	simIn.read((char *)conMFtoGRN, (NUMMF+1)*NUMGRPERMFN*sizeof(int));
	simIn.read((char *)conMFtoGON, (NUMMF+1)*MFGOSYNPERMF*sizeof(short));
	simIn.read((char *)conGOtoGRN, (NUMGO+1)*NUMGROUTPERGON*sizeof(int));
	simIn.read((char *)conGRtoGO, (NUMGR+1)*GRGOSYNPERGR*sizeof(short));
	simIn.read((char *)conBCtoPC, NUMBC*BCPCSYNPERBC*sizeof(char));
	simIn.read((char *)conIOCouple, NUMIO*IOCOUPSYNPERIO*sizeof(char));
	simIn.read((char *)conPCtoNC, NUMPC*PCNCSYNPERPC*sizeof(char));

	simIn.read((char *)typeMFs, (NUMMF+1)*sizeof(char));
	simIn.read((char *)bgFreqContsMF, NUMCONTEXTS*(NUMMF+1)*sizeof(float));
	simIn.read((char *)incFreqMF, (NUMMF+1)*sizeof(float));
	simIn.read((char *)csStartMF, (NUMMF+1)*sizeof(short));
	simIn.read((char *)csEndMF, (NUMMF+1)*sizeof(short));
	accessConnLock.unlock();

	pfSynWeightPCLock.lock();
	simIn.read((char *)pfSynWeightPC, NUMGR*sizeof(float));
	pfSynWeightPCLock.unlock();
}
