/*
 * innetnomfgo.cpp
 *
 *  Created on: Aug 3, 2011
 *      Author: admin
 */

#include "../../includes/innetmodules/innetnomfgo.h"

InNetNoMFGO::InNetNoMFGO(const bool *actInMF):InNet(actInMF)
{
	gMFIncGO=0;
}

InNetNoMFGO::InNetNoMFGO(ifstream &infile, const bool *actInMF):InNet(infile, actInMF)
{
	gMFIncGO=0;
}

