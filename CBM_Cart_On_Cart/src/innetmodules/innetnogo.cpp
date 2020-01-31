/*
 * innetnogo.cpp
 *
 *  Created on: Aug 2, 2011
 *      Author: consciousness
 */

#include "../../includes/innetmodules/innetnogo.h"

InNetNoGO::InNetNoGO(const bool *actInMF):InNet(actInMF)
{
	gMFIncGO=0;
	gGRIncGO=0;
}

InNetNoGO::InNetNoGO(ifstream &infile, const bool *actInMF):InNet(infile, actInMF)
{
	gMFIncGO=0;
	gGRIncGO=0;
}
