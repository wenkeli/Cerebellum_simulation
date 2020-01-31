/*
 * innetnogrgo.cpp
 *
 *  Created on: Aug 3, 2011
 *      Author: consciousness
 */

#include "../../includes/innetmodules/innetnogrgo.h"

InNetNoGRGO::InNetNoGRGO(const bool *actInMF):InNet(actInMF)
{
	gGRIncGO=0;
	gMFIncGO=gMFIncGO*12.5;
}

InNetNoGRGO::InNetNoGRGO(ifstream &infile, const bool *actInMF):InNet(infile, actInMF)
{
	gGRIncGO=0;
	gMFIncGO=gMFIncGO*12.5;
}
