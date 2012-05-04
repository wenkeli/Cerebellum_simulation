/*
 * innetspatialvis.h
 *
 *  Created on: May 1, 2012
 *      Author: consciousness
 */

#ifndef INNETSPATIALVIS_H_
#define INNETSPATIALVIS_H_

#include <math.h>
#include <iostream>
#include <QtGui/QPixmap>
#include <QtGui/QPainter>
#include <QtCore/QString>
#include <QtGui/QColor>

#include "../datamodules/psh.h"

class InNetSpatialVis
{
public:
	InNetSpatialVis(PSHData *gr, PSHData *go);
	~InNetSpatialVis();

	QPixmap *paintSpatial(unsigned int binN);

private:
	InNetSpatialVis();

	PSHData *grPSH;
	PSHData *goPSH;

	QPixmap **bufs;

	unsigned int numTrials;
	unsigned int totalNumBins;
	unsigned int numGR;
	unsigned int numGO;
	unsigned int grPSHMaxVal;
	unsigned int goPSHMaxVal;
};



#endif /* INNETSPATIALVIS_H_ */
