/*
 * parameters.h
 *
 *  Created on: Jan 26, 2009
 *      Author: wen
 */

#ifndef PARAMETERS_H_
#define PARAMETERS_H_

//#of mossy fibers
#define NUMMF 1024

//#of climbing fibers
#define NUMCF 128

//#of granule cells
#define NUMGR 1048576

//#of golgi cells
#define NUMGO 1024

//#of purkinje cells
#define NUMPC 1024

//#of stellate cells
#define NUMSTELLATE 64

//#of basket cells
#define NUMBASKET 64

//#of deep nuclei cells
#define NUMNUC 32

//************
//defines divergence/convergence ratio of different cells and fibers
//#of synapses each mossy fiber makes with granule cells
#define MFGRSYNPERMF 4096
//#of synapses each mossy fiber makes with golgi cells
#define MFGOSYNPERMF 64
//#of glomeruli per mossy fiber
#define NUMGLPERMF 64

//#of synapses each golgi cell makes with granule cells
#define GOGRSYNPERGO 4096
//#of synapses each golgi cell receives from granule cells
#define GRGOSYNPERGO 2048

//#of synapses each granule cells makes with golgi cells
#define GRGOSYNPERGR 2

//#of dendrites per granule cell
#define DENPERGR 4

//#of mossy fiber synapses per golgi cell
#define MFDENPERGO 64

//************
//defines grid layout for different types of cells
//#of columns (width) of the granule cell grid
#define GRX 2048
//#of rows (height) of the granule cell grid
#define GRY 512

//#of columns (width) of the golgi cell grid
#define GOX 64
//#of rows (height) of the golgi cell grid
#define GOY 16

//#of columns of Glomerulus (for mossy fiber) grid
#define GLX 512
//#of rows of Glomerulus grind
#define GLY 128

//************
//distance specifications of fibers and dendrites of different cells
//the horizontal span of a granule cell dendrite
#define GRDENSPANX 64
//the vertical span of a granule cell dendrite
#define GRDENSPANY 64

//the horizontal span of a golgi cell dendrite
#define GODENSPANX 16
//the vertical span of a golgi cell dendrite
#define GODENSPANY 16

//the max horizontal distance a golgi cell can be from a granule cell to be able to make a GO->GR connection
#define GOFROMGRDENSPANX 1500
//the max vertical distance a golgi cell can be from a granule cell to be able to make a GO->GR connection
#define GOFROMGRDENSPANY 200

//the max horizontal distance a granule cell can be from a golgi cell to be able to make a GR->GO connection
#define GRFROMGODENSPANX 64
//the max vertical distance a granule cell can be from a golgi cell to be able to make a GR->GO connection
#define GRFROMGODENSPANY 64

//***********
//electrical properties specifications of different cells
//for granule cells
#define ELEAKGR -70

//for golgi cells
#define ELEAKGO -70

#endif /* PARAMETERS_H_ */
