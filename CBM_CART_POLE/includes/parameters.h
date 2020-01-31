/*
 * parameters.h
 *
 *  Created on: Jan 26, 2009
 *      Author: wen
 *
 * This file contains all global constants
 */

#ifndef PARAMETERS_H_
#define PARAMETERS_H_

//Cart pole definitions
#define NUMPOLANGLEMF 30
#define NUMPOLVELMF 30
#define NUMCARTPOSMF 30
#define NUMCARTVELMF 30

#define MIN_POLE_ANGLE -12.0
#define MAX_POLE_ANGLE 0.0
#define MIN_POLE_VELOCITY -12.0
#define MAX_POLE_VELOCITY 12.0
#define MIN_CART_POS -100.0
#define MAX_CART_POS 100.0
// These velocities are somewhat arbitrarily bounded... Technically
// there should be no bounds, but realistically we need to have some bounds.
#define MIN_CART_VELOCITY -10.0
#define MAX_CART_VELOCITY 10.0

//end cart pole defs

//defines numubers of different cells
//-------------------------------------
//#of mossy fibers
#define NUMMF 1024

//new #of glomeruli
#define NUMGLN 65536//262144

//#of climbing fibers
#define NUMIO 4

//#of granule cells
#define NUMGR 1048576
#define NUMGRPAD 0 //24

//#of golgi cells
#define NUMGO 1024

//#of purkinje cells
#define NUMPC 32

//#of stellate cells
#define NUMSC 512

//#of basket cells
#define NUMBC 128

//#of deep nuclei cells
#define NUMNC 8
//-----------------------------------
//end definition for cell numbers

//************ defines connectivity ratio of different cells and fibers
//=====================================================================
//---Mossy fibers
//#of synapses each mossy fiber makes with granule cells
#define MFGRSYNPERMF 4096
//#of synapses each mossy fiber makes with golgi cells
#define MFGOSYNPERMF 64
//#of glomeruli per mossy fiber
#define NUMGLPERMF 64
//#of nucleus cells each mossy fiber connects with
#define MFNCSYNPERMF 1

//new connectivity
#define NUMGLPERMFN 64//256
#define NUMGRPERMFN 5120
//---

//new -- glomerulus connectivity ratio
#define MAXNGRDENPERGLN 80//20
#define NORMNGRDENPERGLN 64//16

//---Golgi cells
//#of synapses each golgi cell makes with granule cells
#define GOGRSYNPERGO 4096
//#of synapses each golgi cell receives from granule cells
#define GRGOSYNPERGO 2048
//#of mossy fiber synapses per golgi cell
#define MFDENPERGO 64

//new connectivity
#define NUMGLDENPERGON 16//64

#define NUMGLAXPERGON 48//192//256
#define NUMGROUTPERGON 3840
//---

//---granule cells
//#of synapses each granule cell makes with golgi cells
#define GRGOSYNPERGR 2
//#of synapses each granule cell makes with purkinje cells
#define GRPCSYNPERGR 1
//#of synapses each granule cell makes with basket cells
#define GRBCSYNPERGR 1
//#of synapses each granule cell makes with stellate cells
#define GRSCSYNPERGR 1
//#of dendrites per granule cell
#define DENPERGR 4
//---

//---purkinje cells
//#of parallel fibers synapses per purkinje cell
#define PFPCSYNPERPC 32768
//#of basket cell synapses per purkinje cell
#define BCPCSYNPERPC 16
//#of stellate cell synapses per purkinkje cell
#define SCPCSYNPERPC 16
//#of purkinje to basket cell synaspses per purkinje cell
#define PCBCSYNPERPC 16
//#of basket cells per purkinje cell
#define NUMBCPERPC 4
//#of nucleus cells per purkinje cell
#define PCNCSYNPERPC 3
//---

//---basket cells
//#of parallel fibers per basket cell
#define PFBCSYNPERBC 8192
//#of purkinje cell synapses per basket cell
#define PCBCSYNPERBC 4
//#of basket to purkinje cell synaspses for basket cell
#define BCPCSYNPERBC 4
//---

//---stellate cells
//#of parallel fibers per stellate cell
#define PFSCSYNPERSC 2048
//#of stellate cell to purkinje cell synapses per stellate cell
#define SCPCSYNPERSC 1
//---

//---inferior olivary cells
//#of IO coupling synapses per IO cell
#define IOCOUPSYNPERIO 1
//#of deep nucleus cell synapses per IO cell
#define NCIOSYNPERIO 8
//---

//---nucleus cells
//#of mossy fiber synapses per nucleus cell
#define MFNCSYNPERNC 128
//#of purkinje cell synapses per nucleus cell
#define PCNCSYNPERNC 12
//---

////defines ratios for new connectivity scheme
////number of granule dendrites connected to a glomerulus
//#define GRDENPERGLN 16
////max possible number of granule dendrites connected to a glomerulus
//#define MAXGRDENPERGLN 20
//
////number of glomeruli that is assigned to a mossy fiber
//#define NUMGLPERMFN 256
//
////number of glomeruli that is assigned to a golgi cell
//#define NUMGLPERGON 256
//==========================================================================
//**************end connectivity ratios

//defines grid layout for different types of cells
//------------------------------------------------
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

//define grid layout for new connectivity scheme
#define GLXN 512//1024
#define GLYN 128//256
//-------------------------------------------------

//distance specifications of fibers and dendrites of different cells
//----------------------------------------------------------------
//the horizontal span of a granule cell dendrite
#define GRDENSPANX 64
//the vertical span of a granule cell dendrite
#define GRDENSPANY 64

//the horizontal span of a golgi cell dendrite in golmeruli grid coordinates
#define GODENSPANX 16
//the vertical span of a golgi cell dendrite in glomeruli grid coordinates
#define GODENSPANY 16

//the max horizontal distance a golgi cell can be from a granule cell to be able to make a GR->GO connection
#define GOFROMGRDENSPANX 1170//1500//2048//1920//2048//2048//1500
//the max vertical distance a golgi cell can be from a granule cell to be able to make a GR->GO connection
#define GOFROMGRDENSPANY 48//32//40//64//128//64//128//64//96//128//50//400//200

//the max horizontal distance a granule cell can be from a golgi cell to be able to make a GO->GR connection
#define GRFROMGODENSPANX 400//96//128//50//400//64//256
//the max vertical distance a granule cell can be from a golgi cell to be able to make a GO->GR connection
#define GRFROMGODENSPANY 400//96//128//50//400//64//256

//new connectivity
#define GRDENSPANGLXN 6//8//32
#define GRDENSPANGLYN 6//8//32

#define GODENSPANGLXN 12//24//48
#define GODENSPANGLYN 12//24//48

#define GOAXSPANGLXN 8//10//16//32//48
#define GOAXSPANGLYN 8//10//16//32//48
//-----------------------------------------------------------------


//define parameters for CUDA
//----------------------------
#define CUDAGRNUMTBLOCK 4096//2048
#define CUDAGRNUMTHREAD 256//512
#define CUDAGRIONUMTBLOCK 1024
#define CUDAGRIONUMTHREAD 256
//---------------------------

//define parameters necessary for PF plasticity
//----------------------------
#define NUMHISTBINSGR 40
#define HISTBINWIDTHGR 5 //width of each bin in ms
#define PFLTDTIMERSTARTIO -100
#define PFPCLTPINCPF 0.00008f
#define PFPCLTDDECPF -0.00008f
//-----------------------------

//define parameters necessary for MF to NC plasticity
//-------------------------
#define HISTBINWIDTHPC 5
#define NUMHISTBINSPC 8
#define MFNCLTDTHRESH 12
#define MFNCLTPTHRESH 4
#define MFNCLTDDECNC -0.0000025f
#define MFNCLTPINCNC 0.0002f
//-------------------------


//electrical properties specifications of different cells
//=======================================================
//for granule cells
#define ELEAKGR -70.0f
#define EGOGR -80.0f
#define EMFGR (float)0
//test defines for granule cells
#define THRESHMAXGR -20.0f
#define GICONSTINCGR 0.007f//0.009f//0.010f//0.007f//0.013f//0.022f//0.044f//0.022f//0.00275//0.0055f//0.011f
#define GEDECAYGR 0.9819825f
#define GIDECAYGR 0.9801987f
#define THRESHDECAYGR 0.2834687f
#define GLEAKGR 0.02f

//for golgi cells
#define ELEAKGO -70.0f
#define EMGLURGO -96.0f

//for purkinje cells
#define ELEAKPC -60.0f
#define EBCPC -80.0f
#define ESCPC -80.0f
#define THRESHMAXPC -48.0f
#define THRESHBASEPC -60.0f
#define THRESHDECAYTPC 5.0f
#define THRESHDECAYPC 0.1812692f
#define GPFDECAYTPC 4.15f
#define GPFDECAYPC 0.7858700f
#define GBCDECAYTPC 5.0f //4.15f
#define GBCDECAYPC 0.8187308f //0.7858700f
#define GSCDECAYTPC 4.15f
#define GSCDECAYPC 0.7858700f
#define GSCINCCONSTPC 0.002f//0.001f//0.05f//0.03f//0.068f //0.1088f
#define GPFSCALECONSTPC 0.0018f //0.0007f//0.0005f//0.0001068f //0.007f //0.0054f 0.002f//
#define GBCSCALECONSTPC 0.02f//0.07f//0.05f //0.09f //0.01080f
//#define GSCSCALECONSTPC 0.1088f
#define GLEAKPC 0.04f
#define PFSYNWINITNC 0.5f

//for basket cells
#define ELEAKBC -70.0f
#define EPCBC -70.0f
#define THRESHMAXBC (float)0
#define THRESHBASEBC -50.0f
#define GLEAKBC 0.04f
#define GPFDECAYTBC 4.15f
#define GPFDECAYBC 0.7858700f
#define GPCDECAYTBC 5.0f
#define GPCDECAYBC 0.8187308f
#define THRESHDECAYTBC 10.0f
#define THRESHDECAYBC 0.0951625f
#define PFINCCONSTBC 0.0081f//0.007f//0.0007f//0.0002f//0.0000879f//0.0002197f//0.0000879f //0.0004394f//0.0008789f //0.06f
#define PCINCCONSTBC 0.0388f//0.25f

//for stellate cells
#define ELEAKSC -60.0f
#define THRESHMAXSC (float)0
#define THRESHBASESC -50.0f
#define GLEAKSC 0.04f
#define GPFDECAYTSC 4.15f
#define GPFDECAYSC 0.7858700f
#define THRESHDECAYTSC 22.0f
#define THRESHDECAYSC 0.0444370f
#define PFINCCONSTSC 0.012f//0.008f//0.0002871f//0.0004178f//0.00028711f//0.0014356f //0.0028711f //0.1176f

//for inferior olivary cells
#define ELEAKIO -60.0f//-70.0f for oscillatory model
#define EHIO -30.0f
#define ECAIO 50.0f
#define EKCAIO -80.0f
#define ENCIO -80.0f
#define GLEAKIO 0.03f
#define CADECAYIO 0.96f
#define GLTCAHTIO 51.0f
#define GLTCAHMAXVIO 87.0f
#define GLTCAMMAXVIO -56.0f
#define GHMAXVIO 78.0f
#define GHTAUVIO 69.0f
#define GNCDECTSIO 50.0f
#define GNCDECTTIO 70.0f
#define GNCDECT0IO 56.0f
#define GNCINCSCALEIO 3.0f
#define GNCINCTIO 300.0f
#define COUPLESCALEIO 0.04f
#define THRESHBASEIO -61.0f//-63.0f for oscillatory model
#define THRESHMAXIO 10.0f//-20.0f for oscillatory model
#define THRESHTAUIO 122.0f //for non-oscillatory model
#define THRESHDECAYIO 0.0081632f //for non-oscillatory model

//for Nucleus cells
#define ELEAKNC -65.0f
#define EPCNC -80.0f
#define MFNMDADECAYTNC 50.0f
#define MFNMDADECAYNC 0.9801987f
#define MFAMPADECAYTNC 6.0f
#define MFAMPADECAYNC 0.8464817f
#define GMFNMDAINCNC 0.2834687f
#define GMFAMPAINCNC 0.2834687f
#define GPCSCALEAVGNC 0.2f//0.13f//0.15f
#define GPCDECAYTNC 4.15f
#define GPCDECAYNC 0.7858700f
#define GLEAKNC 0.02f
#define THRESHDECAYTNC 5.0f
#define THRESHDECAYNC 0.1812692f
#define THRESHMAXNC -40.0f
#define THRESHBASENC -72.0f
#define IORELPDECTSNC 40.0f
#define IORELPDECTTNC 1.0f
#define IORELPDECT0NC 78.0f
#define IORELPINCSCALENC 0.25f
#define IORELPINCTNC 0.8f
#define MFSYNWINITNC 0.005f
//===============================================
//end electrical properties

//mossy fiber properties
//==================================================
//assuming non-overlapping mossy fiber activity for each CS
//and context this simulates very different CS's, since we
//don't know much about the activation pattern for similar CS's.

//Number of contexts considered in the simulation
#define NUMCONTEXTS 2

//mossy fiber types
//enum MFType{MFBG, MFCS1TON, MFCS2TON, MFCS3TON, MFCS4TON, MFCS1PHA, MFCS2PHA, MFCS3PHA, MFCS4PHA, MFCONT1, MFCONT2};
#define MFBG 0
#define MFCS1TON 1
#define MFCS2TON 2
#define MFCS3TON 3
#define MFCS4TON 4
#define MFCS1PHA 5
#define MFCS2PHA 6
#define MFCS3PHA 7
#define MFCS4PHA 8
#define MFCONT1 9
#define MFCONT2 10

//proportions of mossy fibers that are in each context
//context is the overall background environment that is
//not related to the stimulus. The MFs that are in this
//category do not change their response to CSs.
#define MFPROPCONTEXT1 0.03
#define MFPROPCONTEXT2 0//0.03

//mossy fiber that respond to one CS do not respond to another
//for CS1, the proportions of MFs that are phasic and tonic
#define MFPROPPHASIC1 0.03//0.004//0.007//0.015 //0.03 //0.06
#define MFPROPTONIC1 0.02//0.003//0.01//0.01 //0.02//0.05//0.02 //0.04

//for CS2
#define MFPROPPHASIC2 0
#define MFPROPTONIC2 0.05

//for CS3
#define MFPROPPHASIC3 0
#define MFPROPTONIC3 0.03

//for CS4
#define MFPROPPHASIC4 0
#define MFPROPTONIC4 0.03
//----------------------

//the duration which the phasic fibers increase their activity
//in response to a CS
#define MFPHASICDUR 40

//background frequency range for the MFs that respond to CSs
#define MFCSBGNDFREQMIN 1
#define MFCSBGNDFREQMAX 5

//frequency range for the MFs that respond to context1
#define MFCONTFREQMIN1 30
#define MFCONTFREQMAX1 60

//frequency range for the MFs that respond to context2
#define MFCONTFREQMIN2 30
#define MFCONTFREQMAX2 60

//background frequency range for all other MFs that do not change
//their response for a context or CS
#define MFBGNDFREQMIN 1
#define MFBGNDFREQMAX 10

//frequency increase of tonic fibers when responding to a CS
#define MFTONICFREQINC1 80 //CS1
#define MFTONICFREQINC2 40 //CS2
#define MFTONICFREQINC3 80 //CS3
#define MFTONICFREQINC4 80 //CS4

//frequency increase of phasic fibers when responding to a CS
#define MFPHASFREQINC1 120 //CS1
#define MFPHASFREQINC2 120 //CS2
#define MFPHASFREQINC3 120 //CS3
#define MFPHASFREQINC4 120 //CS4
//========================================================
//end mossy fiber properties

//US parameters
//----------------------
#define USONSET 2500
#define MAXUSDR 30//10//80
//----------------------

//analysis parameters
#define PSHNUMBINS 240//200
#define PSHBINWIDTH 5

//display types, for visualization
enum ConnDispT{MFGR, MFGO, GRGO, GOGR};

#define TIMESTEP (float)1
#define TRIALTIME 5000

#endif /* PARAMETERS_H_ */
