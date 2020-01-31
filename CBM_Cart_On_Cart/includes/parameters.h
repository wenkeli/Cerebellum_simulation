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

//simulation setup
//number of microzones
#ifdef EYELID
#define NUMMZONES 1
#endif
#ifdef CARTPOLE
#define NUMMZONES 2
#endif

//analysis parameters
//PSH parameters

#define PSHPRESTIMNUMBINS 40
#define PSHPOSTSTIMNUMBINS 20
#define PSHSTIMNUMBINS 200
#define PSHNUMBINS 260
#define PSHBINWIDTH 5
#define APBUFWIDTH 32
#define NUMBINSINAPBUF 5


//defines numbers of different cells
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
//#of nucleus cells each mossy fiber connects with
#define NUMNCOUTPERMF 1

//new connectivity
#define NUMGLOUTPERMF 64//256
#define NUMGOOUTPERMF 64
#define NUMGROUTPERMF 5120
//---

//new -- glomerulus connectivity ratio
#define MAXNGRDENPERGL 80//20
#define NORMNGRDENPERGL 64//16

//---Golgi cells
//#of synapses each golgi cell receives from granule cells
#define NUMGRINPERGO 2048

//new connectivity
#define NUMGLINPERGO 16//64
#define NUMMFINPERGO 16

#define NUMGLOUTPERGO 48//192//256
#define NUMGROUTPERGO 3840
//---

//---granule cells
//#of synapses each granule cell makes with golgi cells
#define NUMGOOUTPERGR 2
//#of synapses each granule cell makes with purkinje cells
#define NUMPCOUTPERGR 1
//#of synapses each granule cell makes with basket cells
#define NUMBCOUTPERGR 1
//#of synapses each granule cell makes with stellate cells
#define NUMSCOUTPERGR 1
//#of dendrites per granule cell
#define NUMINPERGR 4
//---

//---purkinje cells
//#of parallel fibers synapses per purkinje cell
#define NUMPFINPERPC 32768
//#of basket cell synapses per purkinje cell
#define NUMBCINPERPC 16
//#of stellate cell synapses per purkinkje cell
#define NUMSCINPERPC 16
//#of purkinje to basket cell synaspses per purkinje cell
#define NUMBCOUTPERPC 16
//#of basket cells per purkinje cell
#define BCTOPCRATIO 4
//#of nucleus cells per purkinje cell
#define NUMNCOUTPERPC 3
//---

//---basket cells
//#of parallel fibers per basket cell
#define NUMPFINPERBC 8192
//#of purkinje cell synapses per basket cell
#define NUMPCINPERBC 4
//#of basket to purkinje cell synaspses for basket cell
#define NUMPCOUTPERBC 4
//---

//---stellate cells
//#of parallel fibers per stellate cell
#define NUMPFINPERSC 2048
//#of stellate cell to purkinje cell synapses per stellate cell
#define NUMPCOUTPERSC 1
//---

//---inferior olivary cells
//#of IO coupling synapses per IO cell
#define IOCOUPSYNPERIO 1
//#of deep nucleus cell synapses per IO cell
#define NUMNCINPERIO 8
//---

//---nucleus cells
//#of mossy fiber synapses per nucleus cell
#define NUMMFINPERNC 128
//#of purkinje cell synapses per nucleus cell
#define NUMPCINPERNC 12
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

//define grid layout for new connectivity scheme
#define GLX 512//1024
#define GLY 128//256
//-------------------------------------------------

//distance specifications of fibers and dendrites of different cells
//----------------------------------------------------------------


//the max horizontal distance a golgi cell can be from a granule cell to be able to make a GR->GO connection
#define GOFROMGRDENSPANX 2048//1170//1500//2048//1920//2048//2048//1500
//the max vertical distance a golgi cell can be from a granule cell to be able to make a GR->GO connection
#define GOFROMGRDENSPANY 192//48//32//40//64//128//64//128//64//96//128//50//400//200

//new connectivity
#define GRDENSPANGLX 8//8//32
#define GRDENSPANGLY 8//8//32

#define GODENSPANGLX 32//12//24//48
#define GODENSPANGLY 32//12//24//48

#define GOAXSPANGLX 32//8//10//16//32//48
#define GOAXSPANGLY 32//8//10//16//32//48
//-----------------------------------------------------------------


//define parameters for CUDA
//----------------------------
#define CUDANUMSTREAMS 8

#define CUDAGRNUMTBLOCK 8192//16384//2048
#define CUDAGRNUMTHREAD 128//64//512
#define CUDAGRIONUMTBLOCK 256//512//1024
#define CUDAGRIONUMTHREAD 1024//512//256
//---------------------------

//define parameters necessary for PF plasticity
//----------------------------
#define NUMHISTBINSGR 40
#define HISTBINWIDTHGR 5 //width of each bin in ms
#define PFLTDTIMERSTARTIO -100
#define PFPCLTPINCPF 0.0001f//0.000008f
#define PFPCLTDDECPF (-0.001f)//(-0.0009f)//(-0.00008f)
//-----------------------------

//define parameters necessary for MF to NC plasticity
//-------------------------
#define HISTBINWIDTHPC 5
#define NUMHISTBINSPC 8
#define MFNCLTDTHRESH 12//11//12
#define MFNCLTPTHRESH 2//4
#define MFNCLTDDECNC (-0.0000025f)
#define MFNCLTPINCNC 0.0002f
//-------------------------


//electrical properties specifications of different cells
//=======================================================

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

#define GNCDECTSIO 0.5f//50.0f
#define GNCDECTTIO 70.0f
#define GNCDECT0IO 0.56f//56.0f
#define GNCINCSCALEIO 0.003f//0.003f//3.0f
#define GNCINCTIO 300.0f

#define COUPLESCALEIO 0.04f

#define THRESHBASEIO -61.0f//-63.0f for oscillatory model
#define THRESHMAXIO 10.0f//-20.0f for oscillatory model
#define THRESHTAUIO 122.0f //for non-oscillatory model
#define THRESHDECAYIO 0.0081632f //=1-exp(-TIMESTEP/THRESHTAUIO) - for non-oscillatory model


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
#define MFPROPPHASIC1 0.03//0//0.03//0.004//0.007//0.015 //0.03 //0.06
#define MFPROPTONIC1 0.02//0.03//0.02//0.003//0.01//0.01 //0.02//0.05//0.02 //0.04

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
#define USONSET 2500//2200//2500//2000//2200//1900
#define MAXUSDR 30//10//80
//----------------------

//display types, for visualization
enum ConnDispT{MFGR, MFGO, GRGO, GOGR};

#define TIMESTEP ((float)1)
#define TSUNITINS 0.001f
#define TRIALTIME 5000

#endif /* PARAMETERS_H_ */
