/*
 * synapsegenesis.cpp
 *
 * creates the connections
 *
 *  Created on: Jan 27, 2009
 *      Author: wen
 */

#include "../includes/synapsegenesis.h"

int glomeruli[GLX*GLY];
CRandomSFMT0 randGenSFMT(time(NULL));

struct GlomerulusN
{
	bool hasGODen;
	bool hasGOAx;
	bool hasMF;
	short goDenInd;
	short goAxInd;
	short mfInd;
	int grDenInds[MAXNGRDENPERGLN];
	char numGRDen;
};

GlomerulusN glomeruliN[NUMGLN];

int conGRtoGL[NUMGR][DENPERGR];
int conGODentoGL[NUMGO][NUMGLDENPERGON];
int conGOAxtoGL[NUMGO][NUMGLAXPERGON];
int conMFtoGL[NUMMF][NUMGLPERMFN];

#ifdef DEBUG
	unsigned short gOtoGRConGR[NUMGR][DENPERGR];
	unsigned short gOtoGRConGRSN[NUMGR];
#endif

/*
 * new connectivity, initialize glomeruli to default values
 */
void initializeVars()
{
	extern GlomerulusN glomeruliN[NUMGLN];
	extern int conGRtoGL[NUMGR][DENPERGR];
	extern int conGODentoGL[NUMGO][NUMGLDENPERGON];
	extern int conGOAxtoGL[NUMGO][NUMGLAXPERGON];
	extern int conMFtoGL[NUMMF][NUMGLPERMFN];

	for(int i=0; i<NUMGLN; i++)
	{
		glomeruliN[i].hasGODen=false;
		glomeruliN[i].hasGOAx=false;
		glomeruliN[i].hasMF=false;
		glomeruliN[i].goDenInd=NUMGO;
		glomeruliN[i].goAxInd=NUMGO;
		glomeruliN[i].mfInd=NUMMF;
		glomeruliN[i].numGRDen=0;

		for(int j=0; j<MAXNGRDENPERGLN; j++)
		{
			glomeruliN[i].grDenInds[j]=NUMGR;
		}
	}

	for(int i=0; i<NUMGR; i++)
	{
		for(int j=0; j<DENPERGR; j++)
		{
			conGRtoGL[i][j]=NUMGLN;
		}
	}

	for(int i=0; i<NUMGO; i++)
	{
		for(int j=0; j<NUMGLDENPERGON; j++)
		{
			conGODentoGL[i][j]=NUMGLN;
		}
		for(int j=0; j<NUMGLAXPERGON; j++)
		{
			conGOAxtoGL[i][j]=NUMGLN;
		}
	}

	for(int i=0; i<NUMMF; i++)
	{
		for(int j=0; j<NUMGLPERMFN; j++)
		{
			conMFtoGL[i][j]=NUMGLN;
		}
	}
}

/*
 * new connectivity, assign granule cell dendrites to glomeruli
 */
void assignGRDenGLN(stringstream &output)
{
	extern CRandomSFMT0 randGenSFMT;
	extern GlomerulusN glomeruliN[NUMGLN];
	extern int conGRtoGL[NUMGR][DENPERGR];
	float scaleGRGLX;
	float scaleGRGLY;

	scaleGRGLX=(float)GRX/GLXN;
	scaleGRGLY=(float)GRY/GLYN;

	for(char i=0; i<DENPERGR; i++)
	{
		int numConnectedGR;
		bool  grConnected[NUMGR];

		numConnectedGR=0;
		memset(grConnected, false, NUMGR*sizeof(bool));

		while(numConnectedGR<NUMGR)
		{
			int grInd;
			int grPosX;
			int grPosY;
			int tempGRDenSpanGLX;
			int tempGRDenSpanGLY;
			int numGRPerGLLim;
			int attempts;
			bool complete;

			grInd=randGenSFMT.IRandom(0, NUMGR-1);

			if(grConnected[grInd])
			{
				continue;
			}

			grConnected[grInd]=true;
			numConnectedGR++;

			grPosX=grInd%GRX;
			grPosY=(int)(grInd/GRX);
			tempGRDenSpanGLX=GRDENSPANGLXN;
			tempGRDenSpanGLY=GRDENSPANGLYN;

			numGRPerGLLim=NORMNGRDENPERGLN;

			complete=false;
			for(attempts=0; attempts<60000; attempts++)
			{
				int tempGLPosX, tempGLPosY;
				int derivedGLIndex;
				bool unique;

				if(attempts==2000)
				{
					tempGRDenSpanGLX=tempGRDenSpanGLX*2;
					tempGRDenSpanGLY=tempGRDenSpanGLY*2;
				}
				if(attempts==20000)
				{
					numGRPerGLLim=MAXNGRDENPERGLN;
				}

				tempGLPosX=(int)(grPosX/scaleGRGLX);
				tempGLPosY=(int)(grPosY/scaleGRGLY);

				tempGLPosX+=randGenSFMT.IRandom(-tempGRDenSpanGLX/2, tempGRDenSpanGLX/2);
				tempGLPosY+=randGenSFMT.IRandom(-tempGRDenSpanGLY/2, tempGRDenSpanGLY/2);

				tempGLPosX=(tempGLPosX%GLXN+GLXN)%GLXN;
				tempGLPosY=(tempGLPosY%GLYN+GLYN)%GLYN;

				derivedGLIndex=tempGLPosY*GLXN+tempGLPosX;

				unique=true;
				for(int j=0; j<i; j++)
				{
					if(derivedGLIndex==conGRtoGL[grInd][j])
					{
						unique=false;
						break;
					}
				}
				if(!unique)
				{
					continue;
				}

				if(glomeruliN[derivedGLIndex].numGRDen<numGRPerGLLim)
				{
					glomeruliN[derivedGLIndex].grDenInds[glomeruliN[derivedGLIndex].numGRDen]=grInd*4+i;
					glomeruliN[derivedGLIndex].numGRDen++;
					conGRtoGL[grInd][i]=derivedGLIndex;
					complete=true;
					break;
				}
			}

			if(attempts>=60000 && !complete)
			{
				output<<"incomplete GR to GL assignment for GR#"<<grInd<<endl;
				cout<<grInd<<endl;
			}
		}
	}

	output<<"granule cells assigned to glomeruli."<<endl;
}

/*
 * new connectivity, assign golgi cell dendrites and axons to glomeruli
 */
void assignGOGLN(stringstream &output)
{
	extern CRandomSFMT0 randGenSFMT;
	extern GlomerulusN glomeruliN[NUMGLN];
	extern int conGODentoGL[NUMGO][NUMGLDENPERGON];
	extern int conGOAxtoGL[NUMGO][NUMGLAXPERGON];

	float scaleGLGOX;
	float scaleGLGOY;

	int numConnectedGO;
	bool goConnected[NUMGO];

	scaleGLGOX=(float)GLXN/GOX;
	scaleGLGOY=(float)GLYN/GOY;

	for(int i=0; i<NUMGLDENPERGON; i++)
	{
		numConnectedGO=0;
		memset(goConnected, false, NUMGO*sizeof(bool));

		while(numConnectedGO<NUMGO)
		{
			int goInd;
			int goPosX;
			int goPosY;
			int tempGODenSpanGLX;
			int tempGODenSpanGLY;
			int attempts;

			goInd=randGenSFMT.IRandomX(0, NUMGO-1);

			if(goConnected[goInd])
			{
				continue;
			}

			goConnected[goInd]=true;
			numConnectedGO++;

			goPosX=goInd%GOX;
			goPosY=(int)(goInd/GOX);

			tempGODenSpanGLX=GODENSPANGLXN;
			tempGODenSpanGLY=GODENSPANGLYN;

			for(attempts=0; attempts<1000000000; attempts++)
			{
				int tempGLPosX, tempGLPosY;
				int derivedGLIndex;

				if(attempts==50000)
				{
					tempGODenSpanGLX=tempGODenSpanGLX*2;
					tempGODenSpanGLY=tempGODenSpanGLY*2;
				}

				tempGLPosX=(int)(goPosX*scaleGLGOX+scaleGLGOX/2);
				tempGLPosY=(int)(goPosY*scaleGLGOY+scaleGLGOY/2);

				tempGLPosX+=randGenSFMT.IRandom(-tempGODenSpanGLX/2, tempGODenSpanGLX/2);
				tempGLPosY+=randGenSFMT.IRandom(-tempGODenSpanGLY/2, tempGODenSpanGLY/2);

				tempGLPosX=(tempGLPosX%GLXN+GLXN)%GLXN;
				tempGLPosY=(tempGLPosY%GLYN+GLYN)%GLYN;

				derivedGLIndex=tempGLPosY*GLXN+tempGLPosX;

				if(glomeruliN[derivedGLIndex].hasGODen)
				{
					continue;
				}

				glomeruliN[derivedGLIndex].hasGODen=true;
				glomeruliN[derivedGLIndex].goDenInd=goInd;
				conGODentoGL[goInd][i]=derivedGLIndex;
				break;
			}
		}
	}
	output<<"golgi cell dendrites assigned to glomeruli"<<endl;

	for(int i=0; i<NUMGLAXPERGON; i++)
	{
		numConnectedGO=0;
		memset(goConnected, false, NUMGO*sizeof(bool));

		while(numConnectedGO<NUMGO)
		{
			int goInd;
			int goPosX;
			int goPosY;
			int tempGOAxSpanGLX;
			int tempGOAxSpanGLY;
			int attempts;

			goInd=randGenSFMT.IRandomX(0, NUMGO-1);

			if(goConnected[goInd])
			{
				continue;
			}

			goConnected[goInd]=true;
			numConnectedGO++;

			goPosX=goInd%GOX;
			goPosY=(int)(goInd/GOX);

			tempGOAxSpanGLX=GOAXSPANGLXN;
			tempGOAxSpanGLY=GOAXSPANGLYN;

			for(attempts=0; attempts<1000000000; attempts++)
			{
				int tempGLPosX, tempGLPosY;
				int derivedGLIndex;
				bool unique;

				if(attempts==50000)
				{
					tempGOAxSpanGLX=tempGOAxSpanGLX*2;
					tempGOAxSpanGLY=tempGOAxSpanGLY*2;
				}

				tempGLPosX=(int)(goPosX*scaleGLGOX+scaleGLGOX/2);
				tempGLPosY=(int)(goPosY*scaleGLGOY+scaleGLGOY/2);

				tempGLPosX+=randGenSFMT.IRandom(-tempGOAxSpanGLX/2, tempGOAxSpanGLX/2);
				tempGLPosY+=randGenSFMT.IRandom(-tempGOAxSpanGLY/2, tempGOAxSpanGLY/2);

				tempGLPosX=(tempGLPosX%GLXN+GLXN)%GLXN;
				tempGLPosY=(tempGLPosY%GLYN+GLYN)%GLYN;

				derivedGLIndex=tempGLPosY*GLXN+tempGLPosX;

				if(glomeruliN[derivedGLIndex].hasGOAx)
				{
					continue;
				}

				glomeruliN[derivedGLIndex].hasGOAx=true;
				glomeruliN[derivedGLIndex].goAxInd=goInd;
				conGOAxtoGL[goInd][i]=derivedGLIndex;
				break;
			}
		}
	}
	output<<"golgi cell axons assigned to glomeruli"<<endl;
}

/*
 * new connectivity, assign Mossy fibers to glomeruli
 */
void assignMFGLN(stringstream &output)
{
	extern GlomerulusN glomeruliN[NUMGLN];
	extern int conMFtoGL[NUMMF][NUMGLPERMFN];
	int lastMFSynCount;
	for(int i=0; i<NUMMF-1; i++)
	{
		for(int j=0; j<NUMGLPERMFN; j++)
		{
			int glIndex;
			while(true)
			{
				glIndex=randGenSFMT.IRandom(0, NUMGLN-1);
				if(!glomeruliN[glIndex].hasMF)
				{
					glomeruliN[glIndex].mfInd=i;
					glomeruliN[glIndex].hasMF=true;
					conMFtoGL[i][j]=glIndex;
					break;
				}
			}
		}
	}

	lastMFSynCount=0;
	for(int i=0; i<NUMGLN; i++)
	{
		if(!glomeruliN[i].hasMF)
		{
			glomeruliN[i].hasMF=true;
			glomeruliN[i].mfInd=NUMMF-1;
			conMFtoGL[NUMMF-1][lastMFSynCount]=i;
			lastMFSynCount++;
		}
	}
}

/*
 * new connectivity, translate MF to GL connectivity to MF to GR and MF to GO connectivity
 */
void translateMFGLN(stringstream &output)
{
	extern GlomerulusN glomeruliN[NUMGLN];
	extern int conMFtoGL[NUMMF][NUMGLPERMFN];

	for(int i=0; i<NUMMF; i++)
	{
		numMFtoGRN[i]=0;
		numMFtoGON[i]=0;
		for(int j=0; j<NUMGLPERMFN; j++)
		{
			int glIndex;
			glIndex=conMFtoGL[i][j];
			if(glomeruliN[glIndex].hasGODen)
			{
				conMFtoGON[i][numMFtoGON[i]]=glomeruliN[glIndex].goDenInd;
				numMFtoGON[i]++;
			}

			for(int k=0; k<glomeruliN[glIndex].numGRDen; k++)
			{
				conMFtoGRN[i][k+numMFtoGRN[i]]=glomeruliN[glIndex].grDenInds[k];
			}
			numMFtoGRN[i]=numMFtoGRN[i]+glomeruliN[glIndex].numGRDen;
		}
	}
	output<<"mossy fiber to glomeruli connection translated to:"<<endl;
	output<<"mossy fiber to golgi cell connection,"<<endl;
	output<<"mossy fiber to granule cell connection."<<endl;
}

/*
 * new connectivity, translate GO axon to GL connectivity to GO axon to GR connectivity
 */
void translateGOGLN(stringstream &output)
{
	extern GlomerulusN glomeruliN[NUMGLN];
	extern int conGOAxtoGL[NUMGO][NUMGLAXPERGON];

	for(int i=0; i<NUMGO; i++)
	{
		numGOtoGRN[i]=0;
		for(int j=0; j<NUMGLAXPERGON; j++)
		{
			int glIndex;
			glIndex=conGOAxtoGL[i][j];
			for(int k=0; k<glomeruliN[glIndex].numGRDen; k++)
			{
				conGOtoGRN[i][k+numGOtoGRN[i]]=glomeruliN[glIndex].grDenInds[k];
			}
			numGOtoGRN[i]=numGOtoGRN[i]+glomeruliN[glIndex].numGRDen;
		}
	}
	output<<"golgi to glomeruli connection translated to golgi to granule cell connection."<<endl;
}

/*initialize each member of the array to a value outside the mossy fiber indices which ranges from
 * 0 to NUMMF-1. NUMMF itself is outside that range.
 */
void assignMFGL(stringstream &output)
{
	extern int glomeruli[GLX*GLY];
	extern CRandomSFMT0 randGenSFMT;
	int numGL=GLX*GLY;

	for(int i=0; i<numGL; i++)
	{
		glomeruli[i]=NUMMF;
	}

	//mossy fiber to glomeruli assignment
	for(int i=0; i<NUMMF-1; i++)
	{
		//assign each mossy fiber to have NUMGLPERMF glomeruli
		for(int j=0; j<NUMGLPERMF; j++)
		{
			int randIndex;
			while(true)
			{
				randIndex=randGenSFMT.IRandom(0, numGL-1);
				if(glomeruli[randIndex]==NUMMF)
				{
					glomeruli[randIndex]=i;
					break;
				}
			}
		}
	}

	//assign the remaining unassigned glomuli to the last mossy fiber. This is for performance reasons.
	for(int i=0; i<numGL; i++)
	{
		if(glomeruli[i]==NUMMF)
		{
			glomeruli[i]=NUMMF-1;
		}
	}

	output<<"mossy fiber assigned to glomeruli."<<endl;
}

//assign mossy fiber to granule cell connections
void assignMFGRCon(stringstream &output)
{
	extern int glomeruli[GLX*GLY];
	extern CRandomSFMT0 randGenSFMT;
	//define scaling between granule cell grind and glomeruli grid
	float scaleX=(float)GRX/GLX;
	float scaleY=(float)GRY/GLY;

	for(int i=0; i<DENPERGR; i++)
	{
		int numConnectedGR;
		bool grConnected[NUMGR];

		numConnectedGR=0;
		memset(grConnected, false, NUMGR*sizeof(bool));

		while(numConnectedGR<NUMGR)
		{
			int grInd=randGenSFMT.IRandom(0, NUMGR-1);
			int grPosX;
			int grPosY;
			int tempGRDenSpanX;
			int tempGRDenSpanY;
			int attempts;
			bool complete;

			if(grConnected[grInd])
			{
				continue;
			}

			grConnected[grInd]=true;
			numConnectedGR++;

			//get the granule cell position given the cell index.
			grPosX=grInd%GRX;
			grPosY=(int)(grInd/GRX);

			tempGRDenSpanX=GRDENSPANX;
			tempGRDenSpanY=GRDENSPANY;

			complete=false;
			for(attempts=0; attempts<60000; attempts++)
			{
				//these are used to derive glomerulus that the granule cell should connect to
				int tempGLPosX, tempGLPosY;

				int derivedGLIndex, mfIndex;

				tempGLPosX=((int)lroundf(grPosX+tempGRDenSpanX*(randGenSFMT.Random()-0.5))%GRX+GRX)%GRX;
				tempGLPosY=((int)lroundf(grPosY+tempGRDenSpanY*(randGenSFMT.Random()-0.5))%GRY+GRY)%GRY;

				tempGLPosX=((int)(tempGLPosX/scaleX)%GLX+GLX)%GLX;
				tempGLPosY=((int)(tempGLPosY/scaleY)%GLY+GLY)%GLY;

				//from the glomerulus position, derive the glomerulus index
				derivedGLIndex=tempGLPosY*GLX+tempGLPosX;
				//get the mossy fiber that glomerulus belongs to
				mfIndex=glomeruli[derivedGLIndex];
				//if that mossy fiber is not yet saturated with synapses, then make the synapses
				if(numSynMFtoGR[mfIndex]<MFGRSYNPERMF)
				{
					conMFtoGR[mfIndex][numSynMFtoGR[mfIndex]]=grInd*4+i; //granule cell index coded with dendrite index
					numSynMFtoGR[mfIndex]++;
					complete=true;
					break;
				}

				//increase the dendritic span of the granule cell to increase the chance of making connection
				if(attempts==499 || attempts==999 || attempts==1999)
				{
					tempGRDenSpanX=tempGRDenSpanX*2;
					tempGRDenSpanY=tempGRDenSpanY*2;
				}
			}
			if(attempts>=60000 && !complete)
			{
				//output "incomplete MF to GR connection for GR# at grPosX, grPosY
				output<<"incomplete MF to GR connection for GR#"<<grInd<<endl;
			}
		}
	}
	//output connection complete message
	output<<"mossy fibers connected to granule cells"<<endl;
}

//assign mossy fibers to golgi cell connections
void assignMFGOCon(stringstream &output)
{
	extern int glomeruli[GLX*GLY];
	extern CRandomSFMT0 randGenSFMT;

	//define scaling between golgi cell grid and glomeruli grid
	float scaleX=(float) GLX/GOX;
	float scaleY=(float) GLY/GOY;

	for(int i=0; i<MFDENPERGO; i++)
	{
		int numConnectedGO;
		bool goConnected[NUMGO];

		numConnectedGO=0;
		memset(goConnected, false, NUMGO*sizeof(bool));

		while(numConnectedGO<NUMGO)
		{
			int goInd=randGenSFMT.IRandom(0, NUMGO-1);
			int goPosX;
			int goPosY;
			int tempGODenSpanX;
			int tempGODenSpanY;
			int attempts;
			bool complete;

			if(goConnected[goInd])
			{
				continue;
			}

			goConnected[goInd]=true;
			numConnectedGO++;
			//get the golgi cell position given the cell index
			goPosX=goInd%GOX;
			goPosY=(int) (goInd/GOX);

			tempGODenSpanX=GODENSPANX;
			tempGODenSpanY=GODENSPANY;

			attempts;
			complete=false;
			for(attempts=0; attempts<3000; attempts++)
			{
				//these are used to derive glomerulus that the granule cell should connect to
				int tempGLPosX, tempGLPosY;
				float glPosXf, glPosYf;

				int derivedGLIndex, mfIndex;

				glPosXf= ((goPosX+0.5)*scaleX);
				glPosYf= ((goPosY+0.5)*scaleY);
				tempGLPosX=((int)lroundf(glPosXf+tempGODenSpanX*(randGenSFMT.Random()-0.5))%GLX+GLX)%GLX;
				tempGLPosY=((int)lroundf(glPosYf+tempGODenSpanY*(randGenSFMT.Random()-0.5))%GLY+GLY)%GLY;

				//from the glomerulus position, derive the glomerulus index
				derivedGLIndex=tempGLPosY*GLX+tempGLPosX;

				mfIndex=glomeruli[derivedGLIndex];

				if(numSynMFtoGO[mfIndex]<MFGOSYNPERMF)
				{
					conMFtoGO[mfIndex][numSynMFtoGO[mfIndex]]=goInd; //golgi cell index
					numSynMFtoGO[mfIndex]++;
					complete=true;
					break;
				}
				if(attempts==499 || attempts==999)
				{
					tempGODenSpanX=tempGODenSpanX*2;
					tempGODenSpanY=tempGODenSpanY*2;
				}
			}
			if(attempts>=3000 && !complete)
			{
				//output "incomplete MF to GO connections for GO# at goPos X, goPosY
				output<<"incomplete MF to GO connections for GO#"<<goInd<<endl;
			}
		}
	}
	//output connection complete
	output<<"mossy fibers connected to golgi cells"<<endl;
}

//assign granule cell to golgi cell connections
void assignGRGOCon(stringstream &output)
{
	extern CRandomSFMT0 randGenSFMT;

	//assign scaling factors between golgi cell grid and granule cell grid.
	float scaleX=(float) GRX/GOX;
	float scaleY=(float) GRY/GOY;

	for(int i=0; i<GRGOSYNPERGO; i++)
	{
		int numConnectedGO;
		bool goConnected[NUMGO];

		numConnectedGO=0;
		memset(goConnected, false, NUMGO*sizeof(bool));

		while(numConnectedGO<NUMGO)
		{
			int goInd=randGenSFMT.IRandom(0, NUMGO-1);
			int goPosX;
			int goPosY;
			int tempGODenSpanX;
			int tempGODenSpanY;
			int attempts;
			bool complete;

			if(goConnected[goInd])
			{
				continue;
			}

			goConnected[goInd]=true;
			numConnectedGO++;

			//get golgi cell coordinates from the cell index
			goPosX=goInd%GOX;
			goPosY=(int) goInd/GOX;

			tempGODenSpanX=GOFROMGRDENSPANX;
			tempGODenSpanY=GOFROMGRDENSPANY;

			attempts;
			complete=false;
			for(attempts=0; attempts<50000; attempts++)
			{
				int tempGRPosX, tempGRPosY;
				double tempGRPosXf, tempGRPosYf; //to eliminate truncation errors

				int derivedGRIndex;

				//given a golgi cell coordinate, randomly find a granule cell coordinate within the denspan x and y of the golgi cell
				//% operations are to take care of wraparounds
				tempGRPosXf=(goPosX+0.5)*scaleX;
				tempGRPosYf=(goPosY+0.5)*scaleY;
				tempGRPosX=((int)lroundf(tempGRPosXf+(double)tempGODenSpanX*(randGenSFMT.Random()-0.5))%GRX+GRX)%GRX;
				tempGRPosY=((int)lroundf(tempGRPosYf+(double)tempGODenSpanY*(randGenSFMT.Random()-0.5))%GRY+GRY)%GRY;

				//get the granule cell index given a granule cell coordinate
				derivedGRIndex=tempGRPosY*GRX+tempGRPosX;
				//if that granule cell is not saturated with synapses, make the connection
				if(numSynGRtoGO[derivedGRIndex]<GRGOSYNPERGR)
				{
					conGRtoGO[derivedGRIndex][numSynGRtoGO[derivedGRIndex]]=goInd;
					numSynGRtoGO[derivedGRIndex]++;
					complete=true;
					break;
				}

				//if at the 5000th try still can't find an unsaturated granule cell, increase the den span to 10 less than
				//the granule cell grid, to increase the chance of making connection
				if(attempts==4999)
				{
#ifdef DEBUG
					output<<"expanding GO dendrites for GO#"<<goInd<<" at syn iteration #"<<i<<endl;
#endif
					tempGODenSpanX=GRX-10;
					tempGODenSpanY=GRY-10;
				}
			}
			if(attempts>=50000 && !complete)
			{
				//output "incomplete GR to GO connections for GO#
				output<<"incomplete GR to GO connections for GO#"<<goInd<<endl;
			}
		}
	}
	//****output "granule to golgi synapses connected"
	output<<"granule to golgi synapses connected."<<endl;
}

//assign golgi cell to granule cell connections
void assignGOGRCon(stringstream &output)
{
	extern CRandomSFMT0 randGenSFMT;
#ifdef DEBUG
	extern unsigned short gOtoGRConGR[NUMGR][DENPERGR];
	extern unsigned short gOtoGRConGRSN[NUMGR];
#endif

	//assign coordinate scaling between granule cell grid and golgi cell grid
	float scaleX=(float)GRX/GOX;
	float scaleY=(float)GRY/GOY;

#ifdef DEBUG
	memset(gOtoGRConGR, 0, NUMGR*DENPERGR*sizeof(unsigned short));
	memset(gOtoGRConGRSN, 0, NUMGR*sizeof(unsigned short));
#endif

	for(int i=0; i<DENPERGR; i++)
	{
		//number of granule cells that already have connections computed
		int numConnectedGR;
		//given a Granule cell index i, where 0<=i<NUMGR, grConnected[i]=true
		//if the connection for that cell is already computed, false otherwise
		bool grConnected[NUMGR];

		numConnectedGR=0;
		memset(grConnected, false, NUMGR*sizeof(bool));
		//if the number of granule cells whose connections are already computed
		//equal to the total number of granule cells, then done
		while(numConnectedGR<NUMGR)
		{
			//randomly select a granule cell to compute the connections
			//this is done to minimize the incomplete connections
			int grInd=randGenSFMT.IRandom(0, NUMGR-1);
			int grPosX;
			int grPosY;
			int tempGRDenSpanX;
			int tempGRDenSpanY;
			int attempts;
			bool complete;

			//if for that particular selected index the granule cell is already
			//connected, then don't compute connection again
			if(grConnected[grInd])
			{
				continue;
			}

			grConnected[grInd]=true;
			numConnectedGR++;

			//get granule cell coordinate given a granule cell index
			grPosX=grInd%GRX;
			grPosY=(int) grInd/GRX;

			tempGRDenSpanX=GRFROMGODENSPANX;
			tempGRDenSpanY=GRFROMGODENSPANY;

			attempts;
			complete=false;
			for(attempts=0; attempts<50000; attempts++)
			{
				int tempGOPosX, tempGOPosY;
				int derivedGOIndex;

				/* given a granule cell coordinate, randomly find a golgi cell that lies within the denspan of the
				 * granule cell
				 * The % ops are for taking care of wraparounds. Since the % operator is not a true modulus
				 * operator, a work around is ((x%y)+y)%y, which returns the expected modulus answer for a
				 * negative integer.
				 */
				tempGOPosX=((grPosX+(int)lround(tempGRDenSpanX*(randGenSFMT.Random()-0.5)))%GRX+GRX)%GRX;
				tempGOPosY=((grPosY+(int)lround(tempGRDenSpanY*(randGenSFMT.Random()-0.5)))%GRY+GRY)%GRY;

				//transform to golgi cell coordinates
				tempGOPosX=(((int)(tempGOPosX/scaleX))%GOX+GOX)%GOX;
				tempGOPosY=(((int)(tempGOPosY/scaleY))%GOY+GOY)%GOY;

				//given the golgi cell coordinate, derive the golgi cell index
				derivedGOIndex=tempGOPosY*GOX+tempGOPosX;
				//if the golgi cell is not saturated with synapses, make the connection
				if(numSynGOtoGR[derivedGOIndex]<GOGRSYNPERGO)
				{
					conGOtoGR[derivedGOIndex][numSynGOtoGR[derivedGOIndex]]=grInd*4+i; //encode gr # and dendrite #
					numSynGOtoGR[derivedGOIndex]++;
					complete=true;
#ifdef DEBUG
					gOtoGRConGR[grInd][gOtoGRConGRSN[grInd]]=derivedGOIndex;
					gOtoGRConGRSN[grInd]++;
#endif
					break;
				}
				//if at 5000's attempt no connection is made, double the denspan to increase the chance
				//of making a connection
				if(attempts==4999)
				{
					tempGRDenSpanX=tempGRDenSpanX*2;
					tempGRDenSpanY=tempGRDenSpanY*2;
				}
			}
			if(attempts>=50000 && !complete)
			{
#ifdef DEBUG
				incompGRsGOGR.push_back(grInd);
#endif
				//output "incomplete GO to GR connection for GR#
				output<<"incomplete GO to GR connection for GR#"<<grInd<<endl;
			}
		}
	}
	//output "golgi to granule cells connected"
	output<<"golgi to granule cells connected"<<endl;
}

void assignBCPCCon(stringstream &output)
{
	for(int i=0; i<NUMPC; i++)
	{
		conBCtoPC[i*NUMBCPERPC][0]=((i+1)%NUMPC+NUMPC)%NUMPC;
		conBCtoPC[i*NUMBCPERPC][1]=((i-1)%NUMPC+NUMPC)%NUMPC;
		conBCtoPC[i*NUMBCPERPC][2]=((i+2)%NUMPC+NUMPC)%NUMPC;
		conBCtoPC[i*NUMBCPERPC][3]=((i-2)%NUMPC+NUMPC)%NUMPC;

		conBCtoPC[i*NUMBCPERPC+1][0]=((i+1)%NUMPC+NUMPC)%NUMPC;
		conBCtoPC[i*NUMBCPERPC+1][1]=((i-1)%NUMPC+NUMPC)%NUMPC;
		conBCtoPC[i*NUMBCPERPC+1][2]=((i+3)%NUMPC+NUMPC)%NUMPC;
		conBCtoPC[i*NUMBCPERPC+1][3]=((i-3)%NUMPC+NUMPC)%NUMPC;

		conBCtoPC[i*NUMBCPERPC+2][0]=((i+3)%NUMPC+NUMPC)%NUMPC;
		conBCtoPC[i*NUMBCPERPC+2][1]=((i-3)%NUMPC+NUMPC)%NUMPC;
		conBCtoPC[i*NUMBCPERPC+2][2]=((i+6)%NUMPC+NUMPC)%NUMPC;
		conBCtoPC[i*NUMBCPERPC+2][3]=((i-6)%NUMPC+NUMPC)%NUMPC;

		conBCtoPC[i*NUMBCPERPC+3][0]=((i+4)%NUMPC+NUMPC)%NUMPC;
		conBCtoPC[i*NUMBCPERPC+3][1]=((i-4)%NUMPC+NUMPC)%NUMPC;
		conBCtoPC[i*NUMBCPERPC+3][2]=((i+9)%NUMPC+NUMPC)%NUMPC;
		conBCtoPC[i*NUMBCPERPC+3][3]=((i-9)%NUMPC+NUMPC)%NUMPC;
	}
	output<<"basket to purkinje cells connected"<<endl;
}

void assignIOCoupling(stringstream &output)
{
	output.str("");
	for(int i=0; i<NUMIO; i++)
	{
		for(int j=0; j<IOCOUPSYNPERIO; j++)
		{
			conIOCouple[i][j]=((i+1)%NUMIO+NUMIO)%NUMIO;
		}
	}
	output<<"IO coupling connected"<<endl;
}

void assignPCNCCon(stringstream &output)
{
	extern CRandomSFMT0 randGenSFMT;
	char numSynPCPerNC[NUMNC];
	memset(numSynPCPerNC, 0, NUMNC*sizeof(char));

	for(int i=0; i<NUMPC; i++)
	{
		conPCtoNC[i][0]=(char)(i/(PCNCSYNPERNC/PCNCSYNPERPC))*PCNCSYNPERNC+(char)(i%(PCNCSYNPERNC/PCNCSYNPERPC));
		for(int j=1; j<PCNCSYNPERPC; j++)
		{
			while(true)
			{
				int indSynNC;
				indSynNC=randGenSFMT.Random()*NUMNC*PCNCSYNPERNC;

				if((indSynNC%PCNCSYNPERNC)<(PCNCSYNPERNC/PCNCSYNPERPC))
				{
					continue;
				}
				if(numSynPCPerNC[indSynNC/PCNCSYNPERNC]<(PCNCSYNPERNC-(PCNCSYNPERNC/PCNCSYNPERPC)))
				{
					numSynPCPerNC[indSynNC/PCNCSYNPERNC]++;
					conPCtoNC[i][j]=indSynNC;
					break;
				}
			}
		}
	}

#ifdef DEBUG
	for(int i=0; i<NUMPC; i++)
	{
		for(int j=0; j<PCNCSYNPERPC; j++)
		{
			output<<(int)conPCtoNC[i][j]<<" ";
		}
		output<<endl;
	}
	for(int i=0; i<NUMNC; i++)
	{
		output<<(int)numSynPCPerNC[i]<<" ";
	}
	output<<endl;
#endif

	output<<"Purkinje cells to nucleus cells connected"<<endl;
}

void assignGRDelays(stringstream &output)
{
	unsigned char delayMasks[8]={0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};

	//calculate delay masks for each granule cells
	for(int i=0; i<NUMGR; i++)
	{
		int grPosX;
		int grBCPCSCDist;


		//calculate x coordinate of GR position
		grPosX=i%GRX;

		//calculate distance of GR to BC, PC, and SC, and assign time delay.
		grBCPCSCDist=abs(1024-grPosX); //todo: put constants as marcos in params
		delayBCPCSCMaskGR[i]=delayMasks[0];//[(int)grBCPCSCDist/147+1];//256+1]; //todo: put consts as marcos

		for(int j=0; j<GRGOSYNPERGR; j++)
		{
			int goPosX;
			int grGODist;
			int distTemp;
			//calculate x position of GO that the GR is outputting to
			goPosX=conGRtoGO[i][j]%GOX;

			//convert from golgi coordinate to granule coordinates
			goPosX=(goPosX+0.5)*(GRX/GOX);

			//calculate distance between GR and GO
			grGODist=GRX;
			distTemp=abs(grPosX-goPosX);
			grGODist=(distTemp<grGODist)*distTemp+(distTemp>=grGODist)*grGODist;
			distTemp=(GRX-goPosX)+grPosX;
			grGODist=(distTemp<grGODist)*distTemp+(distTemp>=grGODist)*grGODist;
			distTemp=(GRX-grPosX)+goPosX;
			grGODist=(distTemp<grGODist)*distTemp+(distTemp>=grGODist)*grGODist;
			if(grGODist>1024)
			{
				output<<"error in calculating gr to go distances: "<<grGODist<<endl;
				grGODist=1023;
			}

			//calculate time delay based distance
			delayGOMasksGR[i][j]=delayMasks[0];//(int)grGODist/147+1];//256+1]; //todo: put consts as marcos
		}
	}
}

#ifdef DEBUG
void connectivityDebugInfo(stringstream &output)
{
	for(int i=0; i<NUMGO; i++)
	{
		if(numSynGOtoGR[i]<GOGRSYNPERGO-1)//numSynGOGR[i]
		{
			incompGOsGOGR.push_back(i);
			output<<"GO #"<<i<<" connections: "<<numSynGOtoGR[i]<<endl;
		}
	}

	//calculate distribution of unique GO connections of GR cells
	{
		unsigned short numUGOPerGR[NUMGR];
		unsigned int numGRNumUGO[DENPERGR];
		memset(numUGOPerGR, 0, NUMGR*sizeof(unsigned short));
		memset(numGRNumUGO, 0, DENPERGR*sizeof(unsigned int));

		for(int i=0; i<NUMGR; i++)
		{
			for(int j=0; j<gOtoGRConGRSN[i]; j++)
			{
				for(int k=j; k<gOtoGRConGRSN[i]; k++)
				{
					if(k==j)
					{
						continue;
					}
					if(gOtoGRConGR[i][k]==gOtoGRConGR[i][j])
					{
						gOtoGRConGR[i][k]=NUMGO;
					}
				}
			}
		}
		for(int i=0; i<NUMGR; i++)
		{
			for(int j=0; j<gOtoGRConGRSN[i]; j++)
			{
				if(gOtoGRConGR[i][j]<NUMGO)
				{
					numUGOPerGR[i]++;
				}
			}

			numGRNumUGO[numUGOPerGR[i]-1]++;

		}

		for(int i=0; i<DENPERGR; i++)
		{
			output<<"number of granules cells connected to "<<i+1<<" unique golgi cells: "<<numGRNumUGO[i]<<endl;
		}
	}
}
#endif

void genesisCLI()
{
	stringstream output;
	//check if connections is already made, initialized is a global variable
	if(initialized)
	{
		return;
	}
	output.str("");
	assignMFGL(output);
	cout<<output.str()<<endl;

	//new connectivity
	initializeVars();

	output.str("");
	assignGRDenGLN(output);
	cout<<output.str()<<endl;

	output.str("");
	assignGOGLN(output);
	cout<<output.str()<<endl;

	output.str("");
	assignMFGLN(output);
	cout<<output.str()<<endl;

	output.str("");
	translateMFGLN(output);
	cout<<output.str()<<endl;

	output.str("");
	translateGOGLN(output);
	cout<<output.str()<<endl;

	//------------------------end new connectivity

	output.str("");
	assignMFGRCon(output);
	cout<<output.str()<<endl;

	output.str("");
	assignMFGOCon(output);
	cout<<output.str()<<endl;

	output.str("");
	assignGRGOCon(output);
	cout<<output.str()<<endl;

	output.str("");
	assignGOGRCon(output);
	cout<<output.str()<<endl;

	output.str("");
	assignBCPCCon(output);
	cout<<output.str()<<endl;

	output.str("");
	assignIOCoupling(output);
	cout<<output.str()<<endl;

	output.str("");
	assignPCNCCon(output);
	cout<<output.str()<<endl;

	output.str("");
	assignGRDelays(output);
	cout<<output.str()<<endl;

	//debug information
#ifdef DEBUG
	output.str("");
	connectivityDebugInfo(output);
	cout<<output.str()<<endl;
#endif
}

void genesisGUI(QTextBrowser *output)
{
	stringstream formatout;
	//error check
	if(output==NULL)
	{
		cerr<<"error! null gui text output pointer. Exiting..."<<endl;
		return;
	}

	//check if connections is already made, initialized is a global variable
	if(initialized)
	{
		output->textCursor().insertText("Connections already made\n");
		output->repaint(); //update immediately
		return;
	}

	//begin new connectivity
	initializeVars();

	formatout.str("");
	assignGRDenGLN(formatout);
	output->textCursor().insertText(formatout.str().c_str());
	output->repaint();

	formatout.str("");
	assignGOGLN(formatout);
	output->textCursor().insertText(formatout.str().c_str());
	output->repaint();

	formatout.str("");
	assignMFGLN(formatout);
	output->textCursor().insertText(formatout.str().c_str());
	output->repaint();

	formatout.str("");
	translateMFGLN(formatout);
	output->textCursor().insertText(formatout.str().c_str());
	output->repaint();

	formatout.str("");
	translateGOGLN(formatout);
	output->textCursor().insertText(formatout.str().c_str());
	output->repaint();

	//------------------------end new connectivity

	formatout.str("");
	assignMFGL(formatout);
	output->textCursor().insertText(formatout.str().c_str());
	output->repaint();

	formatout.str("");
	assignMFGRCon(formatout);
	output->textCursor().insertText(formatout.str().c_str());
	output->repaint();

	formatout.str("");
	assignMFGOCon(formatout);
	output->textCursor().insertText(formatout.str().c_str());
	output->repaint();

	formatout.str("");
	assignGRGOCon(formatout);
	output->textCursor().insertText(formatout.str().c_str());
	output->repaint();

	formatout.str("");
	assignGOGRCon(formatout);
	output->textCursor().insertText(formatout.str().c_str());
	output->repaint();

	formatout.str("");
	assignBCPCCon(formatout);
	output->textCursor().insertText(formatout.str().c_str());
	output->repaint();

	formatout.str("");
	assignIOCoupling(formatout);
	output->textCursor().insertText(formatout.str().c_str());
	output->repaint();

	formatout.str("");
	assignPCNCCon(formatout);
	output->textCursor().insertText(formatout.str().c_str());
	output->repaint();

	formatout.str("");
	assignGRDelays(formatout);
	output->textCursor().insertText(formatout.str().c_str());
	output->repaint();
	//debug information
#ifdef DEBUG
	formatout.str("");
	connectivityDebugInfo(formatout);
	output->textCursor().insertText(formatout.str().c_str());
	output->repaint();
#endif
}
