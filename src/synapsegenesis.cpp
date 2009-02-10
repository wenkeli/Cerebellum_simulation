/*
 * synapsegenesis.cpp
 *
 *  Created on: Jan 27, 2009
 *      Author: wen
 */

#include "../includes/synapsegenesis.h"

void genesis(QTextBrowser *output)
{
	int *numSyn=new int[NUMGR];
	int glomeruli[GLX*GLY];
	int numGL=GLX*GLY;

	float scaleX;
	float scaleY;

	stringstream formatOut;
	QString outStr;

	CRandomMother randGen(time(NULL));

	if(connsMade)
	{
		output->textCursor().insertText("Connections already made\n");
		output->repaint();
		return;
	}

	/*initialize each member of the array to a value outside the mossy fiber indices which ranges from
	 * 0 to NUMMF-1. NUMMF itself is outside that range.
	 */
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
				randIndex=randGen.iRandom(0,numGL-1);
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

	//****gui output to indicate mossy fiber assigned to glomeruli.
	output->textCursor().insertText("mossy fiber assigned to glomeruli.\n");
	output->repaint();
	//cout<<"mossy fiber assigned to glomeruli."<<endl;

	//initialize the number of (granule cell) synapses already made for each mossy fiber to 0
	for(int i=0; i<NUMMF; i++)
	{
		numSyn[i]=0;
	}
	//assign mossy fiber to granule cell connections
	//define the grid scaling between glomeruli grid and granule cell grid. the scaling factors
	//are defined in terms of glomeruli because glomeruli grid is smaller
	scaleX=(float) GRX/GLX;
	scaleY=(float) GRY/GLY;
	for(int i=0; i<DENPERGR; i++)
	{
		for(int j=0; j<NUMGR; j++)
		{
			//get the granule cell position given the cell index.
			int grPosX=j%GRX;
			int grPosY=(int)(j/GRX);

			int tempGRDenSpanX=GRDENSPANX;
			int tempGRDenSpanY=GRDENSPANY;

			int attempts;
			bool complete=false;
			for(attempts=0; attempts<60000; attempts++)
			{
				//these are used to derive glomerulus that the granule cell should connect to
				int tempGLPosX, tempGLPosY;

				int derivedGLIndex, mfIndex;

				tempGLPosX=(int) lroundf(grPosX+tempGRDenSpanX*(randGen.fRandom()-0.5));
				tempGLPosY=(int) lroundf(grPosY+tempGRDenSpanY*(randGen.fRandom()-0.5));
				tempGLPosX=(int) lroundf(tempGLPosX/scaleX-0.5);
				tempGLPosY=(int) lroundf(tempGLPosY/scaleY-0.5);

				//wrap around if out of bounds
				if(tempGLPosX>=GLX)
				{
					tempGLPosX=tempGLPosX-GLX;
				}
				if(tempGLPosX<0)
				{
					tempGLPosX=tempGLPosX+GLX;
				}
				if(tempGLPosY>=GLY)
				{
			 		tempGLPosY=tempGLPosY-GLY;
				}
				if(tempGLPosY<0)
				{
					tempGLPosY=tempGLPosY+GLY;
				}

				//from the glomerulus position, derive the glomerulus index
				derivedGLIndex=tempGLPosY*GLX+tempGLPosX;

				mfIndex=glomeruli[derivedGLIndex];

				if(numSyn[mfIndex]<MFGRSYNPERMF)
				{
					conMFtoGR[mfIndex][numSyn[mfIndex]][0]=j; //granule cell index
					conMFtoGR[mfIndex][numSyn[mfIndex]][1]=i; //dendrite index
					numSyn[mfIndex]++;
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
				//cout<<"incomplete MF to GR connection for GR#"<<j<<endl;
				formatOut.str("");
				formatOut<<"incomplete MF to GR connection for GR#"<<j<<endl;
				outStr=formatOut.str().c_str();
				output->textCursor().insertText(outStr);
			}
		}
	}
	//****output "mossy fibers connected to granule cells"
	//cout<<"mossy fibers connected to granule cells"<<endl;
	output->textCursor().insertText("mossy fibers connected to granule cells\n");
	output->repaint();

	//initialize the number of (golgi cell) synapses already made to 0 for each mossy fiber
	for(int i=0; i<NUMMF; i++)
	{
		numSyn[i]=0;
	}
	//assign mossy fibers to golgi cell connections
	//define scaling between golgi cell grid and glomeruli grid
	scaleX=(float) GLX/GOX;
	scaleY=(float) GLY/GOY;
	for(int i=0; i<MFDENPERGO; i++)
	{
		for(int j=0; j<NUMGO; j++)
		{
			//get the golgi cell position given the cell index
			int goPosX=j%GOX;
			int goPosY=(int) (j/GOX);

			int tempGODenSpanX=GODENSPANX;
			int tempGODenSpanY=GODENSPANY;

			int attempts;
			bool complete=false;
			for(attempts=0; attempts<3000; attempts++)
			{
				//these are used to derive glomerulus that the granule cell should connect to
				int tempGLPosX, tempGLPosY;
				float glPosXf, glPosYf;

				int derivedGLIndex, mfIndex;

				glPosXf= ((goPosX+0.5)*scaleX);
				glPosYf= ((goPosY+0.5)*scaleY);
				tempGLPosX=(int) lroundf(glPosXf+tempGODenSpanX*(randGen.fRandom()-0.5));
				tempGLPosY=(int) lroundf(glPosYf+tempGODenSpanY*(randGen.fRandom()-0.5));

				//wrap around if out of bounds
				if(tempGLPosX>=GLX)
				{
					tempGLPosX=tempGLPosX-GLX;
				}
				if(tempGLPosX<0)
				{
					tempGLPosX=tempGLPosX+GLX;
				}
				if(tempGLPosY>=GLY)
				{
					tempGLPosY=tempGLPosY-GLY;
				}
				if(tempGLPosY<0)
				{
					tempGLPosY=tempGLPosY+GLY;
				}

				//from the glomerulus position, derive the glomerulus index
				derivedGLIndex=tempGLPosY*GLX+tempGLPosX;

				mfIndex=glomeruli[derivedGLIndex];

				if(numSyn[mfIndex]<MFGOSYNPERMF)
				{
					conMFtoGO[mfIndex][numSyn[mfIndex]][0]=j; //golgi cell index
					conMFtoGO[mfIndex][numSyn[mfIndex]][1]=i; //dendrite index
					numSyn[mfIndex]++;
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
				//cout<<"incomplete MF to GO connections for GO#"<<j<<endl;
				formatOut.str("");
				formatOut<<"incomplete MF to GO connections for GO#"<<j<<endl;
				outStr=formatOut.str().c_str();
				output->textCursor().insertText(outStr);
			}
		}
	}
	//****output "mossy fibers connected to golgi cells"
	//cout<<"mossy fibers connected to golgi cells"<<endl;
	output->textCursor().insertText("mossy fibers connected to golgi cells\n");
	output->repaint();

	for(int i=0; i<NUMGR; i++)
	{
		numSyn[i]=0;
	}
	//assign granule cell to golgi cell connections
	//assign scaling factors between golgi cell grid and granule cell grid.
	scaleX=(float) GRX/GOX;
	scaleY=(float) GRY/GOY;
	for(int i=0; i<GRGOSYNPERGO; i++)
	{
		for(int j=0; j<NUMGO; j++)
		{
			int goPosX=j%GOX;
			int goPosY=(int) j/GOX;

			int tempGODenSpanX=GOFROMGRDENSPANX;
			int tempGODenSpanY=GOFROMGRDENSPANY;

			int attempts;
			bool complete=false;
			for(attempts=0; attempts<50000; attempts++)
			{
				int tempGRPosX, tempGRPosY;

				int derivedGRIndex;

				tempGRPosX=(int) lroundf((goPosX+0.5)*scaleX);
				tempGRPosY=(int) lroundf((goPosY+0.5)*scaleY);
				tempGRPosX=(int) lroundf(tempGRPosX+tempGODenSpanX*(randGen.fRandom()-0.5));
				tempGRPosY=(int) lroundf(tempGRPosY+tempGODenSpanY*(randGen.fRandom()-0.5));

				//wrap around if out of bounds
				if(tempGRPosX>=GRX)
				{
					tempGRPosX=tempGRPosX-GRX;
				}
				if(tempGRPosX<0)
				{
					tempGRPosX=tempGRPosX+GRX;
				}
				if(tempGRPosY>=GRY)
				{
					tempGRPosY=tempGRPosY-GRY;
				}
				if(tempGRPosY<0)
				{
					tempGRPosY=tempGRPosY+GRY;
				}

				derivedGRIndex=tempGRPosY*GRX+tempGRPosX;

				if(numSyn[derivedGRIndex]<GRGOSYNPERGR)
				{
					conGRtoGO[derivedGRIndex][numSyn[derivedGRIndex]][0]=j;
					conGRtoGO[derivedGRIndex][numSyn[derivedGRIndex]][1]=i;
					numSyn[derivedGRIndex]++;
					complete=true;
					break;
				}

				//change the dendritic span of the golgi cell to increase the chance of making connection
				if(attempts==4999)
				{
					tempGODenSpanX=GRX-10;
					tempGODenSpanY=GRY-10;
				}
			}
			if(attempts>=50000 && !complete)
			{
				//output "incomplete GR to GO connections for GO#
				//cout<<"incomplete GR to GO connections for GO#"<<j<<endl;
				formatOut.str("");
				formatOut<<"incomplete GR to GO connections for GO#"<<j<<endl;
				outStr=formatOut.str().c_str();
				output->textCursor().insertText(outStr);
			}
		}
	}
	//****output "granule to golgi synapses connected"
	//cout<<"granule to golgi synapses connected."<<endl;
	output->textCursor().insertText("granule to golgi synapses connected.\n");
	output->repaint();

	connsMade=true;


	for(int i=0; i<NUMGR; i++)
	{
		numSyn[i]=0;
	}
	//randGen.randomInit(time(NULL)+randGen.iRandom(0, 533232));
	//assign golgi cell to granule cell connections
	for(int i=0; i<DENPERGR; i++)
	{
		for(int j=0; j<NUMGR; j++)
		{
			int grPosX=j%GRX;
			int grPosY=(int) j/GRX;

			int tempGRDenSpanX=GRFROMGODENSPANX;
			int tempGRDenSpanY=GRFROMGODENSPANY;

			int attempts;
			bool complete=false;
			for(attempts=0; attempts<50000; attempts++)
			{
				int tempGOPosX, tempGOPosY;
				float tempGOPosXf, tempGOPosYf;
				int derivedGOIndex;

				tempGOPosXf= (float)grPosX+(float)tempGRDenSpanX*(randGen.fRandom()-0.5);
				tempGOPosYf= (float)grPosY+(float)tempGRDenSpanY*(randGen.fRandom()-0.5);
				tempGOPosX=(int) lroundf(tempGOPosXf/scaleX-0.5);
				tempGOPosY=(int) lroundf(tempGOPosYf/scaleY-0.5);

				if(tempGOPosX>=GOX)
				{
					tempGOPosX=tempGOPosX-GOX;
				}
				if(tempGOPosX<0)
				{
					tempGOPosX=tempGOPosX+GOX;
				}
				if(tempGOPosY>=GOY)
				{
					tempGOPosY=tempGOPosY-GOY;
				}
				if(tempGOPosY<0)
				{
					tempGOPosY=tempGOPosY+GOY;
				}

				derivedGOIndex=tempGOPosY*GOX+tempGOPosX;
				if(numSyn[derivedGOIndex]<GOGRSYNPERGO)
				{
					conGOtoGR[derivedGOIndex][numSyn[derivedGOIndex]][0]=j;
					conGOtoGR[derivedGOIndex][numSyn[derivedGOIndex]][1]=i;
					numSyn[derivedGOIndex]++;
					complete=true;
					break;
				}

				if(attempts==4999)
				{
					tempGRDenSpanX=tempGRDenSpanX*2;
					tempGRDenSpanY=tempGRDenSpanY*2;
					//cout<<"GR #"<<j<<" at 4999"<<endl;
				}
			}
			if(attempts>=50000 && !complete)
			{
				//output "incomplete GO to GR connection for GR#
				//cout<<"incomplete GO to GR connection for GR#"<<j<<endl;
				incompGRsGOGR.push_back(j);
				formatOut.str("");
				formatOut<<"incomplete GO to GR connection for GR#"<<j<<endl;//<<" at dendrite #"<<i<<endl;
				outStr=formatOut.str().c_str();
				output->textCursor().insertText(outStr);
			}
		}
	}
	//output "golgi to granule cells connected"
	//cout<<"golgi to granule cells connected"<<endl;
	output->textCursor().insertText("golgi to granule cells connected\n");
	output->repaint();
	for(int i=0; i<NUMGO; i++)
	{
		if(numSyn[i]<GOGRSYNPERGO-1)
		{
			incompGOsGOGR.push_back(i);
			formatOut.str("");
			formatOut<<"GO #"<<i<<" connections: "<<numSyn[i]<<endl;
			output->textCursor().insertText(formatOut.str().c_str());
		}
	}

	//free memory
	for(int i=0; i<NUMGR; i++)
	{
		delete &numSyn[i];
	}
	delete [] numSyn;
}
