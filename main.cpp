/************************************************************************
Main program to call greens
Version 2.0, May 1, 2010.
Version 3.0, May 17, 2011.
Version 4.0, May 1, 2015.
See greens.cpp for description of changes.
***********************************************************************/
#define _CRT_SECURE_NO_DEPRECATE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "nrutil.h"
#include <Windows.h>	//needed for CopyFile
//#include <boost/filesystem.hpp>	//needed for copy_file, no longer requires windows.h
//namespace fs = boost::filesystem;

void input(void);
void analyzenet(void);
void picturenetwork(float *nodvar, float *segvar, const char fname[]);
void greens(void);
void contour(const char fname[]);
void histogram(const char fname[]);
void setuparrays0();
void setuparrays1(int nseg, int nnod);
void setuparrays2(int nnv, int nnt);
void blood(float c, float hem, float *p, float *pp);
void cmgui(float *segvar);
void postgreens(void);
float bloodconc(float p, float h);

void bicgstabBLASDinit(int nnvGPU);
void bicgstabBLASDend(int nnvGPU);
void bicgstabBLASStinit(int nntGPU);
void bicgstabBLASStend(int nntGPU);
void tissueGPUinit(int nntGPU, int nnvGPU);
void tissueGPUend(int nntGPU, int nnvGPU);

int max=100,nmaxvessel,nmaxtissue,nmax, nmaxbc, rungreens,initgreens,g0method,linmethod;
int mxx,myy,mzz,nnt,nseg,nnod,nnodfl,nnv,nsp,nnodbc,nodsegm,nsegfl,kmain;
int slsegdiv,nsl1,nsl2;
int is2d; //needed for 2d version
int nvaryparams,nruns,ntissparams,npostgreensparams,npostgreensout;	//needed for varying parameters, postgreens

float *dtmin;//added July 2011
int *mainseg,*permsolute,*nodrank,*nodtyp,*nodout,*bcnodname,*bcnod,*bctyp,*lowflow;
int *nodname,*segname,*segtyp,*nspoint,*istart,*nl,*nk,*indx,*ista,*iend;
int *errvesselcount,*errtissuecount;
int *imaxerrvessel,*imaxerrtissue,*nresis;  //added April 2010
int *oxygen,*diffsolute; //added April 2010
int **segnodname,**nodseg,**tisspoints,**nodnod;
int ***nbou;

int **tissfix;	//added September 2010
float **tisserr,**dmtissdp,*mptissref;//September 2010;
int **ivaryparams;	//added April 2015

float gtt;	//added September 2010
float fn,c,alphab,p50,cs,cext,hext,req,q0fac,totalq,flowfac=1.e6/60.;
float plow,phigh,clowfac,chighfac,pphighfac;//added January 2012
float pi1 = atan(1.)*4.,fac = 1./4./pi1;
float lb,maxl,v,vol,vdom,errfac,tlength,alx,aly,alz,lowflowcrit;
float tlengthq,tlengthqhd;//added 8/09
float xmax,ymax,scalefac;
float w2d,r2d; //needed for 2d version

//Jan. 2019 parallel region BC Flow
float outVenFlow, outVenFlux, outArtFlow, outArtFlux, outCapFlow, outCapFlux;
float inVenFlow, inVenFlux, inArtFlow, inArtFlux, inCapFlow, inCapFlux;
float outVenConc, outArtConc, outCapConc, inVenConc, inArtConc, inCapConc;
float inVenPO2, inArtPO2, inCapPO2, outVenPO2, outArtPO2, outCapPO2;
float inVendcdp, inArtdcdp, inCapdcdp, outVendcdp, outArtdcdp, outCapdcdp;
float inFlux, outFlux, diffFlux, Vsim, extraction, qeff, consumption, perfusion, HDin0, outConcErr, outConcErrMax;

float *axt,*ayt,*azt,*ds,*diff,*pmin,*pmax,*pmean,*pref,*g0,*g0fac,*g0facnew,*sumal;
float *diam,*rseg,*q,*qdata,*qq,*hd,*oxflux,*segc,*bcprfl,*bchd,*nodvar,*segvar,*qvtemp,*qvfac;//added qdata November 2016
float **start,**scos,**ax,**cnode,**resisdiam,**resis,**bcp; //added April 2010
float **qv,**qt,**pv,**pev,**pt;
float **qvseg,**pvseg,**pevseg;
float **paramvalue,*solutefac,*intravascfac,*postgreensparams,*postgreensout;	//added April 2015

float *x,*y,*lseg,*ss,*cbar,*mtiss,*mptiss,*dqvsumdg0,*dqtsumdg0;
float *epsvessel,*epstissue,*eps,*errvessel,*errtissue,*pinit,*p;
float *rhs,*rhstest,*g0old,*ptt,*ptpt,*qtsum,*qvsum;
float **pvt,**pvprev,**qvprev,**cv,**dcdp,**tissparam;
float **ptprev,**ptv,**gamma1,**qcoeff1,**cv0,**conv0;
float **gvv,**end,**al;
float ***rsta,***rend,***dtt;
float *xsl0,*xsl1,*xsl2,*clmin,*clint,*cl,**zv,***psl;
float **qtp;
double *rhstiss,*matxtiss;
double **mat,**rhsg,*rhsl,*matx;


//Needed for GPU version
int useGPU,nnvGPU,nntGPU;
double *h_x, *h_b, *h_a, *h_rs;
float *h_rst;
double *d_a, *d_x, *d_b, *d_res, *d_r, *d_rs, *d_v, *d_s, *d_t, *d_p, *d_er;
float *d_xt, *d_bt, *d_rest, *d_rt, *d_rst, *d_vt, *d_st, *d_tt, *d_pt, *d_ert;

int *h_tisspoints,*d_tisspoints;
float *pt000,*qt000,*qtp000,*pv000,*qv000,*dtt000,*h_tissxyz,*h_vessxyz;
float *d_qt000,*d_qtp000,*d_pt000,*d_qv000,*d_pv000,*d_dtt000;
float *d_tissxyz,*d_vessxyz;

char numstr[6];

int main(int argc, char *argv[])
{
	int iseg, inod, inodbc, imain, j, k, isp;
	char fname[80];
	BOOL NoOverwrite = FALSE;
	FILE *ofp;

	//copy input data files to "Current" directory
	CopyFile("SoluteParams.dat","Current\\SoluteParams.dat",NoOverwrite);
	CopyFile("IntravascRes.dat","Current\\IntravascRes.dat",NoOverwrite);
	CopyFile("ContourParams.dat","Current\\ContourParams.dat",NoOverwrite);
	CopyFile("VaryParams.dat","Current\\VaryParams.dat",NoOverwrite);
	CopyFile("network.dat","Current\\network.dat",NoOverwrite);
	CopyFile("tissrate.cpp.dat","Current\\tissrate.cpp.dat",NoOverwrite);

	input();

	is2d = 0; //set to 1 for 2d version, 0 otherwise
	if(mzz == 1) is2d = 1; //assumes 2d version if all tissue points lie in one z-plane

	setuparrays0();

	setuparrays1(nseg,nnod);

	analyzenet();

	setuparrays2(nnv,nnt);

	if(useGPU){
		nntGPU = mxx*myy*mzz;	//this is the maximum possible number of tissue points
		nnvGPU = 2000;	//start by assigning a good amount of space on GPU - may increase nnvGPU later
		bicgstabBLASDinit(nnvGPU);
		bicgstabBLASStinit(nntGPU);
		tissueGPUinit(nntGPU, nnvGPU);
	}

	for(iseg=1; iseg<=nseg; iseg++) segvar[iseg] = segname[iseg];
	for(inod=1; inod<=nnod; inod++) nodvar[inod] = nodname[inod];
	picturenetwork(nodvar,segvar,"Current\\NetNodesSegs.ps");
	//for(iseg=1; iseg<=nseg; iseg++) segvar[iseg] = fabs(diam[iseg]);
	for(iseg=1; iseg<=nseg; iseg++) 
		segvar[iseg] = log(fabs(qdata[iseg]));
	cmgui(segvar);

	ofp = fopen("Current\\summary.out","w");
	//print headings for summary output file
	fprintf(ofp,"imain kmain ");
	for(j=1; j<=nvaryparams; j++){
		switch(ivaryparams[j][1]){
		case 1:
			{
				fprintf(ofp,"   q0fac    ");
				break;
			}
		case 2:
			{
				fprintf(ofp," solutefac[%i]",ivaryparams[j][2]);
				break;
			}
		case 3:
			{
				fprintf(ofp," diff[%i]     ",ivaryparams[j][2]);
				break;
			}
		case 4:
			{
				fprintf(ofp," intravascfac[%i]",ivaryparams[j][2]);
				break;
			}
		case 5:
			{
				fprintf(ofp," tissparam[%i][%i]",ivaryparams[j][2],ivaryparams[j][3]);
				break;
			}
		}
	}
	for(isp=1; isp<=nsp; isp++) fprintf(ofp,"  pmean[%i]  ",isp);
	for(j=1; j<=npostgreensout; j++) fprintf(ofp," postgreens[%i]",j);
	fprintf(ofp,"\n");
    fclose(ofp);

	//The following loop allows running a series of cases with varying parameters
	for(imain=1; imain<=nruns; imain++){
		sprintf(numstr,"%03i",imain);	//need 3-digit frame number for file name. November 2016
		for(j=1; j<=nvaryparams; j++){
			switch(ivaryparams[j][1])
			{
			case 1:
				{
					q0fac = paramvalue[imain][j];
					break;
				}
			case 2:	
				{
					isp = ivaryparams[j][2];	//updated November 2016
					if(isp <= nsp) solutefac[isp] = paramvalue[imain][j];
					break;
				}
			case 3:
				{
					isp = ivaryparams[j][2];
					if(isp <= nsp) diff[isp] = paramvalue[imain][j];
					break;
				}
			case 4: 
				{
					isp = ivaryparams[j][2];
					if(isp <= nsp) intravascfac[isp] = paramvalue[imain][j];
					break;
				}
			case 5:
				{
					isp = ivaryparams[j][3];
					if(isp <= nsp) tissparam[ivaryparams[j][2]][isp] = paramvalue[imain][j];
					break;
				}
			}
		}

		//*************************************
		//********** Begin Main Loop **********
		//*************************************

		//******************* initial concentrations ***********************
		inArtConc = bloodconc(inArtPO2, HDin0); //0.161345;	//corresponds to 60 mmHg, does not change
		inVenConc = bloodconc(inVenPO2, HDin0); //0.145;	
		inCapConc = bloodconc(inCapPO2, HDin0); //0.140;	

		for (k=1; k<nmaxbc; k++){
        	printf("\n\n======= omain = %d ======\n\n", k); 	

			greens();
			//initialize flow/fluxes
			outVenFlow = 0, outVenFlux = 0, outArtFlow = 0, outArtFlux = 0, outCapFlow = 0, outCapFlux = 0;
			inCapFlow = 0, inVenFlow = 0, inArtFlow = 0, inCapFlux = 0, inVenFlux = 0, inArtFlux = 0;

			for (inodbc = 1; inodbc <= nnodbc; inodbc++) {
				inod = bcnod[inodbc];
				iseg = nodseg[1][inod];
            	switch(bctyp[inodbc]){
                	case 4: //inflow venule
                    	inVenFlow += qq[iseg];
        	            inVenFlux += segc[iseg]/flowfac;
        	            break;
        	        case 5: //inflow arteriole
        	            inArtFlow += qq[iseg];
            	        inArtFlux += segc[iseg]/flowfac;
        	            break;
            	    case 6: //inflow capillary
            	        inCapFlow += qq[iseg];
            	        inCapFlux += segc[iseg]/flowfac;
            	        break;
            	    case 7: //outflow venule
						outVenFlow += qq[iseg];
						outVenFlux += segc[iseg]/flowfac;
            	        break;
            	    case 8: //outflow arteriole
            	        outArtFlow += qq[iseg];
                	    outArtFlux += segc[iseg]/flowfac;
                    	break;
            	    case 9: //outflow capillary
                	    outCapFlow += qq[iseg];
                    	outCapFlux += segc[iseg]/flowfac;
						break;
                	default:
                    	printf("*** Error: boundary node %i not classified\n",inodbc);
						break;
            	}
        	}	

			// Calculate concentrations in outflowing vessels
			outVenConc = outVenFlux / outVenFlow;
			outArtConc = outArtFlux / outArtFlow;
			outCapConc = outCapFlux / outCapFlow;

			blood(inVenConc, HDin0, &inVenPO2, &inVendcdp);
			blood(inArtConc, HDin0, &inArtPO2, &inArtdcdp);
			blood(inCapConc, HDin0, &inCapPO2, &inCapdcdp);
			blood(outVenConc, HDin0, &outVenPO2, &outVendcdp);
			blood(outArtConc, HDin0, &outArtPO2, &outArtdcdp);
			blood(outCapConc, HDin0, &outCapPO2, &outCapdcdp);

			// Calculate Consumption, Extraction, and Perfusion, March 2019
			outFlux = outVenFlux + outArtFlux + outCapFlux;
			inFlux = inVenFlux + inArtFlux + inCapFlux;
			diffFlux = inFlux - outFlux;

			qeff = diffFlux / (inArtConc - outVenConc); // effective flow, units nl/min
			Vsim = vol * nnt / 1e6; // volume of simulation region, units cm^3
			consumption = diffFlux / Vsim; // oxygen consumption, units cm^3O2/100cm^3/min
			perfusion = qeff / Vsim; // perfusion, units cm^3/100cm^3/min
			extraction = diffFlux / (Vsim * inArtConc); // extraction, units

			// Generate Concfile.out
			strcpy(fname, "Current/ConcFile");
			strcat(fname, numstr);
			strcat(fname, ".out");
			ofp = fopen(fname, "a");

			fprintf(ofp, "omain =  %d\n", k);
			fprintf(ofp, "Type		Flow	Conc	Flux	PO2\n");
			fprintf(ofp, "4	In Ven	%.3f	%.3f	%.3f	%.3f\n", inVenFlow, inVenConc, inVenFlux, inVenPO2);
			fprintf(ofp, "5	In Art	%.3f	%.3f	%.3f	%.3f\n", inArtFlow, inArtConc, inArtFlux, inArtPO2);
			fprintf(ofp, "6	In Cap	%.3f	%.3f	%.3f	%.3f\n", inCapFlow, inCapConc, inCapFlux, inCapPO2);
			fprintf(ofp, "7	Out Ven	%.3f	%.3f	%.3f	%.3f\n", outVenFlow, outVenConc, outVenFlux, outVenPO2);
			fprintf(ofp, "8	Out Art	%.3f	%.3f	%.3f	%.3f\n", outArtFlow, outArtConc, outArtFlux, outArtPO2);
			fprintf(ofp, "9	Out Cap	%.3f	%.3f	%.3f	%.3f\n", outCapFlow, outCapConc, outCapFlux, outCapPO2);
			fprintf(ofp, "Delta C = %f\n", inArtConc - outVenConc);
			fprintf(ofp, "Effective Flow = %f nl/min\n", qeff);
			fprintf(ofp, "Perfusion = %f cm^3/100cm^3/min\n", perfusion);
			fprintf(ofp, "Consumption = %f cm^3O2/100cm^3/min\n", consumption);
			fprintf(ofp, "Extraction =  %f\n\n", extraction);
			fclose(ofp);

			outConcErr = FMAX(fabs(inCapConc - outCapConc), fabs(inVenConc - outVenConc));

			// Reset inflow concentrations to match computed outflow concentrations
			inCapConc = outCapConc;
			inVenConc = outVenConc;

            ofp = fopen("Current\\summary.out", "a");

			fprintf(ofp,"%4i  %4i  ",imain, kmain);
			for(j=1; j<=nvaryparams; j++) fprintf(ofp,"%12f ",paramvalue[imain][j]);
			for(isp=1; isp<=nsp; isp++) fprintf(ofp,"%12f ",pmean[isp]);

			if(npostgreensparams) postgreens();
			if(npostgreensout) for(j=1; j<=npostgreensout; j++) fprintf(ofp,"%12f ",postgreensout[j]); 
			fprintf(ofp,"\n");
			fclose(ofp);

			for(iseg=1; iseg<=nseg; iseg++) segvar[iseg] = pvseg[iseg][1];
			for(inod=1; inod<=nnod; inod++) nodvar[inod] = nodname[inod];

			strcpy(fname, "Current\\NetNodesOxygen");
			strcat(fname,numstr); 
			strcat(fname,".ps");
			picturenetwork(nodvar,segvar,fname);

			cmgui(segvar);

			strcpy(fname, "Current\\Contour");
			strcat(fname,numstr); 
			strcat(fname,".ps");
			contour(fname);

			strcpy(fname, "Current\\Histogram");
			strcat(fname,numstr); 
			strcat(fname,".out");
			histogram(fname);
		}
		
		if(useGPU){
			tissueGPUend(nntGPU, nnvGPU);
			bicgstabBLASDend(nnvGPU);
			bicgstabBLASStend(nntGPU);
		}

		if (fabs(outConcErr) < outConcErrMax) {
			fclose(ofp);
			return 0;
		}

	}
	fclose(ofp);
	return 0;
}