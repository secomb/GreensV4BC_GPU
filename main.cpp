/************************************************************************
GreensV4BC_GPU - Greens function method with unknown oxygen boundary conditions
Based on GreensV4, with contributions from Jose Celaya-Alcala and Jeffrey Lee
May 2019
***************************************************************************
Boundary classification codes (revised May 2019 to give better color codes in cmgui):
9: Inflow arteriole
8: Outflow arteriole
7: Inflow capillary
6: Outflow capillary
5: Inflow venule
4: Outflow venule
***********************************************************************/
#define _CRT_SECURE_NO_DEPRECATE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "nrutil.h"
#if defined(__linux__) 
//linux goes here
#elif defined(_WIN32)	//Windows version
#include <Windows.h>
#endif

void input(void);
void analyzenet(void);
void picturenetwork(float *nodvar, float *segvar, const char fname[]);
void greens(void);
void contour(const char fname[]);
void histogram(const char fname[]);
void setuparrays0();
void setuparrays1(int nseg, int nnod);
void setuparrays2(int nnv, int nnt);
void cmgui(float *segvar);
void postgreens(void);
void blood(float c, float h, float *p, float *pp);
float bloodconc(float p, float h);

void bicgstabBLASDinit(int nnvGPU);
void bicgstabBLASDend(int nnvGPU);
void bicgstabBLASStinit(int nntGPU);
void bicgstabBLASStend(int nntGPU);
void tissueGPUinit(int nntGPU, int nnvGPU);
void tissueGPUend(int nntGPU, int nnvGPU);

int max=100,nmaxvessel,nmaxtissue,nmax, nmaxbc, rungreens,initgreens,g0method,linmethod, is2d;
int mxx, myy, mzz, nnt, nseg, nnod, nnodfl, nnv, nsp, nnodbc, nodsegm, nsegfl, kmain, imain;
int slsegdiv,nsl1,nsl2;
int nvaryparams,nruns,ntissparams,npostgreensparams,npostgreensout;	//needed for varying parameters, postgreens
int *mainseg,*permsolute,*nodrank,*nodtyp,*nodout,*bcnodname,*bcnod,*bctyp,*lowflow;
int *nodname,*segname,*segtyp,*nspoint,*istart,*nl,*nk,*indx,*ista,*iend;
int *errvesselcount,*errtissuecount;
int *imaxerrvessel,*imaxerrtissue,*nresis;  //added April 2010
int *oxygen,*diffsolute; //added April 2010
int **segnodname,**nodseg,**tisspoints,**nodnod;
int ***nbou;

int **ivaryparams;	//added April 2015

float gtt, fn,c,alphab,p50,cs,cext,hext,req,q0fac,totalq,flowfac=1.e6/60.;
float plow,phigh,clowfac,chighfac,pphighfac;
float pi1 = atan(1.)*4.,fac = 1./4./pi1;
float lb,maxl,v,vol,vdom,errfac,tlength,alx,aly,alz,lowflowcrit;
float tlengthq,tlengthqhd, xmax,ymax,scalefac, w2d,r2d;
float outVenFlow, outVenFlux, outArtFlow, outArtFlux, outCapFlow, outCapFlux;
float inVenFlow, inVenFlux, inArtFlow, inArtFlux, inCapFlow, inCapFlux;
float outVenConc, outArtConc, outCapConc, inVenConc, inArtConc, inCapConc, inArteryConc;
float inArteryPO2, inVenPO2, inArtPO2, inCapPO2, outVenPO2, outArtPO2, outCapPO2;
float inVendcdp, inArtdcdp, inCapdcdp, outVendcdp, outArtdcdp, outCapdcdp;
float inFlux, outFlux, diffFlux, Vsim, extraction, QeffS, consumption, perfusionS, HDin0, outConcErr, outConcErrMax;
float Vups = 0., Vpar = 0., inArteryFlow, perfusionA, inArtConcNew, inArteryFlux;

float *axt,*ayt,*azt,*ds,*diff,*pmin,*pmax,*pmeant, *pmeanv, *psdt, *psdv, *pref,*g0,*g0fac,*g0facnew,*sumal, *dtmin;
float *diam,*rseg,*q,*qdata,*qq,*hd,*oxflux,*segc,*bcprfl,*bchd,*nodvar,*segvar,*qvtemp,*qvfac;
float *x,*y,*lseg,*ss,*cbar,*mtiss,*mptiss,*dqvsumdg0,*dqtsumdg0;
float *epsvessel,*epstissue,*eps,*errvessel,*errtissue,*pinit,*p;
float *rhs,*rhstest,*g0old,*ptt,*ptpt,*qtsum,*qvsum, *xsl0,*xsl1,*xsl2,*clmin,*clint,*cl;
float **start,**scos,**ax,**cnode,**resisdiam,**resis,**bcp, **qv,**qt,**pv,**pev,**pt, **qvseg,**pvseg,**pevseg;
float **paramvalue,*solutefac,*intravascfac,*postgreensparams,*postgreensout;
float **pvt,**pvprev,**qvprev,**cv,**dcdp,**tissparam;
float **ptprev, **ptv, **gamma1, **cv0, **conv0, **gvv,**end,**al,**zv;
float ***rsta,***rend,***dtt,***psl;
double **mat,**rhsg,*rhsl,*matx;

//Needed for GPU version
int useGPU,nnvGPU,nntGPU;
int *h_tisspoints,*d_tisspoints;
double *h_x, *h_b, *h_a, *h_rs;
float *h_rst;
double *d_a, *d_x, *d_b, *d_res, *d_r, *d_rs, *d_v, *d_s, *d_t, *d_p, *d_er;
float *d_xt, *d_bt, *d_rest, *d_rt, *d_rst, *d_vt, *d_st, *d_tt, *d_pt, *d_ert;
float *pt000,*qt000,*qtp000,*pv000,*qv000,*dtt000,*h_tissxyz,*h_vessxyz;
float *d_qt000,*d_qtp000,*d_pt000,*d_qv000,*d_pv000,*d_dtt000;
float *d_tissxyz,*d_vessxyz;

int main(int argc, char *argv[])
{
	int iseg, inod, inodbc, j, kbcs, isp;
	char fname[80];
	FILE *ofp;

#if defined(__linux__) 
	//linux code goes here
#elif defined(_WIN32) 			//Windows version
	bool NoOverwrite = false;
	DWORD ftyp = GetFileAttributesA("Current\\");
	if (ftyp != FILE_ATTRIBUTE_DIRECTORY) system("mkdir Current");		//Create a Current subdirectory if it does not already exist.
	CopyFile("BCparams.dat", "Current\\BCparams.dat", NoOverwrite); //copy input data files to "Current" directory
	CopyFile("ContourParams.dat.dat", "Current\\ContourParams.dat", NoOverwrite);
	CopyFile("SoluteParamsParams.dat.dat", "Current\\SoluteParamsParams.dat", NoOverwrite);
	CopyFile("Network.dat", "Current\\Network.dat", NoOverwrite);
	CopyFile("IntravascRes.dat","Current\\IntravascRes.dat",NoOverwrite);
	CopyFile("tissrate.cpp.dat","Current\\tissrate.cpp.dat",NoOverwrite);
	ftyp = GetFileAttributes("Varyparams.dat");
	if (ftyp != 0xFFFFFFFF) CopyFile("Varyparams.dat", "Current\\Varyparams.dat", NoOverwrite);	//this file may not exist!
	ftyp = GetFileAttributesA("Varyparams.dat");
#endif

	input();
	inArtConc = bloodconc(inArtPO2, HDin0);	//initial concentrations
	inVenConc = bloodconc(inVenPO2, HDin0);
	inCapConc = bloodconc(inCapPO2, HDin0);

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
	picturenetwork(nodvar, segvar, "Current/NetNodesSegs.ps");

	for (iseg = 1; iseg <= nseg; iseg++) {
		if (segtyp[iseg] == 4 || segtyp[iseg] == 5) segvar[iseg] = log(fabs(qdata[iseg]));
		else segvar[iseg] = 0.;
	}
	cmgui(segvar);
	picturenetwork(nodvar, segvar, "Current/FlowRates.ps");

	ofp = fopen("Current/summary.out", "w");	//summary output file
	fprintf(ofp,"Summary file\n");
	fclose(ofp);

	//////////////////////////////////////////////////////////////////////////
	//Run a series of cases with varying parameters
	//imain = 1 defines volume of upstream region, held constant for imain > 1
	///////////////////////////////////////////////////////////////////////////
	for(imain=1; imain<=nruns; imain++){
		for(j=1; j<=nvaryparams; j++){
			switch (ivaryparams[j][1]) {
			case 1:
				q0fac = paramvalue[imain][j];
				break;
			case 2:	
				isp = ivaryparams[j][2];
				if(isp <= nsp) solutefac[isp] = paramvalue[imain][j];
				break;
			case 3:
				isp = ivaryparams[j][2];
				if(isp <= nsp) diff[isp] = paramvalue[imain][j];
				break;
			case 4: 
				isp = ivaryparams[j][2];
				if(isp <= nsp) intravascfac[isp] = paramvalue[imain][j];
				break;
			case 5:
				isp = ivaryparams[j][3];
				if(isp <= nsp) tissparam[ivaryparams[j][2]][isp] = paramvalue[imain][j];
				break;
			case 6:
				p50 = paramvalue[imain][j];
				break;
			case 7:
				inArteryPO2 = paramvalue[imain][j];
			}
		}
		inArteryConc = bloodconc(inArteryPO2, HDin0);

		sprintf(fname, "Current/ConcFile%03i.out", imain);
		ofp = fopen(fname, "w");
		fprintf(ofp, "imain = %i\n", imain);
		fclose(ofp);

		for (kbcs = 1; kbcs <= nmaxbc; kbcs++) {			// Iterate to convergence of boundary conditions 
			printf("\n\n======= kbcs = %d ======\n\n", kbcs);
			////////////
			greens();
			///////////
			outVenFlow = 0, outVenFlux = 0, outArtFlow = 0, outArtFlux = 0, outCapFlow = 0, outCapFlux = 0;
			inCapFlow = 0, inVenFlow = 0, inArtFlow = 0, inCapFlux = 0, inVenFlux = 0, inArtFlux = 0;

			for (inodbc = 1; inodbc <= nnodbc; inodbc++) {	//calculate boundary fluxes and concentrations
				inod = bcnod[inodbc];
				iseg = nodseg[1][inod];
				switch(bctyp[inodbc]){
				case 4: //Outflow venule
					outVenFlow += qq[iseg];
					outVenFlux += segc[iseg] / flowfac;
					break;
				case 5: //Inflow venule
					inVenFlow += qq[iseg];
					inVenFlux += segc[iseg] / flowfac;
					break;
				case 6: //Outflow capillary
					outCapFlow += qq[iseg];
					outCapFlux += segc[iseg] / flowfac;
					break;
				case 7: //Inflow capillary
					inCapFlow += qq[iseg];
					inCapFlux += segc[iseg] / flowfac;
					break;
				case 8: //Outflow arteriole
					outArtFlow += qq[iseg];
					outArtFlux += segc[iseg] / flowfac;
					break;
				case 9: //Inflow arteriole
					inArtFlow += qq[iseg];
					inArtFlux += segc[iseg] / flowfac;
					break;
				default:
					printf("*** Error: boundary node %i not classified\n",inodbc);
					break;
				}
			}

			inArteryFlow = FMAX(inArtFlow, outVenFlow);
			inArteryFlux = inArteryConc * inArteryFlow;

			outVenConc = outVenFlux / outVenFlow;
			outArtConc = outArtFlux / outArtFlow;
			outCapConc = outCapFlux / outCapFlow;

			blood(inVenConc, HDin0, &inVenPO2, &inVendcdp);
			blood(inArtConc, HDin0, &inArtPO2, &inArtdcdp);
			blood(inCapConc, HDin0, &inCapPO2, &inCapdcdp);
			blood(outVenConc, HDin0, &outVenPO2, &outVendcdp);
			blood(outArtConc, HDin0, &outArtPO2, &outArtdcdp);
			blood(outCapConc, HDin0, &outCapPO2, &outCapdcdp);

			ofp = fopen(fname, "a");
			fprintf(ofp, "kbcs = %i\n", kbcs);
			fprintf(ofp, "Type		Flow	Conc	Flux	PO2\n");
			fprintf(ofp, "In Artery	%.3f	%.3f	%.3f	%.3f\n", inArteryFlow, inArteryConc, inArteryFlux, inArteryPO2);
			fprintf(ofp, "9	In Art	%.3f	%.3f	%.3f	%.3f\n", inArtFlow, inArtConc, inArtFlux, inArtPO2);
			fprintf(ofp, "8	Out Art	%.3f	%.3f	%.3f	%.3f\n", outArtFlow, outArtConc, outArtFlux, outArtPO2);
			fprintf(ofp, "7	In Cap	%.3f	%.3f	%.3f	%.3f\n", inCapFlow, inCapConc, inCapFlux, inCapPO2);
			fprintf(ofp, "6	Out Cap	%.3f	%.3f	%.3f	%.3f\n", outCapFlow, outCapConc, outCapFlux, outCapPO2);
			fprintf(ofp, "5	In Ven	%.3f	%.3f	%.3f	%.3f\n", inVenFlow, inVenConc, inVenFlux, inVenPO2);
			fprintf(ofp, "4	Out Ven	%.3f	%.3f	%.3f	%.3f\n", outVenFlow, outVenConc, outVenFlux, outVenPO2);
			fclose(ofp);

			// Calculate consumption, extraction and perfusion
			outFlux = outVenFlux + outArtFlux + outCapFlux;
			inFlux = inVenFlux + inArtFlux + inCapFlux;
			diffFlux = inFlux - outFlux;					// VO2S, nl/min
			QeffS = diffFlux / (inArteryConc - outVenConc); // effective flow, nl/min
			Vsim = vol * nnt / 1e6;							// volume of simulation region, nl
			consumption = diffFlux / Vsim;					// oxygen consumption, cm^3O2/cm^3/min
			perfusionS = QeffS / Vsim;						// perfusion, cm^3/cm^3/min
			extraction = 1. - outVenConc / inArteryConc;	// extraction, no units
			Vpar = inArteryFlow * (inArtConc - outVenConc) / consumption - Vsim;	// volume of parallel region, nl
			if(imain == 1) Vups = inArteryFlow * (inArteryConc - inArtConc) / consumption; //compute volume of upstream region for imain = 1, nl
			perfusionA = inArteryFlow/(Vsim + Vups + Vpar);	// perfusion, cm^3/cm^3/min, alternate calculation

			outConcErr = FMAX(fabs(inCapConc - outCapConc), fabs(inVenConc - outVenConc));
			if(imain > 1){							//keep upstream region volume fixed, calculate new inArtConc
				inArtConcNew = inArteryConc - Vups * consumption / inArteryFlow; 
				outConcErr = FMAX(outConcErr, fabs(inArtConcNew - inArtConc));
			}
			if (fabs(outConcErr) < outConcErrMax) goto bcs_converged;

			// Reset inflow concentrations to match computed outflow concentrations
			inCapConc += 1.5*(outCapConc - inCapConc);	//overrelax here to speed convergence
			inVenConc += 1.5*(outVenConc - inVenConc);			
			if(imain > 1) inArtConc = inArtConcNew;
		}
		printf("***Warning: Boundary conditions not converged\n");
bcs_converged:;											//boundary conditions converged

		ofp = fopen("Current/summary.out", "a");
		fprintf(ofp,"\nimain = %4i, kmain = %4i \n",imain, kmain);
		for(j=1; j<=nvaryparams; j++){
			switch(ivaryparams[j][1]){
			case 1:
				fprintf(ofp,"q0fac = %12f\n", paramvalue[imain][j]);
				break;
			case 2:
				fprintf(ofp,"solutefac[%i] = %12f\n",ivaryparams[j][2], paramvalue[imain][j]);
				break;
			case 3:
				fprintf(ofp,"diff[%i] = %12f\n",ivaryparams[j][2], paramvalue[imain][j]);
				break;
			case 4:
				fprintf(ofp,"intravascfac[%i] = %12f\n",ivaryparams[j][2], paramvalue[imain][j]);
				break;
			case 5:
				fprintf(ofp,"tissparam[%i][%i] = %12f\n",ivaryparams[j][2],ivaryparams[j][3], paramvalue[imain][j]);
				break;
			case 6:
				fprintf(ofp, "p50 = %12f\n", paramvalue[imain][j]);
				break;
			case 7:
				fprintf(ofp, "inArteryPO2 = %12f\n", paramvalue[imain][j]);
				break;
			}
		}

		fprintf(ofp, "Type		Flow	Conc	Flux	PO2\n");
		fprintf(ofp, "In Artery	%.3f	%.3f	%.3f	%.3f\n", inArteryFlow, inArteryConc, inArteryFlux, inArteryPO2);
		fprintf(ofp, "9	In Art	%.3f	%.3f	%.3f	%.3f\n", inArtFlow, inArtConc, inArtFlux, inArtPO2);
		fprintf(ofp, "8	Out Art	%.3f	%.3f	%.3f	%.3f\n", outArtFlow, outArtConc, outArtFlux, outArtPO2);
		fprintf(ofp, "7	In Cap	%.3f	%.3f	%.3f	%.3f\n", inCapFlow, inCapConc, inCapFlux, inCapPO2);
		fprintf(ofp, "6	Out Cap	%.3f	%.3f	%.3f	%.3f\n", outCapFlow, outCapConc, outCapFlux, outCapPO2);
		fprintf(ofp, "5	In Ven	%.3f	%.3f	%.3f	%.3f\n", inVenFlow, inVenConc, inVenFlux, inVenPO2);
		fprintf(ofp, "4	Out Ven	%.3f	%.3f	%.3f	%.3f\n", outVenFlow, outVenConc, outVenFlux, outVenPO2);
		fprintf(ofp, "Simulation volume = %.3f nl\n", Vsim);
		fprintf(ofp, "Upstream volume = %.3f nl\n", Vups);
		fprintf(ofp, "Parallel volume = %.3f nl\n", Vpar);
		fprintf(ofp, "Effective flow (S) = %.3f nl/min\n", QeffS);
		fprintf(ofp, "Perfusion (S) = %f cm^3/cm^3/min\n", perfusionS);
		fprintf(ofp, "Perfusion (A) = %f cm^3/cm^3/min\n", perfusionA);
		fprintf(ofp, "Consumption (S) = %f cm^3O2/cm^3/min\n", consumption);
		fprintf(ofp, "Vessel PO2 (mean +- sd) = %.3f +- %.3f\n", pmeanv[1], psdv[1]);
		fprintf(ofp, "Tissue PO2 (mean +- sd) = %.3f +- %.3f\n", pmeant[1], psdt[1]);
		fprintf(ofp, "Delta C = %f\n", inArteryConc - outVenConc);
		fprintf(ofp, "Extraction = %f\n\n", extraction);
		if(npostgreensparams){
			postgreens();
			for(j=1; j<=npostgreensout; j++) fprintf(ofp,"%12f ",postgreensout[j]); 
			fprintf(ofp,"\n");
		}
		fclose(ofp);

		for (iseg = 1; iseg <= nseg; iseg++) if (segtyp[iseg] == 4 || segtyp[iseg] == 5) segvar[iseg] = pvseg[iseg][1];
		for(inod=1; inod<=nnod; inod++) nodvar[inod] = nodname[inod];
		sprintf(fname, "Current/NetNodesOxygen%03i.ps", imain);
		picturenetwork(nodvar,segvar,fname);
		cmgui(segvar);

		sprintf(fname, "Current/Contour%03i.ps", imain);
		contour(fname);

		sprintf(fname, "Current/Histogram%03i.out", imain);
		histogram(fname);
	}

	if(useGPU){
		tissueGPUend(nntGPU, nnvGPU);
		bicgstabBLASDend(nnvGPU);
		bicgstabBLASStend(nntGPU);
	}
	return 0;
}