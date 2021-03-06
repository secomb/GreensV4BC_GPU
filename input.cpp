/******************************************************
input - reads input files.  TWS January 08
Note that input files refer to segname and nodname, but
all arrays use iseg and inod, which are assigned when
reading the file, as indices.
Note use of format "%*[^\n]" to read comment text of
unknown length.  From Tuhin Roy, Nov. 08.
Version 2.0, May 1, 2010.
Version 3.0, May 17, 2011.
Version 4.0, March 1, 2018.
In soluteparams.dat, total inflow is replaced by flow factor in GreensV4
Includes capability to vary up to 3 parameters. Allowable parameters to vary:
1: q0fac, 2: solutefac[isp], 3: diff[isp], 4: intravascfac[isp], 5: tissparam[i][isp], 6: p50
Values stored in ivaryparams[i][1], with i and or isp stored in ivaryparams[i][2 or 3]
*******************************************************/
#define _CRT_SECURE_NO_DEPRECATE
#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "nrutil.h"

#include <string>

int match(std::string const &pat, std::string const &target) {	//string matching function, this includes wild card '?'
	int pos,start,max;
	if (pat.size() > target.size()) return -1;
	max = target.size() - pat.size() + 1;
	for(start=0; start<max; start++){
		for(pos=0; pos<pat.size(); pos++) if(pat[pos] != '?' && pat[pos] != target[start + pos]) break;
		if(pos == pat.size()) return start;
	}
	return -1;
}

void input(void)
{
	extern int max, nmaxvessel, nmaxtissue, nmax, nsl1, nsl2, slsegdiv, nmaxbc, useGPU;
	extern int mxx,myy,mzz,nnt,nnv,nseg,nnod,nsp,nnodbc,nodsegm;
	extern int rungreens, g0method, linmethod, is2d;
	extern int nvaryparams, nruns, ntissparams, npostgreensparams, npostgreensout;
	extern int **tisspoints,*permsolute,*bcnodname,*bcnod,*bctyp,**ivaryparams;
	extern int *segname, *segtyp, *nodname, *nspoint, *nl, *oxygen, *diffsolute, *nresis;
	extern int **segnodname,**nodseg;

	extern float fn, alphab, p50, cs, q0fac, errfac, lowflowcrit;
	extern float plow, phigh, clowfac, chighfac, pphighfac;
	extern float lb, maxl, v, vol, req, pi1, alx, aly, alz, xmax, ymax, scalefac;
	extern float inArteryPO2, inArtPO2, inCapPO2, inVenPO2, HDin0, outConcErrMax; //added April 2019
	extern float *axt,*ayt,*azt,*ds,*g0,*diff,*pref,*g0fac;
	extern float *diam, *qdata, *hd, *bcprfl, *bchd;
	extern float *x,*xsl0,*xsl1,*xsl2,*clmin,*clint,*p,*cl,**zv,***psl;
	extern float **start,**scos,**ax,**cnode,**bcp,**resisdiam,**resis,**tissparam;
	extern float **paramvalue,*solutefac,*intravascfac,*postgreensparams,*postgreensout;

	int i,j,iseg,isp,nlmax,idum;
	FILE *ifp;
	char bb[100];

	ifp = fopen("SoluteParams.dat", "r");
	fgets(bb,max,ifp);
	printf("%s\n",bb);
	fscanf(ifp,"%i %i %i %i%*[^\n]", &rungreens,&g0method,&linmethod,&useGPU);
	if(g0method != 1 && g0method != 2) printf("*** Error: soluteparams.dat, g0method must be 1 or 2\n");
	if(linmethod < 1 || linmethod > 3) printf("*** Error: soluteparams.dat, linmethod must be 1, 2 or 3\n");
	fscanf(ifp,"%i %i %i%*[^\n]", &nmaxvessel,&nmaxtissue,&nmax);
	fscanf(ifp,"%f%*[^\n]", &errfac);
	fscanf(ifp,"%f%*[^\n]", &lowflowcrit);
	fscanf(ifp,"%f%*[^\n]", &p50);
	fscanf(ifp,"%f%*[^\n]", &fn);
	fscanf(ifp,"%f%*[^\n]", &cs);
	fscanf(ifp,"%f%*[^\n]", &alphab);
	fscanf(ifp, "%f%*[^\n]", &q0fac);
	fscanf(ifp,"%i %i%*[^\n]", &nsp,&ntissparams);
	permsolute = ivector(1,nsp);
	diffsolute = ivector(1,nsp);
	oxygen = ivector(1,nsp);
	pref = vector(1,nsp);
	diff = vector(1,nsp);
	g0 = vector(1,nsp);
	g0fac = vector(1,nsp);
	tissparam = matrix(1,ntissparams,1,nsp);
	for(isp=1; isp<=nsp; isp++){
		fgets(bb,max,ifp);
		fgets(bb,max,ifp);
		printf("%s\n",bb);
		fscanf(ifp,"%i %i %i%*[^\n]", &permsolute[isp],&diffsolute[isp],&oxygen[isp]);
		if(diffsolute[isp] != 0 && diffsolute[isp] != 1) printf("*** Error: soluteparams.dat, diffsolute[isp] must be 0 or 1\n");
		if(oxygen[isp] != 0 && oxygen[isp] != 1) printf("*** Error: soluteparams.dat, oxygen[isp] must be 0 or 1\n");
		if(oxygen[isp] == 1) permsolute[isp] = 1; //oxygen is permeable
		if(permsolute[isp] == 1) diffsolute[isp] = 1;  //permeable solutes must be diffusible
		fscanf(ifp,"%f%*[^\n]", &pref[isp]);
		fscanf(ifp,"%f%*[^\n]", &diff[isp]);
		diff[isp] = diff[isp]*1.e8;
		for(i=1; i<=ntissparams; i++) fscanf(ifp,"%f%*[^\n]", &tissparam[i][isp]);
		fscanf(ifp,"%f%*[^\n]", &g0[isp]);
		fscanf(ifp,"%f%*[^\n]", &g0fac[isp]);
	}
	fclose(ifp);

	plow = 0.1*p50;		//parameters for function "blood"
	phigh = 5.*p50;
	clowfac = cs*(1.0 - 1.0/(1.0 + pow((plow/p50),fn)));
	chighfac = cs*(1.0 - 1.0/(1.0 + pow((phigh/p50),fn)));
	pphighfac = cs*fn/p50*pow(phigh/p50,(fn-1))/SQR(1. + pow(phigh/p50,fn));

	nvaryparams = 0;	//default if no VaryParams.dat is present
	nruns = 1;
	solutefac = vector(1,nsp);
	intravascfac = vector(1,nsp);
	for(isp=1; isp<=nsp; isp++){
		solutefac[isp] = 1.;
		intravascfac[isp] = 1.;
	}
	if(ifp = fopen("VaryParams.dat", "r")){
		fgets(bb,max,ifp);
		fgets(bb,max,ifp);
		fgets(bb,max,ifp);
		fscanf(ifp,"%i%*[^\n]", &nvaryparams);
		if(nvaryparams){
			ivaryparams = imatrix(1,nvaryparams,1,3);
			fgets(bb,max,ifp);
			for(i=1; i<=nvaryparams; i++){
				for(j=1; j<=3; j++) ivaryparams[i][j] = 0;
				fgets(bb,max,ifp);
				if(match("q0fac", bb) >= 0){
					ivaryparams[i][1] = 1;
					printf("Variable parameter %i: q0fac\n",i);
				}
				else if(match("solutefac", bb) >= 0){
					j = match("solutefac", bb);
					ivaryparams[i][1] = 2;
					ivaryparams[i][2] = bb[j+10] - '0';
					printf("Variable parameter %i: solutefac[%i]\n",i,ivaryparams[i][2]);
				}
				else if(match("diff", bb) >= 0){
					j = match("diff", bb);
					ivaryparams[i][1] = 3;
					ivaryparams[i][2] = bb[j+5] - '0';
					printf("Variable parameter %i: diff[%i]\n",i,ivaryparams[i][2]);
				}
				else if(match("intravascfac", bb) >= 0){
					j = match("intravascfac", bb);
					ivaryparams[i][1] = 4;
					ivaryparams[i][2] = bb[j+13] - '0';
					printf("Variable parameter %i: intravascfac[%i]\n",i,ivaryparams[i][2]);
				}
				else if(match("tissparam", bb) >= 0){
					j = match("tissparam", bb);
					ivaryparams[i][1] = 5;
					ivaryparams[i][2] = bb[j+10] - '0';
					ivaryparams[i][3] = bb[j+13] - '0';
					printf("Variable parameter %i: tissparams[%i][%i]\n",i,ivaryparams[i][2],ivaryparams[i][3]);
				}
				else if (match("p50", bb) >= 0) {
					ivaryparams[i][1] = 6;
					printf("Variable parameter %i: p50\n", i);
				}
				else if (match("inArteryPO2", bb) >= 0) {
					ivaryparams[i][1] = 7;
					printf("Variable parameter %i: inArteryPO2\n", i);
				
				}
			}
			fscanf(ifp,"%i%*[^\n]", &nruns);
			paramvalue = matrix(1,nruns,1,nvaryparams);
			fgets(bb,max,ifp);
			fgets(bb,max,ifp);
			for(i=1; i<=nruns; i++){
				fscanf(ifp, "%i", &idum);
				for(j=1; j<=nvaryparams; j++) fscanf(ifp, "%f", &paramvalue[i][j]);
				fscanf(ifp,"%*[^\n]");
			}
		}
		fclose(ifp);
	} 

	resisdiam = matrix(1, 20, 1, nsp);		//intravascular oxygen resistance data.  Assume zero unless specified in data file.
	resis = matrix(1, 20, 1, nsp);
	nresis = ivector(1, nsp);
	ifp = fopen("IntravascRes.dat", "r");
	for(isp=1; isp<=nsp; isp++){
		fscanf(ifp, "%i", &nresis[isp]);
		if(nresis[isp] > 20) printf("Error: too many points in IntravascRes.dat, nresis = %i > 20\n",nresis[isp]);
		fgets(bb,max,ifp);
		if(nresis[isp] > 0){
			fgets(bb,max,ifp);
			for(i=1; i<=nresis[isp]; i++) fscanf(ifp,"%f %f", &resisdiam[i][isp],&resis[i][isp]);
		}
	}
	fclose(ifp);

	ifp = fopen("Network.dat", "r");		//network data file
	fgets(bb,max,ifp);
	printf("%s\n",bb);
	fscanf(ifp, "%f %f %f%*[^\n]", &alx, &aly, &alz);		//dimensions of box in microns; vertex must be at origin
	fscanf(ifp,"%i %i %i%*[^\n]", &mxx,&myy,&mzz);
	fscanf(ifp,"%f%*[^\n]", &lb);
	fscanf(ifp,"%f%*[^\n]", &maxl);
	fscanf(ifp,"%i%*[^\n]", &nodsegm);
	fscanf(ifp, "%i%*[^\n]", &nseg);		//number of segments in vessel network
	fgets(bb,max,ifp);
	fgets(bb,max,ifp);
	segname = ivector(1,nseg);
	segtyp = ivector(1,nseg);
	segnodname = imatrix(1,2,1,nseg);
	diam = vector(1,nseg);
	qdata = vector(1, nseg);
	hd = vector(1,nseg);
	for(iseg=1; iseg<=nseg; iseg++) fscanf(ifp, "%i %i %i %i %f %f %f%*[^\n]",
		&segname[iseg],&segtyp[iseg],&segnodname[1][iseg],&segnodname[2][iseg],&diam[iseg],&qdata[iseg],&hd[iseg]);
	fscanf(ifp, "%i%*[^\n]", &nnod);		//number of nodes in vessel network
	fgets(bb,max,ifp);
	fgets(bb,max,ifp);
	nodname = ivector(1,nnod);
	cnode = matrix(1, 3, 1, nnod);			//coordinates of nodes
	for(i=1; i<=nnod; i++) fscanf(ifp, "%i %f %f %f%*[^\n]", &nodname[i],&cnode[1][i],&cnode[2][i],&cnode[3][i]);
	fscanf(ifp, "%i%*[^\n]", &nnodbc);		//boundary nodes
	fgets(bb,max,ifp);
	fgets(bb,max,ifp);
	bcnodname = ivector(1,nnodbc);
	bcnod = ivector(1,nnodbc);
	bctyp = ivector(1,nnodbc);
	bcprfl = vector(1,nnodbc);
	bchd = vector(1,nnodbc);
	bcp = matrix(1,nnodbc,1,nsp);
	for(i=1; i<=nnodbc; i++){
		fscanf(ifp, "%i %i %f %f", &bcnodname[i], &bctyp[i], &bcprfl[i], &bchd[i]);
		for(isp=1; isp<=nsp; isp++) if(permsolute[isp] == 1) fscanf(ifp,"%f",&bcp[i][isp]);
		fscanf(ifp,"%*[^\n]");	//ignore any 'extra' solutes in data file
	}
	fclose(ifp);

	v = alx * aly*alz;						//v = total box volume,
	vol = v / (mxx*myy*mzz);				//vol = volume represented by each tissue point; 
	if(mzz == 1) req = pow(vol*1./alz/pi1,0.5);//2d version
	else req = pow(vol*0.75 / pi1, 0.333333);	//req = radius of equivalent sphere
	nl = ivector(1,nsp);
	xsl0 = vector(1,3);
	xsl1 = vector(1,3);
	xsl2 = vector(1,3);
	clmin = vector(1,nsp);
	clint = vector(1,nsp);

	ifp = fopen("ContourParams.dat", "r");	//Read parameters for slice on which P is computed for contour plot
	fscanf(ifp, "%f %f %f %i%*[^\n]", &xsl0[1],&xsl0[2],&xsl0[3],&slsegdiv);
	fscanf(ifp, "%f %f %f %i%*[^\n]", &xsl1[1],&xsl1[2],&xsl1[3],&nsl1);
	fscanf(ifp, "%f %f %f %i%*[^\n]", &xsl2[1],&xsl2[2],&xsl2[3],&nsl2);
	nlmax = 1;
	for(isp=1; isp<=nsp; isp++){
		fscanf(ifp, "%f %f %i%*[^\n]", &clmin[isp],&clint[isp],&nl[isp]);
		if(nl[isp] > nlmax) nlmax = nl[isp];
	}
	fclose(ifp);
	xmax = sqrt(SQR(xsl1[1]-xsl0[1]) + SQR(xsl1[2]-xsl0[2]) + SQR(xsl1[3]-xsl0[3]));
	ymax = sqrt(SQR(xsl2[1]-xsl0[1]) + SQR(xsl2[2]-xsl0[2]) + SQR(xsl2[3]-xsl0[3]));
	scalefac = FMIN(500. / xmax, 700. / ymax);
	cl = vector(1,nlmax);
	zv = matrix(1,nsl1,1,nsl2);
	psl = f3tensor(1,nsl1,1,nsl2,1,nsp);

	ifp = fopen("BCparams.dat", "r"); 
	fscanf(ifp, "%i%*[^\n]", &nmaxbc);
	fscanf(ifp, "%f%*[^\n]", &inArteryPO2);
	fscanf(ifp, "%f%*[^\n]", &inArtPO2);
	fscanf(ifp, "%f%*[^\n]", &inVenPO2);
	fscanf(ifp, "%f%*[^\n]", &inCapPO2);
	fscanf(ifp, "%f%*[^\n]", &HDin0);
	fscanf(ifp, "%f%*[^\n]", &outConcErrMax);
	fclose(ifp);

	npostgreensparams = 0;
	if (ifp = fopen("PostGreensParams.dat", "r")) {		//Read parameters to run postgreens
		fgets(bb,max,ifp);
		fscanf(ifp,"%i%*[^\n]", &npostgreensout);
		fscanf(ifp,"%i%*[^\n]", &npostgreensparams);
		if(npostgreensout) postgreensout = vector(1,npostgreensout);
		if(npostgreensparams){
			postgreensparams = vector(1,npostgreensparams);
			for(i=1; i<=npostgreensparams; i++) fscanf(ifp,"%f%*[^\n]", &postgreensparams[i]);
		}
	}
}

