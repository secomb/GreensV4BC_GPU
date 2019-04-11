/**************************************************************************
tissueGPUcopy
copy dtt and tisspoints matrices to GPU
TWS, December 2011
**************************************************************************/
#include <cutil_inline.h>
#include "nrutil.h"

void tissueGPUcopy()
{
	extern int mxx,myy,mzz,nntGPU,nnt,nnv,useGPU;
	extern int **tisspoints,*h_tisspoints,*d_tisspoints;
	extern float *dtt000,*d_dtt000,***dtt;
	extern float *axt,*ayt,*azt,**ax;
	extern float *h_tissxyz,*d_tissxyz,*h_vessxyz,*d_vessxyz;

	int jx,jy,jz,jtp,ivp;

	for(jtp=0; jtp<nnt; jtp++){
		h_tisspoints[jtp] = tisspoints[1][jtp+1];
		h_tisspoints[jtp+nnt] = tisspoints[2][jtp+1]*mxx;//multiply here to save multiplication in kernel
		h_tisspoints[jtp+2*nnt] = tisspoints[3][jtp+1]*mxx*myy;
		h_tissxyz[jtp] = axt[tisspoints[1][jtp+1]];
		h_tissxyz[jtp+nnt] = ayt[tisspoints[2][jtp+1]];
		h_tissxyz[jtp+2*nnt] = azt[tisspoints[3][jtp+1]];
	}
	for(ivp=0; ivp<nnv; ivp++){
		h_vessxyz[ivp] = ax[1][ivp+1];
		h_vessxyz[ivp+nnv] = ax[2][ivp+1];
		h_vessxyz[ivp+2*nnv] = ax[3][ivp+1];
	}
	for(jx=1; jx<=mxx; jx++) for(jy=1; jy<=myy; jy++) for(jz=1; jz<=mzz; jz++)
		dtt000[(jx-1) + (jy-1)*mxx + (jz-1)*mxx*myy] = dtt[jx][jy][jz];

	cudaSetDevice( useGPU-1 );
	cudaMemcpy(d_tisspoints, h_tisspoints, 3*nnt*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tissxyz, h_tissxyz, 3*nnt*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vessxyz, h_vessxyz, 3*nnv*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dtt000, dtt000, nntGPU*sizeof(float), cudaMemcpyHostToDevice);
}