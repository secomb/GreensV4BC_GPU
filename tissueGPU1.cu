/***********************************************************
tissueGPU1.cu
GPU kernel to accumulate contributions of tissue source
strengths qt to tissue solute levels pt.
Each tissue point is assigned one or more threads: step is the number of threads
This spreads it over more threads.
TWS December 2011
************************************************************/
#include <stdio.h>
#include <cutil_inline.h>


__global__ void tissueGPU1Kernel(int *d_tisspoints, float *d_dtt000, float *d_pt000, float *d_qt000, int nnt, int step)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	int itp = i/step;
    int itp1 = i%step;
	int jtp,ixyz,ix,iy,iz,jx,jy,jz,nnt2=2*nnt,istep;
	float p = 0.;
    if(itp < nnt){
		ix = d_tisspoints[itp];
		iy = d_tisspoints[itp+nnt];
		iz = d_tisspoints[itp+nnt2];
		for(jtp=itp1; jtp<nnt; jtp+=step){
			jx = d_tisspoints[jtp];
			jy = d_tisspoints[jtp+nnt];
			jz = d_tisspoints[jtp+nnt2];
			ixyz = abs(jx-ix) + abs(jy-iy) + abs(jz-iz);
			p += d_qt000[jtp]*d_dtt000[ixyz];
		}
		if(itp1 == 0) d_pt000[itp] = p;
	}
	//The following is apparently needed to assure that d_pt000 is incremented in sequence from the needed threads
	for(istep=1; istep<step; istep++){
		__syncthreads();
		if(itp1 == istep && itp < nnt) d_pt000[itp] += p;
	}
}

extern "C" void tissueGPU1(int *d_tisspoints, float *d_dtt000, float *d_pt000, float *d_qt000, int nnt, int useGPU)
{
	int threadsPerBlock = 256;
	int step = 4;//has to be a power of two apparently
	int blocksPerGrid = (step*nnt + threadsPerBlock - 1) / threadsPerBlock;
	tissueGPU1Kernel<<<blocksPerGrid, threadsPerBlock>>>(d_tisspoints,d_dtt000,d_pt000,d_qt000,nnt,step);
}
