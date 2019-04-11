/***********************************************************
tissueGPU2.cu
GPU kernel to accumulate contributions of tissue source
strengths qt to vessel solute levels pv.
Each vessel point is assigned one thread.
TWS January 2012
************************************************************/
#include <omp.h>
#include <stdio.h>
#include <cutil_inline.h>

__global__ void tissueGPU2Kernel(float *d_tissxyz, float *d_vessxyz, float *d_pv000, float *d_qt000,
	int nnt, int nnv, int is2d, float req, float r2d)
{
    int ivp = blockDim.x * blockIdx.x + threadIdx.x;
	int jtp,nnt2=2*nnt;
	float p = 0., xv,yv,zv,x,y,z,dist2,gvt,req2=req*req,r2d2=r2d*r2d;
    if(ivp < nnv){
		xv = d_vessxyz[ivp];
		yv = d_vessxyz[ivp+nnv];
		zv = d_vessxyz[ivp+nnv*2];
		for(jtp=0; jtp<nnt; jtp++){
			x = d_tissxyz[jtp] - xv;
			y = d_tissxyz[jtp+nnt] - yv;
			z = d_tissxyz[jtp+nnt2] - zv;
			dist2 = x*x + y*y + z*z;
			if(dist2 < req2){
				if(is2d) gvt = log(r2d2/req2) + 1. - dist2/req2;
				else gvt = (1.5 - 0.5*dist2/req2)/req;
			}
			else{
				if(is2d) gvt = log(r2d2/dist2);
				else gvt = 1./sqrt(dist2);
			}
			p += d_qt000[jtp]*gvt;
		}
		d_pv000[ivp] = p;
	}
}

extern "C" void tissueGPU2(float *d_tissxyz, float *d_vessxyz, float *d_pv000, float *d_qt000,
		int nnt, int nnv, int is2d, float req, float r2d)
{
	int threadsPerBlock = 256;
	int blocksPerGrid = (nnv + threadsPerBlock - 1) / threadsPerBlock;
	tissueGPU2Kernel<<<blocksPerGrid, threadsPerBlock>>>(d_tissxyz, d_vessxyz, d_pv000, d_qt000,
		nnt, nnv, is2d, req, r2d);
}