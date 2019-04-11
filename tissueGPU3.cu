/***********************************************************
tissueGPU3.cu
GPU kernel to accumulate contributions of vessel source
strengths qv to tissue solute levels pt.
Each tissue point is assigned one thread.
TWS January 2012
************************************************************/
#include <stdio.h>
#include <cutil_inline.h>

__global__ void tissueGPU3Kernel(float *d_tissxyz, float *d_vessxyz, float *d_pt000, float *d_qv000,
	int nnt, int nnv, int is2d, float req, float r2d)
{
    int itp = blockDim.x * blockIdx.x + threadIdx.x;
	int jvp,nnv2=2*nnv;
	float p = 0., xt,yt,zt,x,y,z,dist2,gtv,req2=req*req,r2d2=r2d*r2d;
    if(itp < nnt){
		xt = d_tissxyz[itp];
		yt = d_tissxyz[itp+nnt];
		zt = d_tissxyz[itp+nnt*2];
		for(jvp=0; jvp<nnv; jvp++){
			x = d_vessxyz[jvp] - xt;
			y = d_vessxyz[jvp+nnv] - yt;
			z = d_vessxyz[jvp+nnv2] - zt;
			dist2 = x*x + y*y + z*z;
			if(dist2 < req2){
				if(is2d) gtv = log(r2d2/req2) + 1. - dist2/req2;
				else gtv = (1.5 - 0.5*dist2/req2)/req;
			}
			else{
				if(is2d) gtv = log(r2d2/dist2);
				else gtv = 1./sqrt(dist2);
			}
			p += d_qv000[jvp]*gtv;
		}
		d_pt000[itp] = p;
	}
}

extern "C" void tissueGPU3(float *d_tissxyz, float *d_vessxyz, float *d_pt000, float *d_qv000,
		int nnt, int nnv, int is2d, float req, float r2d)
{
	int threadsPerBlock = 256;
	int blocksPerGrid = (nnt + threadsPerBlock - 1) / threadsPerBlock;
	tissueGPU3Kernel<<<blocksPerGrid, threadsPerBlock>>>(d_tissxyz, d_vessxyz, d_pt000, d_qv000,
		nnt, nnv, is2d, req, r2d);
}