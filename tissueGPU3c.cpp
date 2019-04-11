/**************************************************************************
tissueGPU3.cpp
program to call tissueGPU3.cu on GPU
TWS, January 2012
**************************************************************************/
#include <shrUtils.h>
#include <cutil_inline.h>

extern "C" void tissueGPU3(float *d_tissxyz, float *d_vessxyz, float *d_pt000, float *d_qv000,
		int nnt, int nnv, int is2d, float req, float r2d);

void tissueGPU3c()
{
	extern int nnt,nnv,is2d,useGPU;
	extern float *d_tissxyz,*d_vessxyz,*d_pt000,*d_qv000,*qv000,*pt000,req,r2d;
	cudaError_t error;

	cudaSetDevice( useGPU-1 );
	error = cudaMemcpy(d_qv000, qv000, nnv*sizeof(float), cudaMemcpyHostToDevice);
	tissueGPU3(d_tissxyz,d_vessxyz,d_pt000,d_qv000,nnt,nnv,is2d,req,r2d);
	error = cudaMemcpy(pt000, d_pt000, nnt*sizeof(float), cudaMemcpyDeviceToHost);
}