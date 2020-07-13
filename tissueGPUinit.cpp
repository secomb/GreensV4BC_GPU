/**************************************************************************
tissueGPUinit
initialize tissueGPU
TWS, December 2011
Cuda 10.1 Version, August 2019
**************************************************************************/
//#include <cutil_inline.h>
#include "cuda_runtime.h"
#include "nrutil.h"

void tissueGPUinit(int nntGPU, int nnvGPU)
{
	extern int *h_tisspoints,*d_tisspoints;
	extern float *pt000,*qt000,*qtp000,*pv000,*qv000,*dtt000;
	extern float *d_pt000,*d_qt000,*d_qtp000,*d_pv000,*d_qv000,*d_dtt000;
	extern float *h_tissxyz,*h_vessxyz,*d_tissxyz,*d_vessxyz;

	h_tisspoints = ivector(0,3*nntGPU-1);	//integer coordinates of tissue points
	pt000 = vector(0,nntGPU-1);
	qt000 = vector(0,nntGPU-1);
	qtp000 = vector(0,nntGPU-1);
	pv000 = vector(0,nnvGPU-1);
	qv000 = vector(0,nnvGPU-1);
	dtt000 = vector(0,nntGPU-1);			//this has to cover all possible distances
	h_tissxyz = vector(0,3*nntGPU-1);		//coordinates of tissue points
	h_vessxyz = vector(0,3*nnvGPU-1);		//coordinates of vessel points

	cudaMalloc((void **)&d_tisspoints, 3*nntGPU*sizeof(int));
	cudaMalloc((void **)&d_pt000, nntGPU*sizeof(float));
	cudaMalloc((void **)&d_qt000, nntGPU*sizeof(float));
	cudaMalloc((void **)&d_qtp000, nntGPU*sizeof(float));
	cudaMalloc((void **)&d_pv000, nnvGPU*sizeof(float));
	cudaMalloc((void **)&d_qv000, nnvGPU*sizeof(float));
	cudaMalloc((void **)&d_dtt000, nntGPU*sizeof(float));
	cudaMalloc((void **)&d_tissxyz, 3*nntGPU*sizeof(float));
	cudaMalloc((void **)&d_vessxyz, 3*nnvGPU*sizeof(float));
}
