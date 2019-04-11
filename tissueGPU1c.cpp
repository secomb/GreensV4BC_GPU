/**************************************************************************
tissueGPU1.cpp
program to call tissueGPU1.cu on GPU
TWS, December 2011
**************************************************************************/
#include <omp.h>
#include <shrUtils.h>
#include <cutil_inline.h>
#include "nrutil.h"

extern "C" void tissueGPU1(int *tisspoints, float *dtt000, float *pt000, float *qt000, int nnt, int useGPU);

void tissueGPU1c()
{
	extern int nnt,useGPU,*d_tisspoints;
	extern float *pt000,*qt000,*d_qt000,*d_pt000,*d_dtt000;
	cudaError_t error;

	cudaSetDevice( useGPU-1 );
	error = cudaMemcpy(d_qt000, qt000, nnt*sizeof(float), cudaMemcpyHostToDevice);
	tissueGPU1(d_tisspoints,d_dtt000,d_pt000,d_qt000,nnt,useGPU);
	error = cudaMemcpy(pt000, d_pt000, nnt*sizeof(float), cudaMemcpyDeviceToHost);
}