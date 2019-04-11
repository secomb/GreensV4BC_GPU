/**************************************************************************
bicgstabBLASStinit - single precision
initialize bicgstabBLASSt
TWS, March 2011
**************************************************************************/
#include <shrUtils.h>
#include <cutil_inline.h>
#include <cusparse.h>
#include <cublas.h>
#include "nrutil.h"

void bicgstabBLASStinit(int nnt)
{
	extern int useGPU;
	extern float *h_rst;
	extern float *d_rest, *d_xt, *d_bt;
	extern float *d_rt, *d_rst, *d_vt, *d_st, *d_tt, *d_pt, *d_ert;
 	const int nmem0 = sizeof(float);	//needed for malloc
 	const int nmem1 = nnt*sizeof(float);	//needed for malloc

	h_rst = vector(0,nnt-1);

	cudaSetDevice( useGPU-1 );//device 0 for vessel calculations
	cublasInit(); 

	cudaMalloc((void **)&d_rt, nmem1);
	cudaMalloc((void **)&d_rst, nmem1);
	cudaMalloc((void **)&d_vt, nmem1);
	cudaMalloc((void **)&d_st, nmem1);
	cudaMalloc((void **)&d_tt, nmem1);
	cudaMalloc((void **)&d_pt, nmem1);
	cudaMalloc((void **)&d_ert, nmem1);
	cudaMalloc((void **)&d_xt, nmem1);
	cudaMalloc((void **)&d_bt, nmem1);
	cudaMalloc((void **)&d_rest, nmem0);
}
