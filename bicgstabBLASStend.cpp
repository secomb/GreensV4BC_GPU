/**************************************************************************
bicgstabBLASStend - single precision
end bicgstabBLASSt
TWS, March 2011
**************************************************************************/
#include <shrUtils.h>
#include <cutil_inline.h>
#include <cusparse.h>
#include <cublas.h>
#include "nrutil.h"

void bicgstabBLASStend(int nnt)
{
	extern int useGPU;
	extern float *h_rst;
	extern float *d_rest, *d_xt, *d_bt;
	extern float *d_rt, *d_rst, *d_vt, *d_st, *d_tt, *d_pt, *d_ert;
 	const int nmem0 = sizeof(float);	//needed for malloc
 	const int nmem1 = nnt*sizeof(float);	//needed for malloc
	const int nmem2 = nnt*nnt*sizeof(float);

	cudaSetDevice( useGPU-1 );

	cudaFree(d_rest);
	cudaFree(d_bt);
	cudaFree(d_xt);
	cudaFree(d_ert);
	cudaFree(d_pt);
	cudaFree(d_tt);
	cudaFree(d_st);
	cudaFree(d_vt);
	cudaFree(d_rst);
	cudaFree(d_rt);
	
	cublasShutdown(); 
	cudaThreadExit();

	free_vector(h_rst,0,nnt-1);
}
