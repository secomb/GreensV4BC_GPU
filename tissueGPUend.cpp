/**************************************************************************
tissueGPUend
end tissueGPU
TWS, December 2011
**************************************************************************/
#include <shrUtils.h>
#include <cutil_inline.h>
#include "nrutil.h"

void tissueGPUend(int nntGPU, int nnvGPU)
{
	extern int useGPU,*h_tisspoints,*d_tisspoints;
	extern float *pt000,*qt000,*qtp000,*pv000,*qv000,*dtt000,*h_tissxyz,*h_vessxyz;
	extern float *d_qt000,*d_qtp000,*d_pt000,*d_qv000,*d_pv000,*d_dtt000;
	extern float *d_tissxyz,*d_vessxyz;

	free_ivector(h_tisspoints,0,3*nntGPU-1);
	free_vector(pt000,0,nntGPU-1);
	free_vector(qt000,0,nntGPU-1);
	free_vector(qtp000,0,nntGPU-1);
	free_vector(pv000,0,nnvGPU-1);
	free_vector(qv000,0,nnvGPU-1);
	free_vector(dtt000,0,nntGPU-1);
	free_vector(h_tissxyz,0,3*nntGPU-1);		//coordinates of tissue points
	free_vector(h_vessxyz,0,3*nnvGPU-1);		//coordinates of vessel points

	cudaSetDevice( useGPU-1 );
	cudaFree(d_tisspoints);
	cudaFree(d_pt000);
	cudaFree(d_qt000);
	cudaFree(d_qtp000);
	cudaFree(d_pv000);
	cudaFree(d_qv000);
	cudaFree(d_dtt000);
	cudaFree(d_tissxyz);
	cudaFree(d_vessxyz);
}
