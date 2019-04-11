/**************************************************************************
BiCGSTABblasSt - single precision
Algorithm 12, Parameter-free iterative linear solver by R. Weiss, 1996
system of linear equations:  aa(i,j)*x(j) = b(i)
Version using BLAS library on GPU
TWS, March 2011
**************************************************************************/
#include <shrUtils.h>
#include <cutil_inline.h>
#include <cusparse.h>
#include <cublas.h>
#include "nrutil.h"

extern "C" void tissueGPU4(int *tisspoints, float *dtt000, float *qtp000, float *xt, float *rt,
	int nnt, int useGPU, float diff);

float bicgstabBLASSt(float *b, float *x, int nnt, float eps, int itmax, float diff)
{
	extern int useGPU;
	extern float *h_rst;
	extern float *d_rest, *d_xt, *d_bt;
	extern float *d_rt, *d_rst, *d_vt, *d_st, *d_tt, *d_pt, *d_ert;
 	const int nmem0 = sizeof(float);	//needed for memcpy
 	const int nmem1 = nnt*sizeof(float);
	int j,kk,ierr;
	float lu,lunew,beta,delta,gamma1,t1,t2,err;

	extern int *d_tisspoints;
	extern float *d_dtt000,*d_qtp000,*qtp000;

	for(j=0; j<nnt; j++) h_rst[j] = 1.;

	cudaSetDevice( useGPU-1 );

    cudaMemcpy(d_xt, x, nmem1, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bt, b, nmem1, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rst, h_rst, nmem1, cudaMemcpyHostToDevice);
	cudaMemcpy(d_qtp000, qtp000, nmem1, cudaMemcpyHostToDevice);

//	r[i] += a[i][j]*x[j];
//	cublasSgemv('T', nnt, nnt, 1.f, d_at, nnt, d_xt, 1, 0.f, d_rt, 1);
	tissueGPU4(d_tisspoints,d_dtt000,d_qtp000,d_xt,d_rt,nnt,useGPU,diff);

//	r[i] -= b[i];
	cublasSaxpy (nnt, -1.f, d_bt, 1, d_rt, 1);

//	p[i] = r[i];
	cublasScopy (nnt, d_rt, 1, d_pt, 1);
	
//	lu += r[i]*rs[i];
	lu = cublasSdot (nnt, d_rt, 1, d_rst, 1);

	kk = 1;
	do{
//		v[i] += a[i][j]*p[j];
//		cublasSgemv('T', nnt, nnt, 1.f, d_at, nnt, d_pt, 1, 0.f, d_vt, 1);
		tissueGPU4(d_tisspoints,d_dtt000,d_qtp000,d_pt,d_vt,nnt,useGPU,diff);

//		t1 += v[i]*rs[i];
		t1 = cublasSdot (nnt, d_vt, 1, d_rst, 1);
		if(t1 == 0){
			printf("t1 = 0, reset to 1e-12\n");
			t1 = 1.e-12;	//added January 2012
		}
		delta = -lu/t1;

//		s[i] = r[i] + delta*v[i];
		cublasScopy (nnt, d_rt, 1, d_st, 1);
		cublasSaxpy (nnt, delta, d_vt, 1, d_st, 1);
		
//		t[i] += a[i][j]*s[j];
//		cublasSgemv('T', nnt, nnt, 1.f, d_at, nnt, d_st, 1, 0.f, d_tt, 1);
		tissueGPU4(d_tisspoints,d_dtt000,d_qtp000,d_st,d_tt,nnt,useGPU,diff);

//		t1 += s[i]*t[i];
		t1 = cublasSdot (nnt, d_tt, 1, d_st, 1);

//		t2 += t[i]*t[i];
		t2 = cublasSdot (nnt, d_tt, 1, d_tt, 1);
		if(t2 == 0){
			printf("t2 = 0, reset to 1e-12\n");
			t2 = 1.e-12;	//added January 2012
		}
		gamma1 = -t1/t2;

//		r[i] = s[i] + gamma1*t[i];
		cublasScopy (nnt, d_st, 1, d_rt, 1);
		cublasSaxpy (nnt, gamma1, d_tt, 1, d_rt, 1);

//		er[i] = delta*p[i] + gamma1*s[i];
		cublasScopy (nnt, d_st, 1, d_ert, 1);
		cublasSscal (nnt, gamma1, d_ert, 1);
		cublasSaxpy (nnt, delta, d_pt, 1, d_ert, 1);

//		x[i] += er[i];
		cublasSaxpy (nnt, 1., d_ert, 1, d_xt, 1);

//		lunew += r[i]*rs[i];
		lunew = cublasSdot (nnt, d_rt, 1, d_rst, 1);
		if(lunew == 0){
			printf("lunew = 0, reset to 1e-12\n");
			lunew = 1.e-12;	//added January 2012
		}
		if(gamma1 == 0){
			printf("gamma1 = 0, reset to 1e-12\n");
			gamma1 = 1.e-12;	//added January 2012
		}
		beta = lunew*delta/(lu*gamma1);
		lu = lunew;

//		p[i] = r[i] + beta*(p[i]+gamma1*v[i]);
		cublasSaxpy (nnt, gamma1, d_vt, 1, d_pt, 1);
		cublasSscal (nnt, beta, d_pt, 1);
		cublasSaxpy (nnt, 1., d_rt, 1, d_pt, 1);

		ierr = cublasIsamax (nnt, d_ert, 1);   //Find the maximum value in the array
		cudaMemcpy(&err, d_ert+ierr-1, nmem0, cudaMemcpyDeviceToHost);
		kk++;
	}
	while(kk < itmax && abs(err) > eps);

	cudaMemcpy(x, d_xt, nmem1, cudaMemcpyDeviceToHost);//bring back x for final result

	if(abs(err) > eps) printf("*** Warning: linear solution using BICGSTB not converged, err = %e\n",err);
	return err;	
}
