/************************************************************************
bicgstab - for Greens.  TWS April 2016
Based on algorithm 12, Parameter-free iterative linear solver by R. Weiss, 1996
system of linear equations:  aa(i,j)*x(j) = b(i)
This version for tissue calculation evaluates matrix values from dtt and qtp
*************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "nrutil.h"

double bicgstab_tiss(double *b, double *x, int nnt, double eps, int itmax, int isp)
{
	extern int **tisspoints;
	extern float ***dtt,*diff,**qtp;
	double lu,lunew,beta,delta,er,gamma1,t1,t2,err;
	double *r,*rs,*v,*s,*t,*p;
	int itp,jtp,kk,ix,iy,iz,jx,jy,jz,ixdiff,iydiff,izdiff;
	r = dvector(1,nnt);
	rs = dvector(1,nnt);
	v = dvector(1,nnt);
	s = dvector(1,nnt);
	t = dvector(1,nnt);
	p = dvector(1,nnt);
	lu = 0.;
	for(itp=1; itp<=nnt; itp++){
		ix = tisspoints[1][itp];
		iy = tisspoints[2][itp];
		iz = tisspoints[3][itp];
        r[itp] = x[itp];
		for(jtp=1; jtp<=nnt; jtp++){
			jx = tisspoints[1][jtp];
			jy = tisspoints[2][jtp];
			jz = tisspoints[3][jtp];
			ixdiff = abs(ix - jx) + 1;
			iydiff = abs(iy - jy) + 1;
			izdiff = abs(iz - jz) + 1;
			r[itp] -= dtt[ixdiff][iydiff][izdiff]*qtp[jtp][isp]/diff[isp]*x[jtp];
		}
        r[itp] -= b[itp];
        p[itp] = r[itp];
        rs[itp] = 1.;
        lu += r[itp]*rs[itp];
	}
	kk = 1;
	do
	{
		t1 = 0.;
		for(itp=1; itp<=nnt; itp++){
			ix = tisspoints[1][itp];
			iy = tisspoints[2][itp];
			iz = tisspoints[3][itp];
			v[itp] = p[itp];
			for(jtp=1; jtp<=nnt; jtp++){
				jx = tisspoints[1][jtp];
				jy = tisspoints[2][jtp];
				jz = tisspoints[3][jtp];
				ixdiff = abs(ix - jx) + 1;
				iydiff = abs(iy - jy) + 1;
				izdiff = abs(iz - jz) + 1;
				v[itp] -= dtt[ixdiff][iydiff][izdiff]*qtp[jtp][isp]/diff[isp]*p[jtp];			
			}
			t1 += v[itp]*rs[itp];
		}
		delta = -lu/t1;
		for(itp=1; itp<=nnt; itp++) s[itp] = r[itp] + delta*v[itp];
		for(itp=1; itp<=nnt; itp++){
			ix = tisspoints[1][itp];
			iy = tisspoints[2][itp];
			iz = tisspoints[3][itp];
			t[itp] = s[itp];
			for(jtp=1; jtp<=nnt; jtp++){
				jx = tisspoints[1][jtp];
				jy = tisspoints[2][jtp];
				jz = tisspoints[3][jtp];
				ixdiff = abs(ix - jx) + 1;
				iydiff = abs(iy - jy) + 1;
				izdiff = abs(iz - jz) + 1;
				t[itp] -= dtt[ixdiff][iydiff][izdiff]*qtp[jtp][isp]/diff[isp]*s[jtp];		
			}
		}
		t1 = 0.;
		t2 = 0.;
		for(itp=1; itp<=nnt; itp++){
			t1 += s[itp]*t[itp];
			t2 += t[itp]*t[itp];
		}
		gamma1 = -t1/t2;
		err = 0.;
		lunew = 0.;
		for(itp=1; itp<=nnt; itp++){
			r[itp] = s[itp] + gamma1*t[itp];
			er = delta*p[itp] + gamma1*s[itp];
			x[itp] += er;
			if(fabs(er) > err) err = fabs(er);
			lunew += r[itp]*rs[itp];
		}
		beta = lunew*delta/(lu*gamma1);
		lu = lunew;
		for(itp=1; itp<=nnt; itp++) p[itp] = r[itp] + beta*(p[itp]+gamma1*v[itp]);
		kk += 1;
	}
	while(kk < itmax && err > eps);
	free_dvector(r,1,nnt);
	free_dvector(rs,1,nnt);
	free_dvector(v,1,nnt);
	free_dvector(s,1,nnt);
	free_dvector(t,1,nnt);
	free_dvector(p,1,nnt);
	if(err > eps) printf("*** Warning: linear solution using BICGSTB not converged, err = %e\n",err);
	return err;
}