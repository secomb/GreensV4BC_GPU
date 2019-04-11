/************************************************************************
cgsymm - for Greens.  TWS September 2010
Conjugate gradient method for symmetric system of linear equations:  a(i,j)*x(j) = b(i)
Modified to generate tissue interaction matrix elements on the fly
Method uses following code from wikipedia:
-------------------------------------
function [x] = conjgrad(a,b,x)
r=b-a*x;
p=r;
rsold=r'*r;
for i=1:size(a)(1)
   ap=a*p;
   alpha=(r'*r)/(p'*ap);
   x=x+alpha*p;
   r=r-alpha*ap;
   rsnew=r'*r;
   if sqrt(rsnew)<1e-10
      break;
      end
   p=r+rsnew/rsold*p;
   rsold=rsnew;
end
*************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "nrutil.h"

double cgsymm(double **a, double *b, double *x, int n, double eps, int itmax)
{
	int i,j,jj;
	double *r,*p,*ap,rsold,rsnew,pap;

	r = dvector(1,n);
	p = dvector(1,n);
	ap = dvector(1,n);

	for(i=1; i<=n; i++) x[i] = 0.;  //Initial guess
	rsold = 0.;
	for(i=1; i<=n; i++){
		r[i] = b[i];
		for(j=1;j<=n;++j) r[i] -= a[i][j]*x[j];
		rsold += r[i]*r[i];
		p[i] = r[i]; //  initialize p as r
	}
    jj = 0;
    do{
		pap = 0.;
		for(i=1; i<=n; i++){
			ap[i] = 0.;
			for(j=1; j<=n; j++) ap[i] += a[i][j]*p[j];			
			pap += p[i]*ap[i];
		}
		for(i=1; i<=n; i++){
			x[i] += rsold/pap*p[i];        
			r[i] -= rsold/pap*ap[i];
		}
		rsnew = 0.;
		for(i=1; i<=n; i++)	rsnew += r[i]*r[i];
        for(i=1; i<=n; i++) p[i] = r[i] + rsnew/rsold*p[i];
        jj++;
		rsold = rsnew;
    }
	while(rsnew > eps && jj < itmax);
	printf("cgsymm: %i %i %f\n",n,jj,rsnew);
	free_dvector(r,1,n);
	free_dvector(p,1,n);
 	free_dvector(ap,1,n);
    return rsnew;
}