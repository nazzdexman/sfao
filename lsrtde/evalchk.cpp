#include <cstdio>
#include <cstdlib>
#include <cmath>
double *OShift,*M,*y,*z,*x_bound; int ini_flag=0,n_flag,func_flag,*SS;
#if BENCH==22
#include "cec22_test_func.cpp"
#define CF cec22_test_func
#else
#include "cec17_test_func.cpp"
#define CF cec17_test_func
#endif
int main(int argc,char**argv){ int f=atoi(argv[1]), D=atoi(argv[2]); double x[100], v;
  while(1){ for(int i=0;i<D;i++) if(scanf("%lf",&x[i])!=1) return 0; CF(x,&v,D,1,f); printf("%.17g\n",v);} }
