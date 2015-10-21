#include <iostream>
#include <cstring>
#include <mex.h>
#include <cmath>
#include <ctime>

#include "header/LQR3_P2_pareto_Djr1.h"
#include "header/LQR3_P2_pareto_Djr2.h"
#include "header/LQR3_P2_pareto_Djr3.h"
#include "header/LQR3_P2_pareto_Djr4.h"
#include "header/LQR3_P2_pareto_Djr5.h"
#include "header/LQR3_P2_pareto_Djr6.h"
#include "header/LQR3_P2_pareto_Djr7.h"
#include "header/LQR3_P2_pareto_Djr8.h"
#include "header/LQR3_P2_pareto_Djr9.h"
#include "header/LQR3_P2_pareto_Jr.h"

#define IN_LO        prhs[0]
#define IN_HI        prhs[1]
#define IN_NP        prhs[2]
#define IN_SIMPLEX   prhs[3]
#define IN_RHO       prhs[4]
#define IN_PARAMS    prhs[5]

#define OUT_J        plhs[0]
#define OUT_DJ       plhs[1]

using namespace std;

int useSimplex;

typedef double (*f_integral) (double*, int, double*, int);

void help() {
    mexPrintf(" [J, DJ] = mexIntegrate(lo, hi, Npoints, simplex, rho, params)\n");
    mexPrintf("\n");
    mexPrintf("Inputs: \n");
    mexPrintf("  - lo      : lower bounds of the domain of integration\n");
    mexPrintf("  - hi      : upper bounds of the domain of integration\n");
    mexPrintf("  - Npoints : number of points used for the integral estimate\n");
    mexPrintf("  - simplex : point are extracted from the simplex (0/1)\n");
    mexPrintf("  - rho     : the parameters rho\n");
    mexPrintf("  - params  : other parameters (e.g., utopia, antiutopia, ...\n");
    mexPrintf("\n");
    mexPrintf("Outputs: \n");
    mexPrintf("  - J       : integral of function J(rho)\n");
    mexPrintf("  - DJ      : integral of the derivative of J(rho)\n");
}

void SamplePoint(double *point, double *lo, double *hi, int dim) {
    if (useSimplex == 0) {
        for (int i = 0; i < dim; ++i) {
            double rnd = rand() / ((double)RAND_MAX);
            point[i] = lo[i] + rnd * (hi[i] -lo[i]);
        }
    } else {
        double tot = 0.0;
        for (int i = 0; i < dim; ++i) {
            double rnd = (rand()+1e-16) / ((double)RAND_MAX);
            rnd = -log(rnd);
            point[i] = rnd;
            tot += rnd;
        }
        double rnd = (rand()+1e-16) / ((double)RAND_MAX);
        tot += -log(rnd);

        for (int i = 0; i < dim; ++i) {
            point[i] = lo[i] + point[i] * (hi[i] -lo[i]) / tot;
//             mexPrintf("point[%d]: %f\n", i, point[i]);
        }
    }
}

double Integrate(f_integral f, double *lo, double *hi, int dim, int N, double *parameters, int dimrho) {
    double sum = 0.0, sumsq = 0.0;
    double *point = new double[dim];
    for (int i = 0; i < N; ++i) {
        SamplePoint(point, lo, hi, dim);

        double fx = (*f)(point, dim, parameters, dimrho);

//        bool bvalue = !std::isnan(fx) && !std::isinf(fx);
//        if (bvalue == false) {
//             mexErrMsgTxt("Integration gives NaN or Inf.");
//        }

        sum += fx;
        sumsq += fx * fx;
    }

    double volume  = 1.0;
    for (int i = 0; i < dim; ++i) {
        volume *= fabs(hi[i] - lo[i]);
    }

    if (useSimplex == 1) {
        volume = volume / 2.0;
    }

    delete [] point;

    return volume * sum / N;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs < 6) {
        mexErrMsgTxt("Too few input arguments.");
    }
    if (nlhs > 2) {
        mexErrMsgTxt("Too many output arguments.");
    }
    if (mxGetNumberOfElements(IN_LO) != mxGetNumberOfElements(IN_HI)) {
        mexErrMsgTxt("Bounds of integration must have the same dimension.");
    }
    
    // initialize random seed
    srand (time(NULL));

    double *li     = mxGetPr(IN_LO);
    double *hi     = mxGetPr(IN_HI);
    int Np         = (int) mxGetScalar(IN_NP);
    useSimplex     = (int) mxGetScalar(IN_SIMPLEX);
    double *rho    = mxGetPr(IN_RHO);
    double *params = mxGetPr(IN_PARAMS);

    int tdim = mxGetNumberOfElements(IN_LO);
    int rdim = mxGetNumberOfElements(IN_RHO);

    OUT_DJ = mxCreateDoubleMatrix(rdim,1,mxREAL);
    double* res = mxGetPr(OUT_DJ);

    f_integral ff = 0;

    // #pragma omp parallel for
    for (int i = 1; i <= rdim; ++i) {
        #define FUN_NAME(x) \
        case x: ff = LQR3_P2_pareto_Djr ## x; break;
        switch ( i ) {
            FUN_NAME(1);
            FUN_NAME(2);
            FUN_NAME(3);
            FUN_NAME(4);
            FUN_NAME(5);
            FUN_NAME(6);
            FUN_NAME(7);
            FUN_NAME(8);
            FUN_NAME(9);
        }
        #undef FUN_NAME
        double val = Integrate(ff, li, hi, tdim, Np, rho, rdim);
        res[i-1] = val;
    }

    // integrate Jr
    ff = LQR3_P2_pareto_Jr;
    double val = Integrate(ff, li, hi, tdim, Np, rho, rdim);
    OUT_J = mxCreateDoubleScalar(val);
}
