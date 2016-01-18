#include <iostream>
#include <cstring>
#include <mex.h>
#include <cmath>
#include <ctime>

using namespace std;

#define F            prhs[0]
#define IN_LO        prhs[1]
#define IN_HI        prhs[2]
#define IN_NP        prhs[3]
#define IN_SIMPLEX   prhs[4]
#define OUT          plhs[0]

void help() {
    mexPrintf("out = mexIntegrate(f, lo, hi, Npoints, simplex)\n");
    mexPrintf("\n");
    mexPrintf("Inputs: \n");
    mexPrintf("  - f       : handle of the function to integrate\n");
    mexPrintf("  - lo      : lower bounds of the domain of integration\n");
    mexPrintf("  - hi      : upper bounds of the domain of integration\n");
    mexPrintf("  - Npoints : number of points used for the integral estimate\n");
    mexPrintf("  - simplex : point are extracted from the simplex (0/1)\n");
    mexPrintf("\n");
    mexPrintf("Outputs: \n");
    mexPrintf("  - out:    : value of integration\n");
}

void SamplePoint(double *point, double *lo, double *hi, int dim, int useSimplex) {
    if (useSimplex == 0) {
        for (int i = 0; i < dim; ++i) {
            double rnd = (rand()+1e-16) / ((double)RAND_MAX);
            point[i] = lo[i] + rnd * (hi[i] - lo[i]);
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
            point[i] = lo[i] + point[i] * (hi[i] - lo[i]) / tot;
        }
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs < 5) {
        mexErrMsgTxt("Too few input arguments.");
    }
    if (nlhs > 1) {
        mexErrMsgTxt("Too many output arguments.");
    }
    if (mxGetClassID(F) != mxFUNCTION_CLASS) {
        mexErrMsgTxt("First input argument must be a function handle.");
    }
    if (mxGetNumberOfElements(IN_LO) != mxGetNumberOfElements(IN_HI)) {
        mexErrMsgTxt("Bounds of integration must have the same dimension.");
    }
    
    // initialize random seed
    srand(time(NULL));
    
    // allocate input
    double *lo = mxGetPr(IN_LO);
    double *hi = mxGetPr(IN_HI);
    int Npoints = (int) mxGetScalar(IN_NP);
    int useSimplex = (int) mxGetScalar(IN_SIMPLEX);
    int dim = mxGetNumberOfElements(IN_LO);
    
    // allocate for mexCallMATLAB
    mxArray **rhs;
    rhs = (mxArray **) mxMalloc((dim+1)*sizeof(*rhs));
    mxArray *lhs;
    rhs[0] = mxDuplicateArray(F);
    
    // shift if lower bound is not 0
    double *real_lo = (double*) mxMalloc(dim*sizeof(double));
    memcpy(real_lo, lo, dim*sizeof(double));
    for (int i = 0; i < dim; ++i) {
        hi[i] = hi[i] - lo[i];
        lo[i] = 0;
    }
    
    // compute integral
    double sum = 0.0;
    double *point = new double[dim];
    double *xptr;
    for (int i = 0; i < Npoints; ++i) {
        SamplePoint(point, lo, hi, dim, useSimplex);
        
        // build the vector of points for evaluation
        for (int j = 1; j <= dim; j++) {
            rhs[j] = mxCreateDoubleMatrix(1, 1, mxREAL);
            xptr = mxGetPr(rhs[j]);
            *xptr = point[j-1] + real_lo[j-1]; // shift back;
        }
        
        mexCallMATLAB(1, &lhs, dim+1, rhs, "feval");
        double fx = *mxGetPr(lhs);
//      bool bvalue = !std::isnan(fx) && !std::isinf(fx);
//      if (bvalue == false) {
//          mexErrMsgTxt("Integration gives NaN or Inf.");
//      }
        sum += fx;
        
        mxDestroyArray(lhs);
    }
    
    double volume = 1.0;
    for (int i = 0; i < dim; ++i) {
        volume *= fabs(hi[i] - lo[i]);
    }

    if (useSimplex == 1) {
        int fact = 1;
        // when using the simplex, the volume of the simplex is 1/DIM! of 
        // the hypercuboid defined by [lo,hi]
        for (int i = 1; i <= dim; i++)
            fact = fact * i;
        volume = volume / fact;
    }
    
    // allocate output
    double val = volume * sum / Npoints;
    OUT = mxCreateDoubleScalar(val);
    
    // deallocate memory
    delete [] point;
    mxFree(real_lo);
    mxFree(rhs);
}
