#include "mex.h"
#include "math.h"
#include <ctime>

/* Monte Carlo approximation of the hypervolume of a set of points F.
 * The algorothm generates random samples in the hypercuboid defined by two
 * reference points, namely the utopia and antiutopia, and counts the 
 * number of points dominated by F. The hypervolume is approximated as the 
 * ratio 'dominated points / total points'.
 *
 *    INPUT
 *     - F  : the set of points
 *     - AU : antiutopia point
 *     - U  : utopia point
 *     - N  : number of samples for the approximation
 *
 *    OUTPUT
 *     - hv : hypervolume
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs < 4) {
        mexErrMsgTxt("Too few input arguments.");
    }
    if (nrhs > 4) {
        mexErrMsgTxt("Too many input arguments.");
    }
    if (nlhs > 1) {
        mexErrMsgTxt("Too many output arguments.");
    }
    
    int NF = mxGetM(prhs[0]); // number of points of the set F
    int dimF = mxGetN(prhs[0]); // dimension of the points
    int dimAU = mxGetNumberOfElements(prhs[1]); 
    int dimU = mxGetNumberOfElements(prhs[2]); 
    if (dimAU != dimU || dimAU != dimF )
        mexErrMsgTxt("Dimensions between the set of points, utopia and antiutopia do not match.");
    
    double *F = mxGetPr(prhs[0]); // set of points
    double *AU = mxGetPr(prhs[1]); // antiutopia
    double *U = mxGetPr(prhs[2]); // utopia
    int N = (int) mxGetScalar(prhs[3]); // number of points to approximate the hypervolume
    
    int i, j, k;
    for (i = 0; i < dimF; i++)
        if (AU[i] >= U[i])
            mexErrMsgTxt("Utopia must dominate antiutopia.");
    
    double *(FMatrix[dimF + 1]);
    for (j = 0; j < dimF; ++j) {
        FMatrix[j] = &(F[NF * j]);
    }
    
    srand (time(NULL)); // initialize random seed

    double P[dimF];
    double rnd;
    int count = 0;
    for (i = 0; i < N; i++) {
        for (int j = 0; j < dimF; j++) { // generate a random point P in the hypercuboid between AU and U
            rnd = rand() / ((double)RAND_MAX);
            P[j] = AU[j] + rnd * (U[j] - AU[j]);
        }        
        
        for (j = 0; j < NF; j++) { // check all points in F
            int dominates = 1; // does F(j) dominate P?
            for (k = 0; k < dimF; k++) { // check all dimensions
                if (FMatrix[k][j] < P[k]) { // F(j) does not dominate P, move on to the next point
                    dominates = 0;
                    break;
                }
            }
            if (dominates == 1) { // if P is dominated stop checking
                count++;
                break;
            }
        }
    }
    
    double hv = (double) count / N;
    plhs[0] = mxCreateDoubleScalar(hv);
}