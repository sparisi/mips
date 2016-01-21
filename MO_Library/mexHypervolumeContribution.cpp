#include "mex.h"
#include "math.h"
#include <ctime>

/* Hypervolume contribution of solutions of a frontier F.
 * The computation is done by Monte Carlo approximation and exploiting a 
 * simple trick. First, the algorithm generates N sample points P in the
 * hypercuboid defined by two reference points, namely the utopia and 
 * antiutopia. It then checks whether a point P(i) is dominated by a
 * frontier point F(j). If P(i) is dominated by only one point F(j), then 
 * F(j) contributes to the hypervolume growth of the frontier and its 
 * contribution counter is increased. 
 *
 * NB! The frontier F must contain only unique solutions! Duplicates will 
 * lead to a contribution counter of 0.
 * 
 * See also MEXMETRIC_HV. 
 *
 *    INPUT
 *     - F  : the set of points (NF-by-D)
 *     - AU : antiutopia point (1-by-D)
 *     - U  : utopia point (1-by-D)
 *     - N  : number of samples for the Monte Carlo approximation
 *
 *    OUTPUT
 *     - C : dominance matrix (NF-by-1)
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
    
    plhs[0] = mxCreateDoubleMatrix(NF, 1, mxREAL);
    double *contribution = mxGetPr(plhs[0]); // hypervolume contribution
    
    double P[dimF];
    double rnd;
    for (i = 0; i < N; i++) {
        for (j = 0; j < dimF; j++) { // generate a random point P in the hypercuboid between AU and U
            rnd = rand() / ((double)RAND_MAX);
            P[j] = AU[j] + rnd * (U[j] - AU[j]);
        }
        
        int dominatingIdx = -1; // if P is dominated by only one solution F(j,:), we store index j here
        for (j = 0; j < NF; j++) { // check all points in F
            int dominates = 1; // does F(j,:) dominate P?
            for (k = 0; k < dimF; k++) { // check all dimensions
                if (FMatrix[k][j] < P[k]) { // F(j,:) does not dominate P, move on to the next point
                    dominates = 0;
                    break;
                }
            }
            if (dominates == 1) { // check dominatingIdx
                if (dominatingIdx == -1) // if F(j,:) was the first one so far to dominate P, note it
                    dominatingIdx = j;
                else
                    dominatingIdx = -2; // P is dominated by more than one point, so ignore it
            }
        }
        
        if (dominatingIdx != -1 && dominatingIdx != -2) // P is dominated by only one solution F(j,:)
            contribution[dominatingIdx] += 1; // therefore F(j,:) contributes to the hypervolume growth
    }
    
    for (j = 0; j < NF; j++) { // finally normalize the Monte Carlo estimate
        contribution[j] /= N;
    }
}