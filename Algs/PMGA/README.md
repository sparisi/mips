Description
-----------

Implementation of Pareto-Manifold Gradient Algorithm (PMGA) as presented in

*PIROTTA, M.; PARISI, S.; RESTELLI, M.. Multi-Objective Reinforcement Learning with Continuous Pareto Frontier Approximation. AAAI Conference on Artificial Intelligence, North America, feb. 2015.*

The paper is available at http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9798

Additional requirements: Symbolic Toolbox.


How To Use It
-------------

There are two versions of PMGA: exact and sampled. The former is available only for the LQR domain and can be found in *lqr_exact*. As this implementation relies on exact symbolic equations, for the 3-dimensional LQR we provide a MEX interface for a fast estimate of the integrals. If you want to try different parameterizations / indicators or new exact domains, the necessary steps to generate new MEX files are the following:

 - run *genCcode* to generate C code from the exact symbolic equations,
 - run *genHeader* to generate C headers,
 - run *compileSrc* to build the MEX files.

This procedure will generate files named *mexIntegrate_NAME_PARAMETERIZATION_INDICATOR*.
You can then write your own script to run PMGA (have a look at *manifold_lqr2* and *manifold_lqr3_mex*).

For the sampled implementation of PMGA, you have to define the manifold parametrization in *params_NAME* and run *pmga*. In *pmga* you need to set the domain and the parameters of the indicator function.

The implemented indicators are:

 - *utopia* (L_U)     : distance from utopia
 - *antiutopia* (L_A) : distance from antiutopia
 - *pareto* (L_P)     : Pareto-ascent norm (not implemented for the sampled version)
 - *mix1* (L_M1)      : L_A * (1 - w * L_P) (not implemented for the sampled version)
 - *mix2* (L_M2)      : beta1 * L_A / L_U - beta2
 - *mix3* (L_M3)      : L_A * (1 - w * L_U)
