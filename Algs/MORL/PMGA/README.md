    Description
    -----------

    Implementation of **P**areto-**M**anifold **G**radient **A**lgorithm as presented in

    [PIROTTA, M; PARISI, S; RESTELLI, M. *Multi-Objective Reinforcement Learning with Continuous Pareto Frontier Approximation*, Proceedings of the Conference on Artificial Intelligence (AAAI), 2015](http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9798)

    and 

    [PARISI, S; PIROTTA, M; RESTELLI, M. *Multi-objective Reinforcement Learning through Continuous Pareto Manifold Approximation*, Journal of Artificial Intelligence Research (JAIR), 2016](http://www.ausy.tu-darmstadt.de/uploads/Site/EditPublication/PARISI_JAIR_MORL.pdf)

    It requires the *Symbolic Toolbox*.


    How To Use It
    -------------

    There are two versions of PMGA: exact and sampled. The **exact** is available only for the LQR domain and can be found in `lqr_exact`. As this implementation relies on exact symbolic equations and needs to estimate integrals in closed form, for the 3-dimensional LQR we provide a MEX interface for faster computation. If you want to try different parameterizations / indicators or new exact domains, the necessary steps to generate new MEX files are the following:

     - Run `genCcode` to generate C code from the exact symbolic equations defining your problem,
     - Run `genHeader` to generate C headers,
     - Write your C / C++ source files in `mexFiles/src` (the filename format is `mexIntegrate_MDP_PARAMETERIZATION_INDICATOR`),
     - Run `compileSrc` to build the MEX files.

    You can then write your own script to run PMGA (have a look at `pmga_exact_lqr2` and `pmga_exact_lqr3_mex`).

    For the **sampled** implementation of PMGA, you have to define the manifold parametrization in `params_NAME` and run `pmga`. In `pmga` you need to set the domain and the parameters of the indicator function.

    The implemented indicators are:

     - *utopia* (I<sub>U</sub>)     : distance from utopia
     - *antiutopia* (I<sub>A</sub>) : distance from antiutopia
     - *pareto* (I<sub>P</sub>)     : Pareto-ascent norm (not implemented for the sampled version)
     - *mix1* (I<sub>M1</sub>)      : I<sub>A</sub>(1 - ?I<sub>P</sub>) (not implemented for the sampled version)
     - *mix2* (I<sub>M2</sub>)      : ?<sub>1</sub>I<sub>A</sub>/I<sub>U</sub> - ?<sub>2</sub>
     - *mix3* (I<sub>M3</sub>)      : I<sub>A</sub>(1 - ?I<sub>U</sub>)
