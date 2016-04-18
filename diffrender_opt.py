# Copyright (c) 2015, Javier Gonzalez
# Copyright (c) 2015, the GPy Authors (see GPy AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import GPyOpt
import GPy
import ipdb
from numpy.random import seed
import chumpy as ch

"""
This is a simple demo to demonstrate the use of Bayesian optimization with GPyOpt with some simple options. Run the example by writing:

import GPyOpt
BO_demo_2d = GPyOpt.demos.advanced_optimization_2d()

As a result you should see:

- A plot with the model and the current acquisition function
- A plot with the diagnostic plots of the optimization.
- An object call BO_demo_auto that contains the results of the optimization process (see reference manual for details). Among the available results you have access to the GP model via

>> BO_demo_2d.model

and to the location of the best found location writing.

BO_demo_2d.x_opt

"""




def opendrObjectiveFunction(obj, free_variables):

    def changevars(vs, obj, free_variables):
        vs = vs.ravel()
        cur = 0
        changed = False
        for idx, freevar in enumerate(free_variables):
            sz = freevar.r.size

            newvals = vs[cur:cur+sz].copy().reshape(free_variables[idx].shape)
            if np.max(np.abs(newvals-free_variables[idx]).ravel()) > 0:
                free_variables[idx][:] = newvals
                changed = True

            cur += sz

        return changed

    def objFun(vs):
        vs = np.array(vs)
        if vs.shape[0] == 1:
            changevars(vs, obj, free_variables)
            return obj.r.reshape([1,1])
        else:
            res = []
            for vs_i in vs:
                changevars(vs_i, obj, free_variables)
                res = res + [obj.r.reshape([1,1])]

            return np.vstack(res)

    return objFun

def opendrObjectiveFunctionCRF(free_variables, rendererGT, renderer, color, chVColors, chSHLightCoeffs, lightCoeffs, free_variables_app_light, vis_im, bound_im, resultDir, test_i, stds, method, minAppLight=False):

    def changevars(vs, free_variables):
        vs = vs.ravel()
        cur = 0
        changed = False
        for idx, freevar in enumerate(free_variables):
            sz = freevar.r.size

            newvals = vs[cur:cur+sz].copy().reshape(free_variables[idx].shape)
            if np.max(np.abs(newvals-free_variables[idx]).ravel()) > 0:
                free_variables[idx][:] = newvals
                changed = True

            cur += sz

        return changed

    def objFun(vs):

        vs = np.array(vs)
        res = []
        for vs_i in vs:
            changevars(vs_i, free_variables)

            import densecrf_model

            segmentation, Q = densecrf_model.crfInference(rendererGT.r, vis_im, bound_im, [0.75,0.25,0.01], resultDir + 'imgs/crf/Q_' + str(test_i))
            if np.sum(segmentation==0) == 0:
                vColor = color
            else:
                segmentRegion = segmentation==0
                vColor = np.median(rendererGT.reshape([-1,3])[segmentRegion.ravel()], axis=0)

            chVColors[:] = vColor
            chSHLightCoeffs[:] = lightCoeffs

            options={'disp':False, 'maxiter':5}

            variances = stds**2

            fgProb = ch.exp( - (renderer - rendererGT)**2 / (2 * variances)) * (1./(stds * np.sqrt(2 * np.pi)))

            h = renderer.r.shape[0]
            w = renderer.r.shape[1]

            occProb = np.ones([h,w])
            bgProb = np.ones([h,w])

            errorFun = -ch.sum(ch.log((Q[0].reshape([h,w,1])*fgProb) + (Q[1].reshape([h,w])*occProb)[:,:,None] + (Q[2].reshape([h,w])*bgProb)[:,:,None]))/(h*w)

            if minAppLight:
                def cb(_):
                    print("Error: " + str(errorFun.r))

                ch.minimize({'raw': errorFun}, bounds=None, method=method, x0=free_variables_app_light, callback=cb, options=options)

            res = res + [errorFun.r.reshape([1,1])]

        return np.vstack(res)

    return objFun

def bayesOpt(objFun, bounds, plots=True):

    seed(12345)

    input_dim = len(bounds)

    # Select an specific kernel from GPy
    kernel = GPy.kern.RBF(input_dim, variance=.1, lengthscale=1) + GPy.kern.Bias(input_dim) # we add a bias kernel

    # --- Problem definition and optimization
    BO_model = GPyOpt.methods.BayesianOptimization(f=objFun,  # function to optimize
                                            kernel = kernel,               # pre-specified model
                                            bounds=bounds,                 # box-constrains of the problem
                                            acquisition='EI',             # Selects the Expected improvement
                                            numdata_initial_design=10,
                                            type_initial_design='random',   # latin desing of the initial points
                                            normalize = True)              # normalized y

    # Run the optimization
    max_iter = 10

    # --- Run the optimization                                              # evaluation budget
    ipdb.set_trace()
    BO_model.run_optimization(max_iter,                                   # Number of iterations
                                acqu_optimize_method = 'CMA',       # method to optimize the acq. function
                                acqu_optimize_restarts = 1,                # number of local optimizers
                                eps=10e-3,                        # secondary stop criteria (apart from the number of iterations)
                                true_gradients = True)                     # The gradients of the acquisition function are approximated (faster)

    # # --- Plots
    # if plots:
    #     objective_true.plot()
    #     BO_demo_2d.plot_acquisition()
    #     BO_demo_2d.plot_convergence()

    return BO_model

def advanced_optimization_2d(plots=True):
    import GPyOpt
    import GPy
    from numpy.random import seed
    seed(12345)

    # --- Objective function
    objective_true  = GPyOpt.fmodels.experiments2d.sixhumpcamel()             # true function
    objective_noisy = GPyOpt.fmodels.experiments2d.sixhumpcamel(sd = 0.1)     # noisy version
    bounds = objective_noisy.bounds                                           # problem constrains
    input_dim = len(bounds)


    # Select an specific kernel from GPy
    kernel = GPy.kern.RBF(input_dim, variance=.1, lengthscale=.1) + GPy.kern.Bias(input_dim) # we add a bias kernel


    # --- Problem definition and optimization
    BO_demo_2d = GPyOpt.methods.BayesianOptimization(f=objective_noisy.f,  # function to optimize
                                            kernel = kernel,               # pre-specified model
                                            bounds=bounds,                 # box-constrains of the problem
                                            acquisition='EI',             # Selects the Expected improvement
                                            acquisition_par = 2,           # parameter of the acquisition function
                                            numdata_initial_design = 15,    # 15 initial points
                                            type_initial_design='random',   # latin desing of the initial points
                                            model_optimize_interval= 2,    # The model is updated every two points are collected
                                            normalize = True)              # normalized y


    # Run the optimization
    max_iter = 20

    # --- Run the optimization                                              # evaluation budget
    BO_demo_2d.run_optimization(max_iter,                                   # Number of iterations
                                acqu_optimize_method = 'DIRECT',       # method to optimize the acq. function
                                acqu_optimize_restarts = 30,                # number of local optimizers
                                eps=10e-6,                        # secondary stop criteria (apart from the number of iterations)
                                true_gradients = True)                     # The gradients of the acquisition function are approximated (faster)


    # --- Plots
    if plots:
        objective_true.plot()
        BO_demo_2d.plot_acquisition()
        BO_demo_2d.plot_convergence()


    return BO_demo_2d