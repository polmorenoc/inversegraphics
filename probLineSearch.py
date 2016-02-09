from scipy.special import erf
import numpy as np
from scipy.stats import mvn
import ipdb

def probLineSearch(func, x0, f0, df0, search_direction, alpha0,
 verbosity, outs, paras, var_f0, var_df0):
    # probLineSearch.m -- A probabilistic line search algorithm for nonlinear
    # optimization problems with noisy gradients. 
    #
    # == INPUTS ===============================================================
    # [f,f', var_f, var_df] = func(x) -- function handle 
    # input: 
    #     x -- column vectors (positions) (Dx1) 
    # output: 
    #     f -- scalar function values
    #     df -- vector gradients (Dx1)
    #     var_f -- estimated noise for function values (scalar)
    #     var_df -- estimated noise for gradients (Dx1)
    # x0 -- current position of optimizer (Dx1)
    # f0 -- function value at x0 (scalar, previous output y_tt)
    # df0 -- gradient at x0 ((Dx1), previous output dy_tt)
    # search_direction -- (- df(x0) does not need to be normalized)
    # alpha0: initial step size (scalar, previous output alpha0_out)
    # var_f0 -- variance of function values at x0. (scalar, previous output var_f_tt)
    # var_df0 -- variance of gradients at x0. ((Dx1), previous output var_df_tt)
    # verbosity -- level of stdout output. 
    #         0 -- no output
    #         1 -- steps, function values, state printed to stdout
    #         2 -- plots with only function values
    #         3 -- plots including Gaussian process and Wolfe condition beliefs.
    # paras -- possible parameters that func needs.
    # outs -- struct with collected statistics 
    #
    # == OUTPUTS ==============================================================
    # outs -- struct including counters and statistics
    # alpha0_out -- accepted stepsize * 1.3 (initial step size for next step)
    # x_tt -- accepted position
    # y_tt -- functin value at x_tt
    # dy_tt -- gradient at x_tt
    # var_f_tt -- variance of function values at x_tt
    # var_df_tt -- variance of gradients values at x_tt
    #

    # outs = {}

    # -- setup fixed parameters -----------------------------------------------
    # if ~isfield(outs, 'counter')
    # outs['counter'] = 1
    # end


    outs['alpha_stats'] = alpha0 # running average over accepted step sizes

    limit = 7 # maximum #function evaluations in one line search (+1)

    # constants for Wolfe conditions (must be chosen 0 < c1 < c2 < 1)
    c1 = 0.05   # <---- DECIDED FIXED 0.05
    c2 = 0.8    # <---- DECIDED FIXED 0.8
    # c2 = 0 extends until ascend location reached: lots of extrapolation
    # c2 = 1 accepts any point of increased gradient: almost no extrapolation

    WolfeThreshold = 0.3 # <---- DECIDED FIXED (0.3)
    # the new free parameter of this method relative to sgd: 
    # search is terminated when probability(Wolfe conditions hold) > WolfeThreshold
    # not sensitive between 0.1 and 0.8 (0 = accept everyting, 1= accept nothing)

    offset  = 10 # off-set, for numerical stability. 

    EXT = 1 # extrapolation factor
    tt  = 1 # initial step size in scaled space

    # -- set up GP ------------------------------------------------------------
    # create variables with shared scope. Ugly, but necessary because
    # matlab does not discover shared variables if they are first created in a
    # nested function.

    d2m = np.array([]); d3m = np.array([]); V = np.array([]); Vd = np.array([]); dVd = np.array([])
    m0  = np.array([]); dm0 = np.array([]); V0= np.array([]);Vd0 = np.array([]); dVd0= np.array([])
    V0f = np.array([]); Vd0f= np.array([]); V0df=np.array([]); Vd0df = np.array([])


    # kernel:
    k  = lambda a,b: ((np.minimum(a+offset,b+offset)**3)/3 + 0.5 * np.abs(a-b) * np.minimum(a+offset,b+offset)**2)
    kd =  lambda a,b: np.int32(a<b) * ((a+offset)**2)/2 + np.int32(a>=b) * (np.dot(a+offset,b+offset) - 0.5 * (b+offset)**2)
    dk =  lambda a,b: np.int32(a>b) * ((b+offset)**2)/2 + np.int32(a<=b) * (np.dot(a+offset,b+offset) - 0.5 * (a+offset)**2)
    dkd=  lambda a,b: np.minimum(a+offset,b+offset)

    # further derivatives
    ddk = lambda a,b: np.int32(a<=b) * (b-a)
    ddkd= lambda a,b: np.int32(a<=b)
    dddk= lambda a,b: -np.int32(a<=b)

    # -- helper functions -----------------------------------------------------
    GaussCDF = lambda z: 0.5 * (1 + erf(z/np.sqrt(2)))
    GaussPDF = lambda z: np.exp( - 0.5 * z**2 ) / np.sqrt(2*np.pi)
    EI       = lambda m,s,eta: (eta - m) * GaussCDF((eta-m)/s) + s * GaussPDF((eta-m)/s)

    # -- scale ----------------------------------------------------------------
    beta = np.abs(np.dot(search_direction.T,df0))
    # scale f and df according to 1/(beta*alpha0)

    # -- scaled noise ---------------------------------------------------------
    sigmaf  = np.sqrt(var_f0)/(alpha0*beta)

    sigmadf = np.sqrt(np.dot((search_direction**2).T,var_df0))/beta

    # -- initiate data storage ------------------------------------------------
    T = np.array([0])
    Y = np.array([0])
    dY = np.array(df0)[:,None]
    dY_projected = np.array([np.dot(df0.T,search_direction)/beta])
    Sigmaf = np.array([var_f0])
    Sigmadf = np.array(var_df0)[:,None]
    N = 1

    m = []
    d1m = []
    d2m = []
    d3m = []
    V = []
    Vd = []
    dVd = []
    m0 = []
    dm0 = []
    V0 = []
    Vd0 = []
    dVd0 = []
    V0f = []
    Vd0f = []
    V0df = []
    d0df = []
    # -- helper functions -----------------------------------------------------
    def updateGP(): # using multiscope variables to construct GP

        nonlocal m
        nonlocal d1m
        nonlocal d2m
        nonlocal d3m
        nonlocal V
        nonlocal Vd
        nonlocal dVd
        nonlocal m0
        nonlocal dm0
        nonlocal V0
        nonlocal Vd0
        nonlocal dVd0
        nonlocal V0f
        nonlocal Vd0f
        nonlocal V0df
        nonlocal Vd0df

        # build Gram matrix
        kTT   = np.zeros([N,N]);
        kdTT  = np.zeros([N,N]);
        dkdTT = np.zeros([N,N]);
        for i in range(N):
            for j in range(N):
                kTT[i,j]   = k(T[i],  T[j])
                kdTT[i,j]  = kd(T[i], T[j])
                dkdTT[i,j] = dkd(T[i],T[j])

        # build noise matrix
        Sig = sigmaf**2 * np.ones([2*N, 1]); Sig[N::] = sigmadf**2

        # build Gram matrix
        G = np.diag(Sig.ravel()) + np.r_[np.c_[kTT, kdTT], np.c_[kdTT.T, dkdTT]]


        A = np.linalg.solve(G, np.append(Y, dY_projected))

        # posterior mean function and all its derivatives
        m = lambda t: np.dot(np.concatenate([k(t, T.T)   ,  kd(t,  T.T)]), A)
        d1m = lambda t: np.dot(np.concatenate([dk(t, T.T)  , dkd(t,  T.T)]), A)
        d2m = lambda t: np.dot(np.concatenate([ddk(t, T.T) ,ddkd(t,  T.T)]), A)
        d3m = lambda t: np.dot(np.concatenate([dddk(t, T.T), np.zeros([N])]), A)

        # posterior marginal covariance between function and first derivative
        V = lambda t: k(t,t)   - np.dot(np.concatenate([k(t, T.T) ,  kd(t, T.T)]), np.linalg.solve(G , np.concatenate([k(t, T.T) , kd(t, T.T)]).T))
        Vd = lambda t: kd(t,t)  - np.dot(np.concatenate([k(t, T.T) ,  kd(t, T.T)]), np.linalg.solve(G , np.concatenate([dk(t, T.T),dkd(t, T.T)]).T))
        dVd = lambda t: dkd(t,t) - np.dot(np.concatenate([dk(t, T.T), dkd(t, T.T)]), np.linalg.solve(G , np.concatenate([dk(t, T.T),dkd(t, T.T)]).T))

        # belief at starting point, used for Wolfe conditions
        m0 = m(0)
        dm0 = d1m(0)
        V0 = V(0)
        Vd0 = Vd(0)
        dVd0 = dVd(0)

        # covariance terms with function (derivative) values at origin
        V0f   = lambda t: k(0,t)  - np.dot(np.concatenate([k(0, T.T) ,  kd(0, T.T)]) , np.linalg.solve(G, np.concatenate([k(t, T.T) , kd(t, T.T)]).T))
        Vd0f  = lambda t: dk(0,t) - np.dot(np.concatenate([dk(0, T.T), dkd(0, T.T)]) , np.linalg.solve(G , np.concatenate([k(t, T.T) , kd(t, T.T)]).T))
        V0df  = lambda t: kd(0,t) - np.dot(np.concatenate([k(0, T.T),   kd(0, T.T)]) , np.linalg.solve(G , np.concatenate([dk(t, T.T),dkd(t, T.T)]).T))
        Vd0df = lambda t: dkd(0,t)- np.dot(np.concatenate([dk(0, T.T), dkd(0, T.T)]) , np.linalg.solve(G , np.concatenate([dk(t, T.T),dkd(t, T.T)]).T))

    # -- update GP with new datapoint -----------------------------------------
    updateGP()

    x_tt = []
    y_tt = []
    dy_tt = []
    var_f_tt = []
    var_df_tt = []
    alpha0_out = []

    def make_outs(y, dy, var_f, var_df):

        nonlocal x_tt
        nonlocal y_tt
        nonlocal dy_tt
        nonlocal var_f_tt
        nonlocal var_df_tt
        nonlocal alpha0_out
        nonlocal outs
        x_tt      = x0 + tt*alpha0*search_direction # accepted position
        y_tt      = y*(alpha0*beta) + f0            # function value at accepted position
        dy_tt     = dy                              # gradient at accepted position
        var_f_tt  = var_f                         # variance of function value at accepted position
        var_df_tt = var_df                          # variance of gradients at accepted position

        # set new set size
        # next initial step size is 1.3 times larger than last accepted step size
        alpha0_out = tt*alpha0 * 1.3

        # running average for reset in case the step size becomes very small
        # this is a saveguard
        gamma = 0.9
        outs['alpha_stats'] = gamma*outs['alpha_stats'] + (1-gamma)*tt*alpha0;

        # reset NEXT initial step size to average step size if accepted step
        # size is 100 times smaller or larger than average step size
        if (alpha0_out > 1e2*outs['alpha_stats']) or (alpha0_out < 1e-2*outs['alpha_stats']):
            if verbosity > 0:
                print('making a very small step, resetting alpha0')
            alpha0_out = outs['alpha_stats'] # reset step size


    def probWolfe(t): # probability for Wolfe conditions to be fulfilled

        # marginal for Armijo condition
        ma  = m0 - m(t) + c1 * t * dm0
        Vaa = V0 + (c1 * t)**2 * dVd0 + V(t) + 2 * (c1 * t * (Vd0 - Vd0f(t)) - V0f(t))

        # marginal for curvature condition
        mb  = d1m(t) - c2 * dm0
        Vbb = c2**2 * dVd0 - 2 * c2 * Vd0df(t) + dVd(t)

        # covariance between conditions
        Vab = -c2 * (Vd0 + c1 * t * dVd0) + V0df(t) + c2 * Vd0f(t) + c1 * t * Vd0df(t) - Vd(t)

        if (Vaa < 1e-9) and (Vbb < 1e-9): # deterministic evaluations
            p = np.int32(ma >= 0) * np.int32(mb >= 0)
            return p, None

        # joint probability
        if Vaa <= 0 or Vbb <= 0:
            p   = 0
            p12 = np.array([0,0,0])
            return p,p12

        rho = Vab / np.sqrt(Vaa * Vbb)

        upper = 2 * c2 * ((np.abs(dm0)+2*np.sqrt(dVd0)-mb)/np.sqrt(Vbb) )
        # p = bvn(-ma / np.sqrt(Vaa), np.inf, -mb / np.sqrt(Vbb), upper, rho)
        _, p, _ = mvn.mvndst(np.array([-ma / np.sqrt(Vaa),-mb / np.sqrt(Vbb)]), np.array([np.inf,upper]), np.array([1, 2]), rho)

        # if nargout > 1:
            # individual marginal probabilities for each condition
            # (for debugging)

        p12 = np.array([1 - GaussCDF(-ma/np.sqrt(Vaa)), GaussCDF(upper)- GaussCDF(-mb/np.sqrt(Vbb)), Vab / np.sqrt(Vaa * Vbb)])

        return p, p12

    def cubicMinimum(ts):
        nonlocal d1m
        nonlocal d2m
        nonlocal d3m
        # mean belief at ts is a cubic function. It is defined up to a constant by
        d1mt = d1m(ts)
        d2mt = d2m(ts)
        d3mt = d3m(ts)

        a = 0.5 * d3mt
        b = d2mt - ts * d3mt
        c = d1mt - d2mt * ts + 0.5 * d3mt * ts**2

        if abs(d3mt) < 1e-9: # essentially a quadratic. Single extremum
           tm = - (d1mt - ts * d2mt) / d2mt
           return tm


        # compute the two possible roots:
        detmnt = b**2 - 4*a*c
        if detmnt < 0: # no roots
            tm = np.inf
            return tm

        LR = (-b - np.sign(a) * np.sqrt(detmnt)) / (2*a)  # left root
        RR = (-b + np.sign(a) * np.sqrt(detmnt)) / (2*a)  # right root

        # and the two values of the cubic at those points (up to constant)
        Ldt = LR - ts # delta t for left root
        Rdt = RR - ts # delta t for right root
        LCV = d1mt * Ldt + 0.5 * d2mt * Ldt**2 +  (d3mt * Ldt**3)/6  # left cubic value
        RCV = d1mt * Rdt + 0.5 * d2mt * Rdt**2 + (d3mt * Rdt**3)/6 # right cubic value

        if LCV < RCV:
            tm = LR
        else:
            tm = RR

        return tm

    # -- search (until budget used or converged) ------------------------------
    for N in range(2,limit+1):

        # -- evaluate function (change minibatch!) ----------------------------
        outs['counter'] = outs['counter'] + 1

        [y, dy, var_f, var_df] = func(x0 + tt*alpha0*search_direction) # y: function value at tt


        # # Test with matlab function TESTFUNCTION=3
        # if N==2:
        #     y = 140.9659
        #     dy = np.array([16.5853, 22.3174])
        #     var_f = 1
        #     var_df = np.array([1, 1])
        #
        # if N == 3:
        #     y = 133.2719
        #     dy = np.array([18.4566,22.1113])
        #     var_f = 1
        #     var_df = np.array([1,1])
        #
        # if N == 4:
        #     y = 118.0413
        #     dy = np.array([18.0786,21.8977])
        #     var_f = 1
        #     var_df = np.array([1,1])
        #
        # if N == 5:
        #     y = 85.3557
        #     dy = np.array([17.3779,18.2448])
        #     var_f = 1
        #     var_df = np.array([1,1])
        #
        # if N == 6:
        #     y = 43.5753
        #     dy = np.array([3.1085,11.2455])
        #     var_f = 1
        #     var_df = np.array([1,1])


        if np.isinf(y) or np.isnan(y):
            # this does not happen often, but still needs a fix
            # e.g. if function return inf or nan (probably too large step), 
            # evaluate again at 0.1*tt
            print('function values is inf or nan.')
        print("N " + str(N))
        # -- scale function output --------------------------------------------
        y = (y - f0)/(alpha0*beta)        # substract offset
        dy_projected = np.dot(dy.T,search_direction)/beta   # projected gradient
        
        # -- store output -----------------------------------------------------
        T = np.append(T,tt)
        Y = np.append(Y,y)

        dY = np.hstack([dY,dy.reshape([-1,1])])

        dY_projected = np.append(dY_projected, dy_projected)

        Sigmaf = np.append(Sigmaf, var_f)

        Sigmadf = np.hstack([Sigmadf,var_df.reshape([-1,1])])

        # if N == 7:
        #     ipdb.set_trace()
        # -- update GP with new datapoint -------------------------------------
        updateGP() # store, update belief

        # -- check last evaluated point for acceptance ------------------------
        (probWolfeVal, _) = probWolfe(tt)


        if probWolfeVal > WolfeThreshold: # are we done?
            if verbosity > 0:
                print('found acceptable point.')

            make_outs(y, dy, var_f, var_df)

            return outs, alpha0_out, y_tt, dy_tt, x_tt, var_f_tt, var_df_tt

        # -- find candidates (set up for EI) ----------------------------------
        # decide where to probe next: evaluate expected improvement and prob
        # Wolfe conditions at a number of candidate points, take the most promising one.
        
        # lowest mean evaluation, and its gradient (needed for EI):
        M  = np.zeros([N,1]) 
        dM = np.zeros([N,1])
        for l in range(N):
            M[l]  = m(T[l]) 
            dM[l] = d1m(T[l])

        minj = np.argmin(M)    # minm: lowest GP mean, minj: index in candidate list
        minm = M[minj]

        tmin        = T[minj]   # tt of point with lowest GP mean of function values
        dmin        = dM[minj]  # GP mean of gradient at lowest point
        
        # -- check this point as well for acceptance --------------------------
        if np.abs(dmin) < 1e-5 and Vd(tmin) < 1e-4: # nearly deterministic
            tt = tmin; dy = dY[:, minj]; y = Y[minj]; var_f = Sigmaf[minj]; var_df = Sigmadf[:, minj];
            print('found a point with almost zero gradient. Stopping, although Wolfe conditions not guaranteed.')
            make_outs(y, dy, var_f, var_df)

            return outs, alpha0_out, y_tt, dy_tt, x_tt, var_f_tt, var_df_tt
        
        # -- find candidates --------------------------------------------------
        # CANDIDATES 1: minimal means between all evaluations:
        # iterate through all `cells' (O[N]), check minimum mean locations.
        Tcand  = np.array([]) # positions of candidate points
        Mcand  = np.array([]) # means of candidate points
        Scand  = np.array([]) # standard deviation of candidate points
        Tsort  = np.sort(T) 
        Wolfes = np.array([]) # list of acceptable points.
        reeval = False

        for cel in range(N-1): # loop over cells
            Trep = Tsort[cel] + 1e-6 * (Tsort[cel+1] - Tsort[cel])
            cc   = cubicMinimum(Trep)

            # add point to candidate list if minimum lies in between T(cel) and T(cel+1)
            if cc > Tsort[cel] and cc < Tsort[cel+1]:
                Tcand = np.append(Tcand,cc)
                Mcand = np.append(Mcand,m(cc))
                Scand = np.append(Scand, np.sqrt(V(cc)))
                
            else: # no minimum, just take half-way
                if cel==1 and d1m(0) > 0: # only in first cell
                    if verbosity > 0:
                        print('function seems very steep, reevaluating close to start.')
                    reeval = True
                    Tcand = np.array([0.1 * (Tsort[cel] + Tsort[cel+1])])
                    Mcand = np.append(Mcand, m(0.1 * (Tsort[cel] + Tsort[cel+1])))
                    Scand = np.append(Scand, np.sqrt(V(0.1 * (Tsort[cel] + Tsort[cel+1]))))
                    break

            
            # check whether there is an acceptable point among already
            # evaluated points (since this changes each time the GP gets updated)
            probWolfeVal, _ = probWolfe(Tsort[cel])
            if cel > 1 and (probWolfeVal > WolfeThreshold):
                Wolfes = np.append(Wolfes,Tsort[cel]) # list of acceptable points.
        
        # -- check if at least on point is acceptable -------------------------
        if len(Wolfes) > 0:
            if verbosity > 0:
                # makePlot()
                print('found acceptable point.')
            
            # -- chose best point among Wolfes, return. -----------------------
            MWolfes = 0 * Wolfes;
            for o in range(len(Wolfes)):
                MWolfes[o] = m(Wolfes[o]) # compute GP means of acceptable points
            
            tt = Wolfes[MWolfes == np.min(MWolfes)]
                    
            # find corresponding gradient and variances
            dy = dY[:, T == tt].ravel(); y = Y[T == tt].ravel(); var_f = Sigmaf[T == tt].ravel(); var_df = Sigmadf[:, T==tt].ravel();
            make_outs(y, dy, var_f, var_df)
            # if dy.shape == (2,1):
            #     ipdb.set_trace()
            return outs, alpha0_out, y_tt, dy_tt, x_tt, var_f_tt, var_df_tt

            
        # Candidate 2: one extrapolation step

        if not reeval:
            Tcand = np.append(Tcand, np.max(T) + EXT)
            Mcand = np.append(Mcand, m(np.max(T) + EXT))

            Scand = np.append(Scand, np.sqrt(V(np.max(T)+EXT)))

        # -- discriminate candidates through EI and prob Wolfe ----------------
        EIcand = EI(Mcand, Scand, minm) # minm: lowest GP mean of collected evaluations (not candidates)
        PPcand = np.zeros(EIcand.shape)
        for ti in range(len(EIcand)):
            PPcand[ti], _ = probWolfe(Tcand[ti])


        idx_best = np.argmax(EIcand * PPcand) # find best candidate
        
        if Tcand[idx_best] == tt + EXT: # extrapolating. Extend extrapolation step
           EXT = 2 * EXT
        
        tt = Tcand[idx_best]
        
        # makePlot()

    # -- algorithm reached limit without finding acceptable point. ------------
    # Evaluate a final time, return "best" point (one with lowest function value)

    outs['counter'] = outs['counter'] + 1


    [y, dy, var_f, var_df] = func(x0 + tt*alpha0*search_direction) # y: function value at tt

    if np.isinf(y) or np.isnan(y):
        # this does not happen often, but still needs a fix
        # e.g. if function return inf or nan (probably too large step), 
        # evaluate again at 0.1*tt
        print('function values is inf or nan.')

    # -- scale function output ------------------------------------------------
    y            = (y - f0)/(alpha0*beta) # substract offset
    dy_projected = np.dot(dy.T,search_direction)/beta # projected gradient at tt; 

    # -- store output -----------------------------------------------------
    T = np.append(T,tt)
    Y = np.append(Y,y)
    dY = np.hstack([dY,dy.reshape([-1,1])])
    dY_projected = np.append(dY_projected, dy_projected)
    N = limit + 1
    Sigmaf = np.append(Sigmaf, var_f)
    Sigmadf = np.hstack([Sigmadf,var_df.reshape([-1,1])])

    # -- update GP with new datapoint -----------------------------------------
    updateGP()

    # -- check last evaluated point for acceptance ----------------------------
    probWolfeVal, _ = probWolfe(tt)
    if probWolfeVal > WolfeThreshold: # are we done?
        if verbosity > 0:
            print('found acceptable point right at end of budget. Phew!')
        make_outs(y, dy, var_f, var_df)

        return outs, alpha0_out, y_tt, dy_tt, x_tt, var_f_tt, var_df_tt


    # -- return point with lowest mean ----------------------------------------
    M  = np.ones([N,1])*np.inf
    for ii in range(N):
        M[ii] = m(T[ii]) # compute all GP means of all evaluated locations

    minj = np.argmin(M)  # index of minimal GP mean
    if verbosity > 0:
        print('reached evaluation limit. Returning ''best'' known point.');


    # find corresponding tt, gradient and noise
    tt = T[minj]; dy = dY[:, minj]; y = Y[minj];  var_f = Sigmaf[minj]; var_df = Sigmadf[:, minj]
    make_outs(y, dy, var_f, var_df)

    return outs, alpha0_out, y_tt, dy_tt, x_tt, var_f_tt, var_df_tt