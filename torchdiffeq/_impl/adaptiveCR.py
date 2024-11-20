from .solvers import AdaptiveGridODESolver
import scipy as sci
import torch

class AdaptiveCRSolver(AdaptiveGridODESolver):
    def __init__(self, func, y0, rtol, atol, dtfunc=None, dyfunc=None, step_size=None, theta=None, interp="linear", perturb=False, **unused_kwargs):
        super().__init__(func, y0, atol, step_size, theta, interp, perturb, **unused_kwargs)
        self.dtfunc=dtfunc
        self.dyfunc=dyfunc

    def _step_func(self, func, t0, dt, t1, y0):
        def optim_f(y):
            return y-y0-(t1-t0)*(func(t1,y)+func(t0,y0))/2
        root=sci.optimize.fsolve(optim_f,y0)
        return root, func(t0,y0)
    
    def eval_estimator(self, t0, dt, t1, y0, y1):
        esti=sci.integrate.quad(lambda t: torch.linalg.norm((self.dtfunc(t,y0+(y1-y0)/(t1-t0)*(t-t0))+torch.matmul(self.dyfunc(t,y0+(y1-y0)/(t1-t0)*(t-t0)),(y1-y0)/(t1-t0))))**2,t0,t1)
        return ((t1-t0)**2)*abs(esti[0])