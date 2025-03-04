from .solvers import AdaptiveGridODESolver
import scipy as sci
import torch
import numpy as np

class AdaptiveMidpointSolver(AdaptiveGridODESolver):
    def __init__(self, func, y0, rtol, atol, step_size=None, theta=None, interp="linear", perturb=False, **unused_kwargs):
        super().__init__(func, y0, atol, step_size, theta, interp, perturb, **unused_kwargs)

    def _step_func(self, func, t0, dt, t1, y0):
        f0 = func(t0, y0)
        return y0+ dt *func(t0+dt/2, y0+dt/2*f0), f0
    
    def _step_func_adjoint(self, func, t0, dt, t1, y0):
        f0 = func(t0, y0)
        return y0+ dt *func(t0+dt/2, y0+dt/2*f0), f0

    def eval_estimator(self, t0, dt, t1, y0, y1):
        y0_np=y0.detach().cpu().numpy().reshape(-1)
        y1_np=y1.detach().cpu().numpy().reshape(-1)
        t0_np=t0.detach().cpu().numpy().reshape(-1)
        t1_np=t1.detach().cpu().numpy().reshape(-1)
        f0=self.func(t0,y0).detach().cpu().numpy().reshape(-1)
        
        def integrand(s):
            value=np.zeros(len(s))
            for i in range(len(s)):
                value[i]=(np.linalg.norm(0.5*f0-(s[i]-t0_np)/(t1_np-t0_np)*(y1_np-y0_np)/(t1_np-t0_np))**2)
            return value 
        esti=sci.integrate.fixed_quad(integrand,t0_np,t1_np)
        return torch.tensor(((t1_np-t0_np)**2)*abs(esti[0])+(t1_np-t0_np)**3*np.linalg.norm((y1_np-y0_np)/(t1_np-t0_np))**2).to(y0.device, y0.dtype)
    
    def eval_estimator_adjoint(self, t0, dt, t1, y0, y1):
        return self.eval_estimator(t0,dt,t1,y0,y1)
    
    def eval_estimator_eff(self,t0,dt,t1,y0,y1):
        return self.eval_estimator(t0,dt,t1,y0,y1)
    
    def eval_estimator_eff_adjoint(self,t0,dt,t1,y0,y1):
        return self.eval_estimator(t0,dt,t1,y0,y1)