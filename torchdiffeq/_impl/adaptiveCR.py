from .solvers import AdaptiveGridODESolver
import scipy as sci
import torch
import numpy as np

class AdaptiveCRSolver(AdaptiveGridODESolver):
    def __init__(self, func, y0, rtol, atol, step_size=None, theta=None, interp="linear", perturb=False, **unused_kwargs):
        super().__init__(func, y0, atol, step_size, theta, interp, perturb, **unused_kwargs)

    def _step_func(self, func, t0, dt, t1, y0):
        y0_np=y0.detach().cpu().numpy().reshape(-1)
        def np_func(t, y):
            t = torch.tensor(t).to(y0.device, y0.dtype)
            y = torch.reshape(torch.tensor(y).to(y0.device, y0.dtype), y0.shape)
            with torch.no_grad():
                f = func(t, y)
            return f.detach().cpu().numpy().reshape(-1)
        def optim_f(y):
            return y-y0_np-(t1.detach().cpu().numpy().reshape(-1)-t0.detach().cpu().numpy().reshape(-1))*(np_func(t1.detach().cpu().numpy().reshape(-1),y)+np_func(t0.detach().cpu().numpy().reshape(-1),y0_np))/2
        root=sci.optimize.root(optim_f,y0_np)
        tr=torch.reshape(torch.tensor(root["x"]).to(y0.device, y0.dtype), y0.shape)
        return tr, func(t0,y0)
    
    def eval_estimator(self, t0, dt, t1, y0, y1):
        y0_np=y0.detach().cpu().numpy().reshape(-1)
        y1_np=y1.detach().cpu().numpy().reshape(-1)
        t0_np=t0.detach().cpu().numpy().reshape(-1)
        t1_np=t1.detach().cpu().numpy().reshape(-1)
        def dtfunc(t,y):
            def tfunc(t):
                return self.func(t,y)
            
            return torch.autograd.functional.jacobian(tfunc,t)
        def np_dtfunc(t,y):
            t = torch.tensor(t).to(y0.device, y0.dtype)
            y = torch.reshape(torch.tensor(y).to(y0.device, y0.dtype), y0.shape)
            with torch.no_grad():
                f = dtfunc(t, y)
            return f.detach().cpu().numpy().reshape(-1)
        def dyfunc(t,y,v):
            def yfunc(y):
                return self.func(t,y)
            output, jvp = torch.autograd.functional.jvp(yfunc,y,v)
            return jvp
        def np_dyfunc(t,y,v):
            t = torch.tensor(t).to(y0.device, y0.dtype)
            y = torch.reshape(torch.tensor(y).to(y0.device, y0.dtype), y0.shape)
            v = torch.reshape(torch.tensor(v).to(y0.device, y0.dtype), y0.shape)
            with torch.no_grad():
                f = dyfunc(t, y, v)
            return f.detach().cpu().numpy().reshape(-1)
        def integrand(s):
            value=np.zeros(len(s))
            for i in range(len(s)):
                value[i]=(np.linalg.norm((np_dtfunc(s[i],y0_np+(y1_np-y0_np)/(t1_np-t0_np)*(s[i]-t0_np))+np_dyfunc(s[i],y0_np+(y1_np-y0_np)/(t1_np-t0_np)*(s[i]-t0_np),(y1_np-y0_np)/(t1_np-t0_np)))))**2
            return value 
        esti=sci.integrate.fixed_quad(integrand,t0_np,t1_np)
        return torch.tensor(((t1_np-t0_np)**2)*abs(esti[0])).to(y0.device, y0.dtype)