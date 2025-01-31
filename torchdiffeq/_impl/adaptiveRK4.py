from .solvers import AdaptiveGridODESolver
import scipy as sci
import torch
import numpy as np
from .rk_common import rk4_alt_step_func

class AdaptiveRK4Solver(AdaptiveGridODESolver):
    def __init__(self, func, y0, rtol, atol, step_size=None, theta=None, interp="linear", perturb=False, **unused_kwargs):
        super().__init__(func, y0, atol, step_size, theta, interp, perturb, **unused_kwargs)

    def _step_func(self, func, t0, dt, t1, y0):
        f0 = func(t0, y0)
        return y0+rk4_alt_step_func(func, t0, dt, t1, y0, f0=f0, perturb=self.perturb), f0
    
    def _step_func_adjoint(self, func, t0, dt, t1, y0):
        f0 = func(t0, y0)
        return y0+rk4_alt_step_func(func, t0, dt, t1, y0, f0=f0, perturb=self.perturb), f0

    def eval_estimator(self, t0, dt, t1, y0, y1):
        y0_np=y0.detach().cpu().numpy().reshape(-1)
        y1_np=y1.detach().cpu().numpy().reshape(-1)
        t0_np=t0.detach().cpu().numpy().reshape(-1)
        t1_np=t1.detach().cpu().numpy().reshape(-1)
        
       
        def np_dtfunc(t,y):
            t = torch.tensor(t).to(y0.device, y0.dtype)
            y = torch.reshape(torch.tensor(y).to(y0.device, y0.dtype), y0.shape)
            with torch.no_grad():
                f = self.dtfunc(t, y)
            return f.detach().cpu().numpy().reshape(-1)
        
        
        def np_dyfunc(t,y,v):
            t = torch.tensor(t).to(y0.device, y0.dtype)
            y = torch.reshape(torch.tensor(y).to(y0.device, y0.dtype), y0.shape)
            v = torch.reshape(torch.tensor(v).to(y0.device, y0.dtype), y0.shape)
            with torch.no_grad():
                f = self.dyfunc(t, y, v)
            return f.detach().cpu().numpy().reshape(-1)
        
        def integrand(s):
            value=np.zeros(len(s))
            for i in range(len(s)):
                value[i]=(np.linalg.norm((np_dtfunc(s[i],y0_np+(y1_np-y0_np)/(t1_np-t0_np)*(s[i]-t0_np))+np_dyfunc(s[i],y0_np+(y1_np-y0_np)/(t1_np-t0_np)*(s[i]-t0_np),(y1_np-y0_np)/(t1_np-t0_np)))))**2
            return value 
        esti=sci.integrate.fixed_quad(integrand,t0_np,t1_np)
        return torch.tensor(((t1_np-t0_np)**2)*abs(esti[0])).to(y0.device, y0.dtype)
    
    def eval_estimator_adjoint(self, t0, dt, t1, y0, y1):
        num_paras=sum(p.numel() for p in self.adjoint_params)
        y0_np=y0[range(len(y0)-num_paras)].detach().cpu().numpy().reshape(-1)
        y1_np=y1[range(len(y0)-num_paras)].detach().cpu().numpy().reshape(-1)
        t0_np=t0.detach().cpu().numpy().reshape(-1)
        t1_np=t1.detach().cpu().numpy().reshape(-1)

        def np_dtfunc(t,y):
            y = torch.reshape(torch.tensor(y).to(y0.device, y0.dtype), y0[range(len(y0)-num_paras)].shape)
            yfull=torch.cat((y,torch.zeros(num_paras)))
            t = torch.tensor(t).to(y0.device, y0.dtype)
            with torch.no_grad():
                f = self.dtfunc(t, yfull)
            return f.detach().cpu().numpy().reshape(-1)
        
        def np_dyfunc(t,y,v):
            t = torch.tensor(t).to(y0.device, y0.dtype)
            y = torch.reshape(torch.tensor(y).to(y0.device, y0.dtype), y0[range(len(y0)-num_paras)].shape)
            yfull=torch.cat((y,torch.zeros(num_paras)))
            v = torch.reshape(torch.tensor(v).to(y0.device, y0.dtype), y0[range(len(y0)-num_paras)].shape)
            vfull=torch.cat((v,torch.zeros(num_paras)))
            with torch.no_grad():
                f = self.dyfunc(t, yfull, vfull)
            return f.detach().cpu().numpy().reshape(-1)
        
        def integrand(s):
            value=np.zeros(len(s))
            for i in range(len(s)):
                value[i]=(np.linalg.norm((np_dtfunc(s[i],y0_np+(y1_np-y0_np)/(t1_np-t0_np)*(s[i]-t0_np))+np_dyfunc(s[i],y0_np+(y1_np-y0_np)/(t1_np-t0_np)*(s[i]-t0_np),(y1_np-y0_np)/(t1_np-t0_np)))))**2
            return value 
        esti=sci.integrate.fixed_quad(integrand,t0_np,t1_np)
        return torch.tensor(((t1_np-t0_np)**2)*abs(esti[0])).to(y0.device, y0.dtype)
    
    def eval_estimator_eff(self,t0,dt,t1,y0,y1):
        y0_np=y0.detach().cpu().numpy().reshape(-1)
        y1_np=y1.detach().cpu().numpy().reshape(-1)
        t0_np=t0.detach().cpu().numpy().reshape(-1)
        t1_np=t1.detach().cpu().numpy().reshape(-1)

        def np_func(t, y):
            t = torch.tensor(t).to(y0.device, y0.dtype)
            y = torch.reshape(torch.tensor(y).to(y0.device, y0.dtype), y0.shape)
            with torch.no_grad():
                f = self.func(t, y)
            return f.detach().cpu().numpy().reshape(-1)
        def integrand(s):
            value=np.zeros(len(s))
            for i in range(len(s)):
                value[i]=(np.linalg.norm((y1_np-y0_np)/(t1_np-t0_np)-np_func(s[i],y0_np+(y1_np-y0_np)/(t1_np-t0_np)*(s[i]-t0_np))))**2
            return value
        esti=sci.integrate.fixed_quad(integrand,t0_np,t1_np)
        return torch.tensor(esti[0]).to(y0.device, y0.dtype)
    
    def eval_estimator_eff_adjoint(self,t0,dt,t1,y0,y1):
        num_paras=sum(p.numel() for p in self.adjoint_params)
        y0_np=y0[range(len(y0)-num_paras)].detach().cpu().numpy().reshape(-1)
        y1_np=y1[range(len(y0)-num_paras)].detach().cpu().numpy().reshape(-1)
        t0_np=t0.detach().cpu().numpy().reshape(-1)
        t1_np=t1.detach().cpu().numpy().reshape(-1)

        def np_func(t, y):
            t = torch.tensor(t).to(y0.device, y0.dtype)
            y = torch.reshape(torch.tensor(y).to(y0.device, y0.dtype), y0[range(len(y0)-num_paras)].shape)
            with torch.no_grad():
                yfull=torch.cat((y,torch.zeros(num_paras)))
                f = self.func(t, yfull)[range(len(y0)-num_paras)]
            return f.detach().cpu().numpy().reshape(-1)
        def integrand(s):
            value=np.zeros(len(s))
            for i in range(len(s)):
                value[i]=(np.linalg.norm((y1_np-y0_np)/(t1_np-t0_np)-np_func(s[i],y0_np+(y1_np-y0_np)/(t1_np-t0_np)*(s[i]-t0_np))))**2
            return value
        esti=sci.integrate.fixed_quad(integrand,t0_np,t1_np)
        return torch.tensor(esti[0]).to(y0.device, y0.dtype)