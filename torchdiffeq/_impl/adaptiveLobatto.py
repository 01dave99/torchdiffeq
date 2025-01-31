from .solvers import AdaptiveGridODESolver
import scipy as sci
import torch
import numpy as np

class AdaptiveLobattoSolver(AdaptiveGridODESolver):
    def __init__(self, func, y0, rtol, atol, step_size=None, theta=None, interp="linear", perturb=False, **unused_kwargs):
        super().__init__(func, y0, atol, step_size, theta, interp, perturb, **unused_kwargs)
        self.yhalf=0
    def _step_func(self, func, t0, dt, t1, y0):
        y0_np=y0.detach().cpu().numpy().reshape(-1)
        t0_np=t0.detach().cpu().numpy().reshape(-1)
        t1_np=t1.detach().cpu().numpy().reshape(-1)
        dt=t1_np-t0_np
        def np_func(t, y):
            t = torch.tensor(t).to(y0.device, y0.dtype)
            y = torch.reshape(torch.tensor(y).to(y0.device, y0.dtype), y0.shape)
            with torch.no_grad():
                f = func(t, y)
            return f.detach().cpu().numpy().reshape(-1)
        def optim_f(k):
           k0 = np_func(t0_np,y0_np)
           k1 = k[0:len(y0_np)]
           k2 = k[len(y0_np):2*len(y0_np)]
           f1 = np_func(t0_np+0.5*dt,y0_np+dt*(5/24* k0+8/24*k1-1/24*k2))
           f2 = np_func(t1_np,y0_np+dt*(4/24* k0+16/24*k1+4/24*k2))
           f = np.concatenate((f1,f2))
           return k-f
        k=sci.optimize.root(optim_f,[np_func(t0_np,y0_np),np_func(t0_np,y0_np)])
        y1=y0_np+dt*(4/24* np_func(t0_np,y0_np)+16/24*k["x"][0:len(y0_np)]+4/24*k["x"][len(y0_np):2*len(y0_np)])
        self.yhalf=y0_np+dt*(5/24* np_func(t0_np,y0_np)+8/24*k["x"][0:len(y0_np)]-1/24*k["x"][len(y0_np):2*len(y0_np)])
        tr=torch.reshape(torch.tensor(y1).to(y0.device, y0.dtype), y0.shape)
        return tr, func(t0,y0)
    
    def _step_func_adjoint(self, func, t0, dt, t1, y0):
        num_paras=sum(p.numel() for p in self.adjoint_params)
        y0_np=y0[range(len(y0)-num_paras)].detach().cpu().numpy().reshape(-1)
        t0_np=t0.detach().cpu().numpy().reshape(-1)
        t1_np=t1.detach().cpu().numpy().reshape(-1)
        dt=t1_np-t0_np
        def np_func(t, y):
            t = torch.tensor(t).to(y0.device, y0.dtype)
            y = torch.reshape(torch.tensor(y).to(y0.device, y0.dtype), y0[range(len(y0)-num_paras)].shape)
            with torch.no_grad():
                yfull=torch.cat((y,torch.zeros(num_paras)))
                f = func(t, yfull)[range(len(y0)-num_paras)]
            return f.detach().cpu().numpy().reshape(-1)
        def optim_f(k):
           k0 = np_func(t0_np,y0_np)
           k1 = k[0:len(y0_np)]
           k2 = k[len(y0_np):2*len(y0_np)]
           f1 = np_func(t0_np+0.5*dt,y0_np+dt*(5/24* k0+8/24*k1-1/24*k2))
           f2 = np_func(t1_np,y0_np+dt*(4/24* k0+16/24*k1+4/24*k2))
           f = np.concatenate((f1,f2))
           return k-f
        k=sci.optimize.root(optim_f,[np_func(t0_np,y0_np),np_func(t0_np,y0_np)])
        y1=y0_np+dt*(4/24* np_func(t0_np,y0_np)+16/24*k["x"][0:len(y0_np)]+4/24*k["x"][len(y0_np):2*len(y0_np)])
        tr=torch.reshape(torch.tensor(y1).to(y0.device, y0.dtype), y0[range(len(y0)-num_paras)].shape)
        yhalf=y0_np+dt*(5/24* np_func(t0_np,y0_np)+8/24*k["x"][0:len(y0_np)]-1/24*k["x"][len(y0_np):2*len(y0_np)])
        #restlichen variablen:
        frest0=func(t0,y0)[range(-num_paras,0)]
        fresthalf=func(t0+0.5*(t1-t0),torch.cat((torch.reshape(torch.tensor(yhalf).to(y0.device, y0.dtype), y0[range(len(y0)-num_paras)].shape),torch.zeros(num_paras))))[range(-num_paras,0)]
        frest1=func(t1,torch.cat((tr,torch.zeros(num_paras))))[range(-num_paras,0)]
        rest=y0[range(-num_paras,0)]+torch.tensor(dt,dtype=y0.dtype)*(4/24* frest0+16/24*fresthalf+4/24*frest1)
        rest_half=y0[range(-num_paras,0)]+torch.tensor(dt,dtype=y0.dtype)*(5/24* frest0+8/24*fresthalf-1/24*frest1)
        
        rest_half=rest_half.detach().cpu().numpy().reshape(-1)
        self.yhalf=np.concatenate((yhalf,rest_half))
        result=torch.cat((tr,rest))
        return result, func(t0,y0)
    
    def eval_estimator(self, t0, dt, t1, y0, y1):
        y0_np=y0.detach().cpu().numpy().reshape(-1)
        y1_np=y1.detach().cpu().numpy().reshape(-1)
        t0_np=t0.detach().cpu().numpy().reshape(-1)
        t1_np=t1.detach().cpu().numpy().reshape(-1)
        thalf=(t0_np+t1_np)/2.
        
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
        def interpolant(t):
            return y0_np*(t-thalf)*(t-t1_np)/((t0_np-thalf)*(t0_np-t1_np))+self.yhalf*(t-t0_np)*(t-t1_np)/((thalf-t0_np)*(thalf-t1_np))+y1_np*(t-t0_np)*(t-thalf)/((t1_np-t0_np)*(t1_np-thalf))
        def dt_interpolant(t):
            return y0_np*((t-thalf)+(t-t1_np))/((t0_np-thalf)*(t0_np-t1_np))+self.yhalf*((t-t0_np)+(t-t1_np))/((thalf-t0_np)*(thalf-t1_np))+y1_np*((t-t0_np)+(t-thalf))/((t1_np-t0_np)*(t1_np-thalf))
        def d2t_interpolant(t):
            return y0_np*2./((t0_np-thalf)*(t0_np-t1_np))+self.yhalf*2./((thalf-t0_np)*(thalf-t1_np))+y1_np*2./((t1_np-t0_np)*(t1_np-thalf))
        def integrand(s):
            value=np.zeros(len(s))
            for i in range(len(s)):
                value[i]=(np.linalg.norm(np_dtfunc(s[i],interpolant(s[i]))+np_dyfunc(s[i],interpolant(s[i]),dt_interpolant(s[i]))-d2t_interpolant(s[i])))**2
            return value 
        esti=sci.integrate.fixed_quad(integrand,t0_np,t1_np)
        return torch.tensor(((t1_np-t0_np)**2)*abs(esti[0])).to(y0.device, y0.dtype)
    
    def eval_estimator_adjoint(self, t0, dt, t1, y0, y1):
        num_paras=sum(p.numel() for p in self.adjoint_params)
        y0_np=y0[range(len(y0)-num_paras)].detach().cpu().numpy().reshape(-1)
        y1_np=y1[range(len(y0)-num_paras)].detach().cpu().numpy().reshape(-1)
        yhalf=self.yhalf[range(len(y0)-num_paras)]
        t0_np=t0.detach().cpu().numpy().reshape(-1)
        t1_np=t1.detach().cpu().numpy().reshape(-1)
        thalf=(t0_np+t1_np)/2.

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
        def interpolant(t):
            return y0_np*(t-thalf)*(t-t1_np)/((t0_np-thalf)*(t0_np-t1_np))+yhalf*(t-t0_np)*(t-t1_np)/((thalf-t0_np)*(thalf-t1_np))+y1_np*(t-t0_np)*(t-thalf)/((t1_np-t0_np)*(t1_np-thalf))
        def dt_interpolant(t):
            return y0_np*((t-thalf)+(t-t1_np))/((t0_np-thalf)*(t0_np-t1_np))+yhalf*((t-t0_np)+(t-t1_np))/((thalf-t0_np)*(thalf-t1_np))+y1_np*((t-t0_np)+(t-thalf))/((t1_np-t0_np)*(t1_np-thalf))
        def d2t_interpolant(t):
            return y0_np*2./((t0_np-thalf)*(t0_np-t1_np))+yhalf*2./((thalf-t0_np)*(thalf-t1_np))+y1_np*2./((t1_np-t0_np)*(t1_np-thalf))
        def integrand(s):
            value=np.zeros(len(s))
            for i in range(len(s)):
                value[i]=(np.linalg.norm(np_dtfunc(s[i],interpolant(s[i]))+np_dyfunc(s[i],interpolant(s[i]),dt_interpolant(s[i]))-d2t_interpolant(s[i])))**2
            return value 
        esti=sci.integrate.fixed_quad(integrand,t0_np,t1_np)
        return torch.tensor(((t1_np-t0_np)**2)*abs(esti[0])).to(y0.device, y0.dtype)
    
    def eval_estimator_eff(self,t0,dt,t1,y0,y1):
        y0_np=y0.detach().cpu().numpy().reshape(-1)
        y1_np=y1.detach().cpu().numpy().reshape(-1)
        t0_np=t0.detach().cpu().numpy().reshape(-1)
        t1_np=t1.detach().cpu().numpy().reshape(-1)
        thalf=(t0_np+t1_np)/2.

        def np_func(t, y):
            t = torch.tensor(t).to(y0.device, y0.dtype)
            y = torch.reshape(torch.tensor(y).to(y0.device, y0.dtype), y0.shape)
            with torch.no_grad():
                f = self.func(t, y)
            return f.detach().cpu().numpy().reshape(-1)
        

        def interpolant(t):
            return y0_np*(t-thalf)*(t-t1_np)/((t0_np-thalf)*(t0_np-t1_np))+self.yhalf*(t-t0_np)*(t-t1_np)/((thalf-t0_np)*(thalf-t1_np))+y1_np*(t-t0_np)*(t-thalf)/((t1_np-t0_np)*(t1_np-thalf))
        def dt_interpolant(t):
            return y0_np*((t-thalf)+(t-t1_np))/((t0_np-thalf)*(t0_np-t1_np))+self.yhalf*((t-t0_np)+(t-t1_np))/((thalf-t0_np)*(thalf-t1_np))+y1_np*((t-t0_np)+(t-thalf))/((t1_np-t0_np)*(t1_np-thalf))
        
        def integrand(s):
            value=np.zeros(len(s))
            for i in range(len(s)):
                value[i]=(np.linalg.norm(dt_interpolant(s[i])-np_func(s[i],interpolant(s[i]))))**2
            return value
        esti=sci.integrate.fixed_quad(integrand,t0_np,t1_np)
        return torch.tensor(esti[0]).to(y0.device, y0.dtype)
    
    def eval_estimator_eff_adjoint(self,t0,dt,t1,y0,y1):
        num_paras=sum(p.numel() for p in self.adjoint_params)
        y0_np=y0[range(len(y0)-num_paras)].detach().cpu().numpy().reshape(-1)
        y1_np=y1[range(len(y0)-num_paras)].detach().cpu().numpy().reshape(-1)
        yhalf=self.yhalf[range(len(y0)-num_paras)]
        t0_np=t0.detach().cpu().numpy().reshape(-1)
        t1_np=t1.detach().cpu().numpy().reshape(-1)
        thalf=(t0_np+t1_np)/2.

        def np_func(t, y):
            t = torch.tensor(t).to(y0.device, y0.dtype)
            y = torch.reshape(torch.tensor(y).to(y0.device, y0.dtype), y0[range(len(y0)-num_paras)].shape)
            with torch.no_grad():
                yfull=torch.cat((y,torch.zeros(num_paras)))
                f = self.func(t, yfull)[range(len(y0)-num_paras)]
            return f.detach().cpu().numpy().reshape(-1)
        
        def interpolant(t):
            return y0_np*(t-thalf)*(t-t1_np)/((t0_np-thalf)*(t0_np-t1_np))+yhalf*(t-t0_np)*(t-t1_np)/((thalf-t0_np)*(thalf-t1_np))+y1_np*(t-t0_np)*(t-thalf)/((t1_np-t0_np)*(t1_np-thalf))
        def dt_interpolant(t):
            return y0_np*((t-thalf)+(t-t1_np))/((t0_np-thalf)*(t0_np-t1_np))+yhalf*((t-t0_np)+(t-t1_np))/((thalf-t0_np)*(thalf-t1_np))+y1_np*((t-t0_np)+(t-thalf))/((t1_np-t0_np)*(t1_np-thalf))
        
        def integrand(s):
            value=np.zeros(len(s))
            for i in range(len(s)):
                value[i]=(np.linalg.norm(dt_interpolant(s[i])-np_func(s[i],interpolant(s[i]))))**2
            return value
        esti=sci.integrate.fixed_quad(integrand,t0_np,t1_np)
        return torch.tensor(esti[0]).to(y0.device, y0.dtype)