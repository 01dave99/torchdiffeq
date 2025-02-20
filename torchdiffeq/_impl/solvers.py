import abc
import torch
from .event_handling import find_event
from .misc import _handle_unused_kwargs, _TupleFunc, _ReverseFunc
from pathlib import Path
import matplotlib.pyplot as plt


class AdaptiveStepsizeODESolver(metaclass=abc.ABCMeta):
    def __init__(self, dtype, y0, norm, **unused_kwargs):
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.y0 = y0
        self.dtype = dtype

        self.norm = norm

    def _before_integrate(self, t):
        pass

    @abc.abstractmethod
    def _advance(self, next_t):
        raise NotImplementedError

    @classmethod
    def valid_callbacks(cls):
        return set()

    def integrate(self, t):
        solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        solution[0] = self.y0
        t = t.to(self.dtype)
        self._before_integrate(t)
        for i in range(1, len(t)):
            solution[i] = self._advance(t[i])
        return solution


class AdaptiveStepsizeEventODESolver(AdaptiveStepsizeODESolver, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def _advance_until_event(self, event_fn):
        raise NotImplementedError

    def integrate_until_event(self, t0, event_fn):
        t0 = t0.to(self.y0.device, self.dtype)
        self._before_integrate(t0.reshape(-1))
        event_time, y1 = self._advance_until_event(event_fn)
        solution = torch.stack([self.y0, y1], dim=0)
        return event_time, solution


class FixedGridODESolver(metaclass=abc.ABCMeta):
    order: int

    def __init__(self, func, y0, step_size=None, grid_constructor=None, interp="linear", perturb=False, **unused_kwargs):
        self.atol = unused_kwargs.pop('atol')
        unused_kwargs.pop('rtol', None)
        unused_kwargs.pop('norm', None)
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.dtype = y0.dtype
        self.device = y0.device
        self.step_size = step_size
        self.interp = interp
        self.perturb = perturb

        if step_size is None:
            if grid_constructor is None:
                self.grid_constructor = lambda f, y0, t: t
            else:
                self.grid_constructor = grid_constructor
        else:
            if grid_constructor is None:
                self.grid_constructor = self._grid_constructor_from_step_size(step_size)
            else:
                raise ValueError("step_size and grid_constructor are mutually exclusive arguments.")

    @classmethod
    def valid_callbacks(cls):
        return {'callback_step'}

    @staticmethod
    def _grid_constructor_from_step_size(step_size):
        def _grid_constructor(func, y0, t):
            start_time = t[0]
            end_time = t[-1]

            niters = torch.ceil((end_time - start_time) / step_size + 1).item()
            t_infer = torch.arange(0, niters, dtype=t.dtype, device=t.device) * step_size + start_time
            t_infer[-1] = t[-1]

            return t_infer
        return _grid_constructor

    @abc.abstractmethod
    def _step_func(self, func, t0, dt, t1, y0):
        pass

    def integrate(self, t):
        time_grid = self.grid_constructor(self.func, self.y0, t)
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]

        solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        solution[0] = self.y0

        j = 1
        y0 = self.y0
        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            dt = t1 - t0
            self.func.callback_step(t0, y0, dt)
            dy, f0 = self._step_func(self.func, t0, dt, t1, y0)
            y1 = y0 + dy

            while j < len(t) and t1 >= t[j]:
                if self.interp == "linear":
                    solution[j] = self._linear_interp(t0, t1, y0, y1, t[j])
                elif self.interp == "cubic":
                    f1 = self.func(t1, y1)
                    solution[j] = self._cubic_hermite_interp(t0, y0, f0, t1, y1, f1, t[j])
                else:
                    raise ValueError(f"Unknown interpolation method {self.interp}")
                j += 1
            y0 = y1

        return solution

    def integrate_until_event(self, t0, event_fn):
        assert self.step_size is not None, "Event handling for fixed step solvers currently requires `step_size` to be provided in options."

        t0 = t0.type_as(self.y0.abs())
        y0 = self.y0
        dt = self.step_size

        sign0 = torch.sign(event_fn(t0, y0))
        max_itrs = 20000
        itr = 0
        while True:
            itr += 1
            t1 = t0 + dt
            dy, f0 = self._step_func(self.func, t0, dt, t1, y0)
            y1 = y0 + dy

            sign1 = torch.sign(event_fn(t1, y1))

            if sign0 != sign1:
                if self.interp == "linear":
                    interp_fn = lambda t: self._linear_interp(t0, t1, y0, y1, t)
                elif self.interp == "cubic":
                    f1 = self.func(t1, y1)
                    interp_fn = lambda t: self._cubic_hermite_interp(t0, y0, f0, t1, y1, f1, t)
                else:
                    raise ValueError(f"Unknown interpolation method {self.interp}")
                event_time, y1 = find_event(interp_fn, sign0, t0, t1, event_fn, float(self.atol))
                break
            else:
                t0, y0 = t1, y1

            if itr >= max_itrs:
                raise RuntimeError(f"Reached maximum number of iterations {max_itrs}.")
        solution = torch.stack([self.y0, y1], dim=0)
        return event_time, solution

    def _cubic_hermite_interp(self, t0, y0, f0, t1, y1, f1, t):
        h = (t - t0) / (t1 - t0)
        h00 = (1 + 2 * h) * (1 - h) * (1 - h)
        h10 = h * (1 - h) * (1 - h)
        h01 = h * h * (3 - 2 * h)
        h11 = h * h * (h - 1)
        dt = (t1 - t0)
        return h00 * y0 + h10 * dt * f0 + h01 * y1 + h11 * dt * f1

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)

class AdaptiveGridODESolver(FixedGridODESolver):

    def __init__(self, func, y0, atol, step_size, theta, interp="linear", perturb=False, initial_grid=None,save_last_grid=False, faster_adj_solve=False,adjoint_params=None, conv_ana=False,file_id="None",max_nodes=100000,original_func=None,t_requires_grad=False,shapes=None,efficient=False, **unused_kwargs):
        self.atol = atol
        unused_kwargs.pop('rtol', None)
        unused_kwargs.pop('norm', None)
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.dtype = y0.dtype
        self.device = y0.device
        self.step_size = step_size
        self.interp = interp
        self.perturb = perturb
        self.initial_grid= initial_grid
        self.save_last_grid=save_last_grid
        self.grid_constructor = self._grid_constructor_from_step_size(step_size)
        self.theta=theta
        self.conv_ana=conv_ana
        self.file_id=file_id
        self.faster_adj_solve=faster_adj_solve
        self.adjoint_params=adjoint_params
        self.max_nodes=max_nodes
        self.yhalf=None
        self.efficient=efficient
        if original_func is not None:
            self.original_func=_ReverseFunc(original_func, mul=-1.0)
            def augmented_dynamics_temp(t, y_aug):
                # Dynamics of the original system augmented with
                # the adjoint wrt y, and an integrator wrt t and args.
                y = y_aug[1]
                adj_y = y_aug[2]
                

                with torch.enable_grad():

                   
                    func_eval = self.original_func(t, y)

                    # Workaround for PyTorch bug #39784
                    _t = torch.as_strided(t, (), ())  # noqa
                    _y = torch.as_strided(y, (), ())  # noqa
                    _params = tuple(torch.as_strided(param, (), ()) for param in tuple(adjoint_params))  # noqa

                    vjp_t, vjp_y, *vjp_params = torch.autograd.grad(
                        func_eval, (t, y) + tuple(adjoint_params), -adj_y,
                        allow_unused=True, retain_graph=True,create_graph=True
                    )

                # autograd.grad returns None if no gradient, set to zero.
                vjp_t = torch.zeros_like(t) if vjp_t is None else vjp_t
                vjp_t = torch.zeros_like(t) if not t_requires_grad else vjp_t
                vjp_y = torch.zeros_like(y) if vjp_y is None else vjp_y
                vjp_params = [torch.zeros_like(param) if vjp_param is None else vjp_param
                              for param, vjp_param in zip(tuple(adjoint_params), vjp_params)]

                return (vjp_t, func_eval, vjp_y, *vjp_params)
            augmented_dynamics_temp=_TupleFunc(augmented_dynamics_temp,shapes)

            if self.faster_adj_solve:
                num_paras=sum(p.numel() for p in self.adjoint_params)
                def augmented_dynamics_temp2(t, y_aug):
                    return augmented_dynamics_temp(t,y_aug)[range(len(self.y0)-num_paras)]
            else:
                augmented_dynamics_temp2=augmented_dynamics_temp
            def dtfunc(t,y):
                t = t.detach().requires_grad_(True)
                y = y.detach().requires_grad_(True)
                def tfunc(t):
                    return augmented_dynamics_temp2(t,y)
                return torch.autograd.functional.jacobian(tfunc,t)
            self.dtfunc=dtfunc

            def dyfunc(t,y,v):
                t = t.detach().requires_grad_(True)
                y = y.detach().requires_grad_(True)
                v = v.detach().requires_grad_(True)
                def yfunc(y):
                    return augmented_dynamics_temp2(t,y)
                output, jvp = torch.autograd.functional.jvp(yfunc,y,v)
                return jvp
            self.dyfunc=dyfunc
        else:
            self.original_func=None
            def dtfunc(t,y):
                t = t.detach().requires_grad_(True)
                y = y.detach().requires_grad_(True)
                def tfunc(t):
                    return self.func(t,y)
                return torch.autograd.functional.jacobian(tfunc,t)
            self.dtfunc=dtfunc

            def dyfunc(t,y,v):
                t = t.detach().requires_grad_(True)
                y = y.detach().requires_grad_(True)
                v = v.detach().requires_grad_(True)
                def yfunc(y):
                    return self.func(t,y)
                output, jvp = torch.autograd.functional.jvp(yfunc,y,v)
                return jvp
            self.dyfunc=dyfunc
        

    @abc.abstractmethod
    def _step_func_adjoint(self, func, t0, dt, t1, y0):
        pass    
    
    @abc.abstractmethod
    def eval_estimator(self, t0, dt, t1, y0, y1):
        pass

    @abc.abstractmethod
    def eval_estimator_adjoint(self,t0,dt,t1,y0,y1):
        pass

    @abc.abstractmethod
    def eval_estimator_eff(self,t0,dt,t1,y0,y1):
        pass

    @abc.abstractmethod
    def eval_estimator_eff_adjoint(self,t0,dt,t1,y0,y1):
        pass

    @staticmethod
    def refine_grid(self,grid,estis):
        if self.theta==1.:
            new_vs=(grid.clone().detach()[:-1]+grid.clone().detach()[1:])/2.
            new_grid,idxs=torch.sort(torch.cat((grid,new_vs)))
            return new_grid
        sorted_idx=torch.argsort(estis,descending=True)
        sorted_estis=estis[sorted_idx]
        cumsum=torch.cumsum(sorted_estis,0)
        cumsum_bool=(cumsum>=self.theta*cumsum[-1])*1
        idx=torch.argmax(cumsum_bool)
        marked=sorted_idx[range(idx+1)]
        new_vs=(grid.clone().detach()[marked]+grid.clone().detach()[marked+1])/2.
        new_grid,idxs=torch.sort(torch.cat((grid,new_vs)))
        return new_grid
    
    def integrate(self, t):
        
        if self.initial_grid is not None:
            time_grid=self.initial_grid
        else:
            time_grid=self.grid_constructor(self.func,self.y0,t)
        if self.conv_ana:
            len_grids=torch.tensor([time_grid.size(0)])
            estis_per=torch.tensor([torch.inf])
            grids=time_grid.detach().clone()
        estis = torch.ones(time_grid.size(),dtype=self.y0.dtype,device=self.y0.device)*torch.inf

        while(torch.sqrt(torch.sum(estis))>self.atol and time_grid.size(0)<self.max_nodes):
            solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
            solution[0] = self.y0
            estis = torch.zeros(time_grid.size(0)-1,dtype=self.y0.dtype)
            if self.conv_ana:
                if self.yhalf is not None:
                    sol_grid=torch.empty(2*len(time_grid)-1, *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
                    sol_grid[0]=self.y0
                else:
                    sol_grid=torch.empty(len(time_grid), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
                    sol_grid[0]=self.y0
            assert time_grid[0] == t[0] and time_grid[-1] == t[-1]

            j = 1
            i = 0
            y0 = self.y0
            for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
                dt = t1 - t0
                self.func.callback_step(t0, y0, dt)
                
                if self.efficient:
                    if self.faster_adj_solve:
                        y1, f0 = self._step_func_adjoint(self.func, t0, dt, t1, y0)
                        estis[i] = self.eval_estimator_eff_adjoint(t0 , dt , t1, y0, y1)
                    else:
                        y1, f0 = self._step_func(self.func, t0, dt, t1, y0)
                        estis[i] = self.eval_estimator_eff(t0 , dt , t1, y0, y1)
                else:
                    if self.faster_adj_solve:
                        y1, f0 = self._step_func_adjoint(self.func, t0, dt, t1, y0)
                        estis[i] = self.eval_estimator_adjoint(t0 , dt , t1, y0, y1)
                    else:
                        y1, f0 = self._step_func(self.func, t0, dt, t1, y0)
                        estis[i] = self.eval_estimator(t0 , dt , t1, y0, y1)
                if self.conv_ana:
                    if self.yhalf is not None:
                        sol_grid[2*i+1]=torch.reshape(torch.tensor(self.yhalf).to(y0.device, y0.dtype), y0.shape)
                        sol_grid[2*i+2]=y1.detach().clone()
                    else:    
                        sol_grid[i+1]=y1.detach().clone()
                i=i+1
                while j < len(t) and t1 >= t[j] and t0 < t[j]:
                    if self.interp == "linear":
                        solution[j] = self._linear_interp(t0, t1, y0, y1, t[j])
                    elif self.interp == "cubic":
                        f1 = self.func(t1, y1)
                        solution[j] = self._cubic_hermite_interp(t0, y0, f0, t1, y1, f1, t[j])
                    else:
                        raise ValueError(f"Unknown interpolation method {self.interp}")
                    j += 1
                y0 = y1
            if self.save_last_grid:
                if self.original_func is not None:
                    torch.save(time_grid,"current_grid_adjoint.pt")
                else:
                    torch.save(time_grid,"current_grid.pt")
            time_grid=self.refine_grid(self,time_grid,estis)
            if self.conv_ana:
                plt.step(time_grid[:-1],time_grid[1:]-time_grid[:-1],where="post")
                plt.yscale("log")
                plt.pause(0.05)
                plt.clf()
                print(sum(estis))
                len_grids=torch.cat((len_grids,torch.tensor([time_grid.size(0)])))
                grids=torch.cat((grids,time_grid.detach().clone()))
                if torch.max(estis_per)<torch.inf:
                    estis_per=torch.cat((estis_per,torch.tensor([torch.sum(estis)])))
                    sols.append(sol_grid.detach().clone().requires_grad_(False))
                else:
                    estis_per=torch.tensor([torch.sum(estis)])
                    sols=[sol_grid.detach().clone().requires_grad_(False)]
        if self.conv_ana:
            Path("results/").mkdir(parents=True, exist_ok=True)
            sols=torch.cat(sols,dim=0)
            torch.save(len_grids,"results/len_grids_"+self.file_id+".pt")
            torch.save(estis_per,"results/estis_"+self.file_id+".pt")
            torch.save(sols,"results/sols_"+self.file_id+".pt")
            torch.save(grids,"results/grids_"+self.file_id+".pt")
            return solution
        else:
            return solution
   
