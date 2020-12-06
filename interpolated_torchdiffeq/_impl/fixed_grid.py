from .solvers import FixedGridODESolver
from . import rk_common


class Euler(FixedGridODESolver):

    def step_func(self, func, t, dt, y):
        self.dt_next.append(dt.cpu().detach().numpy().item())
        return tuple(dt * f_ for f_ in func(t, y))

    @property
    def order(self):
        return 1


class Midpoint(FixedGridODESolver):

    def step_func(self, func, t, dt, y):
        self.dt_next.append(dt.cpu().detach().numpy().item())
        # f(z(t)) = func(t, y) on the previous step. Additionally compute the last one
        y_mid = tuple(y_ + f_ * dt / 2 for y_, f_ in zip(y, func(t, y)))#(z(t+dt/2))
        # f_ is f(t), z(t)
        return tuple(dt * f_ for f_ in func(t + dt / 2, y_mid))

    @property
    def order(self):
        return 2


class RK4(FixedGridODESolver):

    def step_func(self, func, t, dt, y):
        self.dt_next.append(dt.cpu().detach().numpy().item())
        (result), inter_val, iter_time = rk_common.rk4_alt_step_func(func, t, dt, y)
        self.intermediate_values += inter_val
        self.intermediate_time += iter_time
        return result

    @property
    def order(self):
        return 4
