import torch
from torch import nn
from .cheb import cheb1_interp_torch, compute_barycentric_weights, r8vec_cheby1space
from .odeint import odeint
from .misc import _flatten, _flatten_convert_none_to_zeros

import time


class OdeintChebyshevMethod(torch.autograd.Function):
    @staticmethod
    def forward(ctx, n_nodes, *args):
        start_t = time.time()
        assert len(args) >= 9, 'Internal error: all arguments required.'
        y0, func, t, flat_params, rtol, atol, method, options, return_intermediate = \
            args[:-8], args[-8], args[-7], args[-6], args[-5], args[-4], args[-3], args[-2], args[-1]
        if t.shape[0] != 2:
            raise NotImplementedError('Time should be [0., T.]')
        ctx.func, ctx.rtol, ctx.atol, ctx.method, ctx.options = func, rtol, atol, method, options

        # if tuple is passed the first one is the solver for forward, the second one is for backward
        method = method[0] if isinstance(method, tuple) else method

        ctx.nodes = torch.tensor(r8vec_cheby1space(n_nodes, t[0].item(), t[-1].item()),
                                 device=y0[0].device, dtype=torch.float32)
        ctx.w = torch.tensor(compute_barycentric_weights(n_nodes),
                             device=y0[0].device, dtype=torch.float32)
        with torch.no_grad():
            ans = odeint(func, y0, ctx.nodes, rtol=rtol, atol=atol,
                         method=method, options=options, context=ctx,
                         return_intermediate_points=return_intermediate)
            if return_intermediate:
                ans, inter_values, inter_times, z_t, f_t = ans

            ctx.values = ans

        # Take results at t=0 and t=1 only
        ans = tuple(torch.cat((value[0:1], value[-1:]), 0) for value in ctx.values)

        ctx.save_for_backward(t, flat_params, *ans)

        if hasattr(func, 'base_func'):
            dst_struct = func.base_func
        else:
            dst_struct = func

        if hasattr(dst_struct, 'forward_t'):
            dst_struct.forward_t.append(time.time() - start_t)

        if return_intermediate:
            setattr(dst_struct, 'inter_values', inter_values)
            setattr(dst_struct, 'inter_times', inter_times)
            setattr(dst_struct, 'z_t', z_t)
            setattr(dst_struct, 'f_t', f_t)
        return ans

    @staticmethod
    def backward(ctx, *grad_output):
        start_t = time.time()
        t, flat_params, *ans = ctx.saved_tensors
        ans = tuple(ans)
        func, rtol, atol, method, options = ctx.func, ctx.rtol, ctx.atol, ctx.method, ctx.options

        # if tuple is passed the first one is the solver for forward, the second one is for backward
        method = method[-1] if isinstance(method, tuple) else method

        n_tensors = len(ans)
        f_params = tuple(func.parameters())

        # TODO: use a nn.Module and call odeint_adjoint to implement higher order derivatives.
        def interpolate(t):
            return tuple(cheb1_interp_torch(ctx.w, ctx.nodes, value, t) for value in ctx.values)

        def augmented_dynamics(t, y_aug):
            if hasattr(ctx.func, 'base_func'):
                if hasattr(ctx.func.base_func, 'nbe'):
                    ctx.func.base_func.nbe += 1
            elif hasattr(ctx.func, 'nbe'):
                ctx.func.nbe += 1

            # Dynamics of the original system augmented with
            # the adjoint wrt y, and an integrator wrt t and args.
            y = interpolate(t)
            adj_y = y_aug[:n_tensors]  # Ignore adj_time and adj_params.

            with torch.set_grad_enabled(True):
                t = t.to(y[0].device).detach().requires_grad_(True)
                y = tuple(y_.detach().requires_grad_(True) for y_ in y)
                func_eval = func(t, y)
                vjp_t, *vjp_y_and_params = torch.autograd.grad(
                    func_eval, (t,) + y + f_params,
                    tuple(-adj_y_ for adj_y_ in adj_y), allow_unused=True, retain_graph=True
                )
            vjp_y = vjp_y_and_params[:n_tensors]
            vjp_params = vjp_y_and_params[n_tensors:]

            # autograd.grad returns None if no gradient, set to zero.
            vjp_t = torch.zeros_like(t) if vjp_t is None else vjp_t
            vjp_y = tuple(torch.zeros_like(y_) if vjp_y_ is None else vjp_y_ for vjp_y_, y_ in zip(vjp_y, y))
            vjp_params = _flatten_convert_none_to_zeros(vjp_params, f_params)

            if len(f_params) == 0:
                vjp_params = torch.tensor(0.).to(vjp_y[0])
            return (*vjp_y, vjp_t, vjp_params)

        T = ans[0].shape[0]
        with torch.no_grad():
            adj_y = tuple(grad_output_[-1] for grad_output_ in grad_output)
            adj_params = torch.zeros_like(flat_params)
            adj_time = torch.tensor(0.).to(t)
            time_vjps = []
            for i in range(T - 1, 0, -1):

                ans_i = tuple(ans_[i] for ans_ in ans)
                grad_output_i = tuple(grad_output_[i] for grad_output_ in grad_output)
                func_i = func(t[i], ans_i)
                # Compute the effect of moving the current time measurement point.
                dLd_cur_t = sum(
                    torch.dot(func_i_.reshape(-1), grad_output_i_.reshape(-1)).reshape(1)
                    for func_i_, grad_output_i_ in zip(func_i, grad_output_i)
                )
                adj_time = adj_time - dLd_cur_t
                time_vjps.append(dLd_cur_t)

                # Run the augmented system backwards in time.
                if adj_params.numel() == 0:
                    adj_params = torch.tensor(0.).to(adj_y[0])
                aug_y0 = (*adj_y, adj_time, adj_params)
                compute_cheb = False
                if options is not None:
                    compute_cheb = options[
                        'use_backward_cheb_points'] if 'use_backward_cheb_points' in options else False
                aug_ans = odeint(
                    augmented_dynamics, aug_y0,
                    torch.tensor([t[i], t[i - 1]]), rtol=rtol, atol=atol, method=method, options=options,
                    context=ctx,
                    compute_cheb=compute_cheb,
                )
                # Unpack aug_ans.
                adj_y = aug_ans[:n_tensors]
                adj_time = aug_ans[n_tensors]
                adj_params = aug_ans[n_tensors + 1]

                adj_y = tuple(adj_y_[1] if len(adj_y_) > 0 else adj_y_ for adj_y_ in adj_y)
                if len(adj_time) > 0: adj_time = adj_time[1]
                if len(adj_params) > 0: adj_params = adj_params[1]

                adj_y = tuple(adj_y_ + grad_output_[i - 1] for adj_y_, grad_output_ in zip(adj_y, grad_output))

                del aug_y0, aug_ans

            time_vjps.append(adj_time)
            time_vjps = torch.cat(time_vjps[::-1])

            if hasattr(func, 'base_func'):
                if hasattr(func.base_func, 'backward_t'):
                    func.base_func.backward_t.append(time.time() - start_t)
            elif hasattr(func, 'backward_t'):
                func.backward_t.append(time.time() - start_t)
            return (None, *adj_y, None, time_vjps, adj_params, None, None, None, None, None)


def odeint_chebyshev_func(func,
                          y0,
                          t,
                          rtol=1e-6,
                          atol=1e-12,
                          method=None,
                          options=None,
                          n_nodes=None,
                          return_intermediate_points=False):
    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if not isinstance(func, nn.Module):
        raise ValueError('func is required to be an instance of nn.Module.')
    if n_nodes is None:
        raise ValueError('n_nodes should be specified.')
    tensor_input = False
    if torch.is_tensor(y0):
        class TupleFunc(nn.Module):
            def __init__(self, base_func):
                super(TupleFunc, self).__init__()
                self.base_func = base_func

            def forward(self, t, y):
                return (self.base_func(t, y[0]),)

        tensor_input = True
        y0 = (y0,)
        func = TupleFunc(func)

    flat_params = _flatten(func.parameters())
    ys = OdeintChebyshevMethod.apply(n_nodes,
                                     *y0,
                                     func,
                                     t,
                                     flat_params,
                                     rtol,
                                     atol,
                                     method,
                                     options,
                                     return_intermediate_points)

    if tensor_input:
        ys = ys[0]
    return ys


class OdeintLinearMethod(torch.autograd.Function):
    @staticmethod
    def forward(ctx, n_nodes, *args):
        start_t = time.time()
        assert len(args) >= 9, 'Internal error: all arguments required.'
        y0, func, t, flat_params, rtol, atol, method, options, return_intermediate = \
            args[:-8], args[-8], args[-7], args[-6], args[-5], args[-4], args[-3], args[-2], args[-1]
        if t.shape[0] != 2:
            raise NotImplementedError('Time should be [0., T.]')
        ctx.func, ctx.rtol, ctx.atol, ctx.method, ctx.options = func, rtol, atol, method, options

        # if tuple is passed the first one is the solver for forward, the second one is for backward
        method = method[0] if isinstance(method, tuple) else method
        ctx.n_nodes = n_nodes
        ctx.t0 = t[0].item()
        ctx.t1 = t[-1].item()
        ctx.h = (ctx.t1 - ctx.t0) / ctx.n_nodes
        if t.shape[0] != 2:
            raise NotImplementedError('Time should be [0., T.]')
        ctx.nodes = torch.linspace(ctx.t0, ctx.t1, ctx.n_nodes + 1,
                                   device=y0[0].device, dtype=torch.float32)
        with torch.no_grad():
            ans = odeint(func, y0, ctx.nodes, rtol=rtol, atol=atol,
                         method=method, options=options, context=ctx,
                         return_intermediate_points=return_intermediate)
            if return_intermediate:
                ans, inter_values, inter_times, z_t, f_t = ans

            ctx.values = ans
        # Take results at t=0 and t=1 only
        ans = tuple(torch.cat((value[0:1], value[-1:]), 0) for value in ctx.values)
        ctx.save_for_backward(t, flat_params, *ans)
        if hasattr(func, 'base_func'):
            dst_struct = func.base_func
        else:
            dst_struct = func

        if hasattr(dst_struct, 'forward_t'):
            dst_struct.forward_t.append(time.time() - start_t)

        if return_intermediate:
            setattr(dst_struct, 'inter_values', inter_values)
            setattr(dst_struct, 'inter_times', inter_times)
            setattr(dst_struct, 'z_t', z_t)
            setattr(dst_struct, 'f_t', f_t)
        return ans

    @staticmethod
    def backward(ctx, *grad_output):
        start_t = time.time()
        t, flat_params, *ans = ctx.saved_tensors
        ans = tuple(ans)
        func, rtol, atol, method, options = ctx.func, ctx.rtol, ctx.atol, ctx.method, ctx.options

        # if tuple is passed the first one is the solver for forward, the second one is for backward
        method = method[-1] if isinstance(method, tuple) else method

        n_tensors = len(ans)
        f_params = tuple(func.parameters())

        def interpolate(t):
            idx = int((t - ctx.t0) // ctx.h)
            if idx <= 0:
                return tuple(value[0] for value in ctx.values)
            if idx >= ctx.n_nodes:
                return tuple(value[-1] for value in ctx.values)
            return tuple(value[idx] + (ctx.nodes[idx + 1] - t) * (
                    value[idx + 1] - value[idx]) / ctx.h for value in ctx.values)

        def augmented_dynamics(t, y_aug):
            if hasattr(ctx.func, 'base_func'):
                if hasattr(ctx.func.base_func, 'nbe'):
                    ctx.func.base_func.nbe += 1
            elif hasattr(ctx.func, 'nbe'):
                ctx.func.nbe += 1

            # Dynamics of the original system augmented with
            # the adjoint wrt y, and an integrator wrt t and args.
            y = interpolate(t)
            adj_y = y_aug[:n_tensors]  # Ignore adj_time and adj_params.

            with torch.set_grad_enabled(True):
                t = t.to(y[0].device).detach().requires_grad_(True)
                y = tuple(y_.detach().requires_grad_(True) for y_ in y)
                func_eval = func(t, y)
                vjp_t, *vjp_y_and_params = torch.autograd.grad(
                    func_eval, (t,) + y + f_params,
                    tuple(-adj_y_ for adj_y_ in adj_y), allow_unused=True, retain_graph=True
                )
            vjp_y = vjp_y_and_params[:n_tensors]
            vjp_params = vjp_y_and_params[n_tensors:]

            # autograd.grad returns None if no gradient, set to zero.
            vjp_t = torch.zeros_like(t) if vjp_t is None else vjp_t
            vjp_y = tuple(torch.zeros_like(y_) if vjp_y_ is None else vjp_y_ for vjp_y_, y_ in zip(vjp_y, y))
            vjp_params = _flatten_convert_none_to_zeros(vjp_params, f_params)

            if len(f_params) == 0:
                vjp_params = torch.tensor(0.).to(vjp_y[0])
            return (*vjp_y, vjp_t, vjp_params)

        T = ans[0].shape[0]
        with torch.no_grad():
            adj_y = tuple(grad_output_[-1] for grad_output_ in grad_output)
            adj_params = torch.zeros_like(flat_params)
            adj_time = torch.tensor(0.).to(t)
            time_vjps = []
            for i in range(T - 1, 0, -1):

                ans_i = tuple(ans_[i] for ans_ in ans)
                grad_output_i = tuple(grad_output_[i] for grad_output_ in grad_output)
                func_i = func(t[i], ans_i)
                # Compute the effect of moving the current time measurement point.
                dLd_cur_t = sum(
                    torch.dot(func_i_.reshape(-1), grad_output_i_.reshape(-1)).reshape(1)
                    for func_i_, grad_output_i_ in zip(func_i, grad_output_i)
                )
                adj_time = adj_time - dLd_cur_t
                time_vjps.append(dLd_cur_t)

                # Run the augmented system backwards in time.
                if adj_params.numel() == 0:
                    adj_params = torch.tensor(0.).to(adj_y[0])
                aug_y0 = (*adj_y, adj_time, adj_params)
                aug_ans = odeint(
                    augmented_dynamics, aug_y0,
                    torch.tensor([t[i], t[i - 1]]), rtol=rtol, atol=atol, method=method, options=options,
                    context=ctx
                )
                # Unpack aug_ans.
                adj_y = aug_ans[:n_tensors]
                adj_time = aug_ans[n_tensors]
                adj_params = aug_ans[n_tensors + 1]

                adj_y = tuple(adj_y_[1] if len(adj_y_) > 0 else adj_y_ for adj_y_ in adj_y)
                if len(adj_time) > 0: adj_time = adj_time[1]
                if len(adj_params) > 0: adj_params = adj_params[1]

                adj_y = tuple(adj_y_ + grad_output_[i - 1] for adj_y_, grad_output_ in zip(adj_y, grad_output))

                del aug_y0, aug_ans

            time_vjps.append(adj_time)
            time_vjps = torch.cat(time_vjps[::-1])

            if hasattr(func, 'base_func'):
                if hasattr(func.base_func, 'backward_t'):
                    func.base_func.backward_t.append(time.time() - start_t)
            elif hasattr(func, 'backward_t'):
                func.backward_t.append(time.time() - start_t)
            return (None, *adj_y, None, time_vjps, adj_params, None, None, None, None, None)


def odeint_linear_func(func,
                       y0,
                       t,
                       rtol=1e-6,
                       atol=1e-12,
                       method=None,
                       options=None,
                       n_nodes=None,
                       return_intermediate_points=False):
    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if not isinstance(func, nn.Module):
        raise ValueError('func is required to be an instance of nn.Module.')
    if n_nodes is None:
        raise ValueError('n_nodes should be specified.')

    tensor_input = False
    if torch.is_tensor(y0):
        class TupleFunc(nn.Module):

            def __init__(self, base_func):
                super(TupleFunc, self).__init__()
                self.base_func = base_func

            def forward(self, t, y):
                return (self.base_func(t, y[0]),)

        tensor_input = True
        y0 = (y0,)
        func = TupleFunc(func)

    flat_params = _flatten(func.parameters())
    ys = OdeintLinearMethod.apply(n_nodes,
                                  *y0,
                                  func,
                                  t,
                                  flat_params,
                                  rtol,
                                  atol,
                                  method,
                                  options,
                                  return_intermediate_points)

    if tensor_input:
        ys = ys[0]
    return ys
