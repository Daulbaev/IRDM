import torch
from torch import nn
from cheb import r8vec_cheby2space
import scipy.sparse.linalg as la
import scipy.sparse as sp
import numpy as np

def prepare_data(xdata, ydata):
    xdata = np.asarray(xdata, dtype=np.float64)
    ydata = np.asarray(ydata, dtype=np.float64)

    data_shape = list(ydata.shape)

    if xdata.ndim > 1:
        raise ValueError('xdata must be a vector')
    if xdata.size < 2:
        raise ValueError('xdata must contain at least 2 data points.')

    if ydata.ndim > 1:
        if data_shape[-1] != xdata.size:
            raise ValueError(
                'ydata data must be a vector or '
                'ND-array with shape[-1] equal of xdata.size')
        if ydata.ndim > 2:
            ydata = ydata.reshape((np.prod(data_shape[:-1]), data_shape[-1]))
    else:
        if ydata.size != xdata.size:
            raise ValueError('ydata vector size must be equal of xdata size')

        ydata = np.array(ydata, ndmin=2)
        
    return xdata, ydata


def prepare_data_torch(xdata, ydata, dtype=torch.double):
#     xdata = np.asarray(xdata, dtype=np.float64)
    xdata = xdata.double()
    
#     ydata = np.asarray(ydata, dtype=np.float64)
    ydata = ydata.double()

    data_shape = torch.tensor(ydata.shape)

    if xdata.dim() > 1:
        raise ValueError('xdata must be a vector')
    if xdata.nelement() < 2:
        raise ValueError('xdata must contain at least 2 data points.')

    if ydata.dim() > 1:
        if data_shape[-1] != xdata.nelement():
            raise ValueError(
                'ydata data must be a vector or '
                'ND-array with shape[-1] equal of xdata.size')
        if ydata.dim() > 2:
#             ydata = ydata.reshape((np.prod(data_shape[:-1]), data_shape[-1]))
            ydata = ydata.reshape((data_shape[:-1].prod(), data_shape[-1]))
    else:
        if ydata.nelement() != xdata.nelement():
            raise ValueError('ydata vector size must be equal of xdata size')

#         ydata = np.array(ydata, ndmin=2)
        ydata = ydata.view(1, -1)
        
    return xdata, ydata


def compute_smooth(a, b):
    """
    The calculation of the smoothing spline requires the solution of a
    linear system whose coefficient matrix has the form p*A + (1-p)*B, with
    the matrices A and B depending on the data sites x. The default value
    of p makes p*trace(A) equal (1 - p)*trace(B).
    """
    def trace(m: sp.dia_matrix):
        return m.diagonal().sum()
    return 1. / (1. + trace(a) / (6. * trace(b)))

def compute_smooth_torch(a, b):
    """
    The calculation of the smoothing spline requires the solution of a
    linear system whose coefficient matrix has the form p*A + (1-p)*B, with
    the matrices A and B depending on the data sites x. The default value
    of p makes p*trace(A) equal (1 - p)*trace(B).
    """
    def trace(m):
        return m.diagonal().sum()
    return 1. / (1. + trace(a) / (6. * trace(b)))
        
def make_spline(nodes, nodes_vals, smooth=None):
    
    """smooth : float
        [Optional] Smoothing parameter in range [0, 1] where:
            - 0: The smoothing spline is the least-squares straight line fit
            - 1: The cubic spline interpolant with natural condition"""
    pcount = nodes.size
    dx = np.diff(nodes)
    
    axis = nodes_vals.ndim - 1
    
    weigths = np.ones_like(nodes)

    if not all(dx > 0):
        raise ValueError(
            'Items of xdata vector must satisfy the condition: x1 < x2 < ... < xN')

    dy = np.diff(nodes_vals, axis=axis)
    
    divdydx = dy / dx

    if pcount > 2:
        # Create diagonal sparse matrices
        diags_r = np.vstack((dx[1:], 2 * (dx[1:] + dx[:-1]), dx[:-1]))
        r = sp.spdiags(diags_r, [-1, 0, 1], pcount - 2, pcount - 2)

        odx = 1. / dx
        diags_qt = np.vstack((odx[:-1], -(odx[1:] + odx[:-1]), odx[1:]))
        qt = sp.diags(diags_qt, [0, 1, 2], (pcount - 2, pcount))

        ow = 1. / weigths
        osqw = 1. / np.sqrt(weigths)
        w = sp.diags(ow, 0, (pcount, pcount))
        qtw = qt @ sp.diags(osqw, 0, (pcount, pcount))

        # Solve linear system for the 2nd derivatives
        qtwq = qtw @ qtw.T

        if smooth is None:
            p = compute_smooth(r, qtwq)
        else:
            p = smooth
        a = (6. * (1. - p)) * qtwq + p * r
        b = np.diff(divdydx, axis=axis).T
        u = np.array(la.spsolve(a, b), ndmin=2)
        ydim = nodes_vals.shape[0]
        
        if ydim == 1:
            u = u.T

        dx = np.array(dx, ndmin=2).T
        d_pad = np.zeros((1, ydim))
        d1 = np.diff(np.vstack((d_pad, u, d_pad)), axis=0) / dx
        d2 = np.diff(np.vstack((d_pad, d1, d_pad)), axis=0)
        yi = np.array(nodes_vals, ndmin=2).T
        yi = yi - ((6. * (1. - p)) * w) @ d2
        c3 = np.vstack((d_pad, p * u, d_pad))
        c2 = np.diff(yi, axis=0) / dx - dx * (2. * c3[:-1, :] + c3[1:, :])
        coeffs = np.hstack((
            (np.diff(c3, axis=0) / dx).T,
            3. * c3[:-1, :].T,
            c2.T,
            yi[:-1, :].T
        ))

        c_shape = ((pcount - 1) * ydim, 4)
        coeffs = coeffs.reshape(c_shape, order='F')
    else:
        p = 1.
        coeffs = np.array(np.hstack((divdydx, np.array(nodes_vals[:, 0], ndmin=2).T)), ndmin=2)
    return coeffs


def make_spline_torch(nodes, nodes_vals, smooth=None):
    
    """smooth : float
        [Optional] Smoothing parameter in range [0, 1] where:
            - 0: The smoothing spline is the least-squares straight line fit
            - 1: The cubic spline interpolant with natural condition"""
    
#     print(nodes_vals.shape)
#     pcount = nodes.size
    pcount = nodes.nelement()
#     dx = np.diff(self._xdata)
    dx = nodes[1:] - nodes[:-1]
    
    axis = nodes_vals.dim() - 1
    
    weigths = torch.ones(nodes.shape, device=nodes.device)

    if not all(dx > 0):
        raise ValueError(
            'Items of xdata vector must satisfy the condition: x1 < x2 < ... < xN')
    
#     dy = np.diff(nodes_vals, axis=axis)
    if axis == 0:
        dy = nodes_vals[0, 1:] - nodes_vals[0, :-1]
    elif axis == 1:
        dy = nodes_vals[:, 1:] - nodes_vals[:, :-1]
    
#     print("dy", dy)
#     print("nodes_vals", nodes_vals)
    
    divdydx = dy / dx
    
#     print(divdydx)

    if pcount > 2:
#         # Create diagonal sparse matrices
#         diags_r = np.vstack((dx[1:].cpu(), 2 * (dx[1:] + dx[:-1]).cpu(), dx[:-1].cpu()))
#         r = sp.spdiags(diags_r, [-1, 0, 1], pcount - 2, pcount - 2)
#         print(r.toarray())
        r = torch.diagflat(dx[1:-1], -1) + \
            torch.diagflat(2 * (dx[1:] + dx[:-1]), 0) + \
            torch.diagflat(dx[1:-1], 1)
#         print(r)
        

        odx = 1. / dx
        diags_qt = np.vstack((odx[:-1].cpu(), -(odx[1:] + odx[:-1]).cpu(), odx[1:].cpu()))
        qt = sp.diags(diags_qt, [0, 1, 2], (pcount - 2, pcount))
        qt_t = torch.from_numpy(qt.toarray()).to(nodes.device)
#         print(qt_t)

#         qt = torch.diagflat()

        ow = 1. / weigths
        osqw = 1. / torch.sqrt(weigths)
        
#         w = sp.diags(ow, 0, (pcount, pcount))
        w = torch.diagflat(ow)
#         qtw = qt @ sp.diags(osqw, 0, (pcount, pcount))
        qtw = qt_t @ torch.diagflat(osqw)

#         # Solve linear system for the 2nd derivatives
        qtwq = qtw @ qtw.t()

        if smooth is None:
            p = compute_smooth_torch(r, qtwq)
        else:
            p = smooth
        a = (6. * (1. - p)) * qtwq + p * r

#         b = np.diff(divdydx, axis=axis).T
        if axis == 0:
            b = divdydx[0, 1:] - divdydx[0, :-1]
        elif axis == 1:
            b = divdydx[:, 1:] - divdydx[:, :-1]
        b = b.t()
#         u = np.array(la.spsolve(a, b), ndmin=2)
#         print(a.shape, b.shape)
        u, _ = torch.solve(b, a)
        
        ydim = nodes_vals.shape[0]
#         print(ydim)
#         if ydim == 1:
#             u = u.t()

#         dx = np.array(dx, ndmin=2).T

#         print("u shape", u.shape)
#         print(u)
        dx = dx.view(-1, 1)
#         print(dx.shape)
        d_pad = torch.zeros((1, ydim), device=nodes.device, dtype=nodes.dtype)

#         d1 = np.diff(np.vstack((d_pad, u, d_pad)), axis=0) / dx
#         print("dpad, u shapes", d_pad.shape, u.shape)
        d1 = torch.cat((d_pad, u, d_pad), dim=0)
#         print((d1[1:, :] - d1[:-1, :]).shape)
        d1 = (d1[1:, :] - d1[:-1, :]) / dx
        
#         d2 = np.diff(np.vstack((d_pad, d1, d_pad)), axis=0)
        d2 = torch.cat((d_pad, d1, d_pad), dim=0)
        d2 = d2[1:, :] - d2[:-1, :]
        
#         print("d1 shape", d1.shape)
#         print("d2 shape", d2.shape)

#         yi = np.array(nodes_vals, ndmin=2).T
        yi = nodes_vals.t()
    
        yi = yi - ((6. * (1. - p)) * w) @ d2
        
#         print(yi)
#         c3 = np.vstack((d_pad, p * u, d_pad))
        c3 = torch.cat((d_pad, p * u, d_pad), dim=0)
#         print("c3", c3)
#         print(c3.shape)
    
#         c2 = np.diff(yi, axis=0) / dx - dx * (2. * c3[:-1, :] + c3[1:, :])
        c2 = (yi[1:, :] - yi[:-1, :]) / dx - dx * (2. * c3[:-1, :] + c3[1:, :])
#         print(c2.shape)
#         coeffs = np.hstack((
#             (np.diff(c3, axis=0) / dx).T,
#             3. * c3[:-1, :].T,
#             c2.T,
#             yi[:-1, :].T
#         ))
        
        c3_diff = ((c3[1:, :] - c3[:-1, :]) / dx).t()
#         print(c3_diff)
#         print("dx", dx)
        coeffs = torch.cat((c3_diff, 3. * c3[:-1, :].t(), c2.t(), yi[:-1, :].t()), dim=1)
        
#         print("coeffs", coeffs.data.cpu().numpy(), coeffs.shape)
#         print(coeffs.shape)
        c_shape = ((pcount - 1) * ydim, 4)
        coeff_reshaped = torch.zeros(c_shape, device=nodes.device, dtype=nodes.dtype)
        factor = int(coeffs.shape[1] / 4)
        nrow = coeffs.shape[0]
        k = 0
        for i in range(4):
            for row_counter, j in enumerate(range(k * factor, (k+1) * factor)):
                coeff_reshaped[row_counter*nrow:(row_counter+1)*nrow, i] = coeffs[:, j]
            k += 1
#         print(coeff_reshaped)
# #         print(c_shape)
#         coeffs = coeffs.reshape(c_shape, order='F')
    else:
        p = 1.
# #         coeffs = np.array(np.hstack((divdydx, np.array(self._ydata[:, 0], ndmin=2).T)), ndmin=2)
#         coeffs = np.array(np.hstack((divdydx, np.array(nodes_vals[:, 0], ndmin=2).T)), ndmin=2)
    return coeff_reshaped


def univariate_evaluate_torch(t, nodes, nodes_vals, coeff, shape):
    '''
    Nodes should be sorted! 
    '''
    t = torch.tensor([t], device=coeff.device)
    index = (nodes > t.item()).nonzero()
    if len(index) == 0:
        index = len(nodes) - 1
    elif index[0] == 0:
        index = 1 
    else:
        index = index[0].item()
    t = (t - nodes[index - 1]).item()
    d = nodes_vals.shape[0]
    if d > 1: 
        idx_a, idx_b = (index - 1) * d, index * d
    else:
        raise ValueError('There should be at least 2 nodes!')
    values = coeff[idx_a:idx_b, 0:1].t()
    for i in range(1, coeff.shape[1]):
        values = t * values + coeff[idx_a:idx_b, i:i+1].t()
    values = values.reshape(shape)
    return values


def compute_linear_transformation(nodes, smooth=None):
    '''
    coeff = D @ nodes_vals
    '''
    n_nodes = nodes.shape[0]
    D = torch.zeros(n_nodes, n_nodes - 1, 4, device=nodes.device, dtype=nodes.dtype)
    for idx in range(n_nodes):
        nodes_vals = torch.zeros(1, n_nodes, device=nodes.device, dtype=nodes.dtype)
        nodes_vals[0, idx] = 1
        D[idx, :] = make_spline_torch(nodes, nodes_vals, smooth=smooth)
    return D


def make_spline_using_transformation(D, nodes_vals):
    coeff = torch.einsum('ijk,mi->jmk', [D, nodes_vals])
    coeff = coeff.reshape(-1, coeff.shape[-1])
    return coeff
