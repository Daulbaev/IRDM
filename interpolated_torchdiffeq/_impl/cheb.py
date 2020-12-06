import numpy as np
import torch
from torch import nn

def r8vec_cheby1space ( n, a, b ):

#*****************************************************************************80
#
## R8VEC_CHEBY1SPACE creates a vector of Type 1 Chebyshev spaced values in [A,B].
#
#  Discussion:
#
#    An R8VEC is a vector of R8's.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    30 June 2015
#
#  Author:
#
#    John Burkardt
#
#  Parameters:
#
#    Input, integer N, the number of entries in the vector.
#
#    Input, real A, B, the first and last entries.
#
#    Output, real X(N), a vector of Type 1 Chebyshev spaced data.
#
    x = np.zeros(n)
    if n == 1:
        x[0] = (a+b)/2.0
    else:
        for i in range(n):
            theta = float(n-i-1)*np.pi/float(n-1)
            c = np.cos(theta )
            if (n%2) == 1:
                if 2*i+1 == n:
                    c=0.0
            x[i]=((1.0-c)*a + (1.0+c)*b)/2.0
    return x


def r8vec_cheby2space (n, a, b):

#*****************************************************************************80
#
## R8VEC_CHEBY2SPACE creates a vector of Type 2  Chebyshev values in [A,B].
#
#  Discussion:
#
#    An R8VEC is a vector of R8's.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    05 July 2017
#
#  Author:
#
#    John Burkardt
#
#  Parameters:
#
#    Input, integer N, the number of entries in the vector.
#
#    Input, real A, B, the first and last entries.
#
#    Output, real X(N), a vector of Type 2 Chebyshev spaced data.
#
    x = np.zeros(n)
    for i in range(n):
        theta = float(n-i)*np.pi/float(n+1)
        c = np.cos(theta)
        x[i] = ( ( 1.0 - c ) * a  \
               + ( 1.0 + c ) * b ) \
               /   2.0
    return x


def cheb1_interp_torch(w, nodes, values, t):
    '''
    We are given a tensor 'values' of shape (number of nodes) × (number of features) × (number of features)
    or nodes × (number of features)
    '''
    idx = torch.abs(nodes - t) < 1e-12
    if nodes[idx].shape[0] > 0:
        return values[idx][0]
    coef = w / (t - nodes)
    # TODO: the order of dimensions should be changed to speed up computations
    s = ''
    if len(values.shape) == 5:
        s = 'i,ijkmn->jkmn'
    elif len(values.shape) == 4:
        s = 'i,ijkm->jkm'
    elif len(values.shape) == 3:
        s = 'i,ijk->jk'
    elif len(values.shape) == 2:
        s = 'i,ij->j'
    num = torch.einsum(s, [coef, values])
    den = coef.sum()
    res = num / den
    return res

def compute_barycentric_weights(n_layers):
    w = np.zeros(n_layers)
    s = 1.0
    for j in range(n_layers):
        w[j] = s * np.sin((2*j+1) * np.pi/ float(2*n_layers))
        s =-s
    return w
