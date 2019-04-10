from scipy import interpolate
import numpy as np


"""
TODOs:

implement periodic call

implement a least-squares constructor

"""
pinned = 2
clamped = 1
extrap = 0
periodic = -1
bc_map =  {clamped: "clamped", pinned: "natural", extrap: None, periodic: None}

class NDBSpline(object):
    def __init__(self, knots, coeffs, orders=3, periodicity=False):
        self.knots = knots
        self.coeffs = coeffs
        self.ndim = knots.shape[0]
        self.mdim = coeffs.shape[0]
        self.orders = np.broadcast_to(orders, (self.ndim,))
        self.periodicity = np.broadcast_to(periodicity, (self.ndim,))

        self.u_ops = []
        self.input_op = list(range(self.ndim+1)) + [...,]
        self.output_op = [0,...]
        self.tcks = []

        for i in np.arange(self.ndim)+1:
            self.u_ops.append([int(i), ...])
            num_bases = self.coeffs.shape[i]
            cs = np.eye(num_bases)
            order = self.orders[i-1]
            knot_sel = ((i-1,) + (0,)*(i-1) + (slice(None,None),) + 
                (0,)*(self.ndim-i))
            ts = self.knots[knot_sel]
            self.tcks.append((ts, cs, order))

    def broadcast_coords(self, coords):
        if coords.shape[0] == self.ndim:
            if coords.size == self.ndim:
                coords = coords.reshape((self.ndim, 1))
        elif coords.size%self.ndim == 0:
            coords = coords.reshape((self.ndim,-1))
        else:
            raise ValueError("Could not broadcast coords")
        return coords

    def indices_from_coords(self, coords):
        indices = np.ones_like(coords, dtype=np.int_)*-1000
        vec_sel = (slice(None),) + (coords.ndim-1)*(None,)
        for k in range(self.ndim):
            knot_sel = (k,) + (0,)*(k) + (slice(None,None),) + (0,)*(self.ndim-k-1)
            lt_test = self.knots[knot_sel][vec_sel] >= coords[k, ...]
            ge_test =  coords[k, ...] >= self.knots[knot_sel][vec_sel]
            total_test = np.r_['0,{}'.format(coords.ndim-1), 
                (self.knots[knot_sel][vec_sel] >= 
                    coords[k, ...])[0,...][None,...],
                    
                lt_test[1:,...] & ge_test[:-1,...],
                
                (self.knots[knot_sel][vec_sel] <=
                    coords[k, ...])[-1,...][None,...],
            ]
            
            where_out = np.nonzero(total_test)
            indices[(k,)+where_out[1:]] = where_out[0] - 1

    def _eval_basis(self, dim, x, nu=0):
        return interpolate.splev(x, self.tcks[dim-1], nu)

    def _eval_bases(self, x, nus=0):
        """
        Evaluate a spline function.

        Parameters
        ----------
        x : array_like
            points to evaluate the spline at.
        nu: int, optional
            derivative to evaluate (default is 0).

        Returns
        -------
        y : array_like
            Shape is determined by replacing the interpolation axis
            in the coefficient array with the shape of `x`.

        """
        nus = np.broadcast_to(nus, (self.ndim,))
        x = self.broadcast_coords(x)

        # TODO: IMPLEMENT PERIODIC BC
        # periodic_sel = np.all(self.bcs==-1, axis=1)
        # ns = np.r_[self.knots.shape[1:]] - self.order - 1
        # x[periodic_sel,...] = self.knots[periodic_sel, self.order[periodic_sel]]
        """
        # With periodic extrapolation we map x to the segment
        # [self.t[k], self.t[n]].        
        n = self.knots.shape[1:][periodic_sel] - self.order - 1
        x[] = self.t[self.k] + (x - self.t[self.k]) % (self.t[n] -
                                                     self.t[self.k])
        """
        u_mats = []        
        for i in np.arange(self.ndim)+1:
            nu = nus[i-1]
            u_mats.append(self._eval_basis(i, x[i-1,...], nu))
        return u_mats

    def __call__(self, x, nus=0):
        u_mats = self._eval_bases(x, nus)
        u_args = [subarg for arg in zip(u_mats, self.u_ops) for subarg in arg]
        y_out = np.einsum(self.coeffs, self.input_op, *u_args, self.output_op)
        return y_out

def make_interp_spline(x, y, bcs=0, orders=3):
    if isinstance(x, np.ndarray): # mesh
        if x.ndim == 1:
            ndim = 1
            x = x[None,...]
        else:
            ndim = x.ndim - 1

    elif not isinstance(x, str) and len(x): # vectors, or try
        ndim = len(x)
        x = np.meshgrid(x, indexing='ij')
        
    else:
        raise ValueError("Don't know how to interpret x")
    
    if y.ndim == ndim:
        # how can you tell if y needs a [None, ...] or [...] ?
        # same question as to whether ndim is y.ndim or y.ndim-1
        # I think this is the right answer. 
        y = y[None, ...]
    elif y.ndim == ndim+1:
        pass
    else:
        raise ValueError("incompatible dimension size")
    mdim = y.shape[0]
    # generally, x.shape = (ndim, n1, n2, ..., n_ndim)
    # and y.sahpe = (mdim, n1, n2, ..., n_ndim)
    bcs = np.broadcast_to(bcs, (ndim,2))
    orders = np.broadcast_to(orders, (ndim,))
    knot_shape = np.r_[x.shape]
    deriv_specs = np.asarray((bcs[:,:]>0),dtype=np.int)
    knot_shape[1:] = knot_shape[1:] + orders + 1 + deriv_specs.sum(axis=1)

    knots = np.zeros(knot_shape)
    coeffs = np.pad(y, np.r_[np.c_[0,0], deriv_specs], 'constant')

    for i in np.arange(ndim)+1:
        all_other_ax_shape = np.asarray(np.r_[coeffs.shape[1:i],
            y.shape[i+1:]], dtype=np.int)
        x_line_sel = ((i-1,) + (0,)*(i-1) + (slice(None,None),) +
            (0,)*(ndim-i))
        xp = x[x_line_sel]
        order = orders[i-1]
        for idx in np.ndindex(*all_other_ax_shape):
            offset_axes_remaining_sel = (tuple(idx[i-1:] + 
                deriv_specs[i:,0]))
            y_line_sel = ((Ellipsis,) + idx[:i-1] + 
                (slice(deriv_specs[i-1,0],-deriv_specs[i-1,0] or None),) +
                offset_axes_remaining_sel)
            coeff_line_sel = ((Ellipsis,) + idx[:i-1] + (slice(None,None),)
                + offset_axes_remaining_sel)
            line_spline = interpolate.make_interp_spline(xp,
                coeffs[y_line_sel].T,
                k = order,
                bc_type=(bc_map[(bcs[i-1,0])],
                         bc_map[(bcs[i-1,1])]),

            )
            coeffs[coeff_line_sel] = line_spline.c.T
        knots[i-1,...] = (line_spline.t[(None,)*(i-1) + 
            (slice(None),) + (None,)*(ndim-i)])
    return NDBSpline(knots, coeffs, orders, np.all(bcs==-1, axis=1))