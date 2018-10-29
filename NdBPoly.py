from scipy import ndimage
import numpy as np
from numpy.lib.arraypad import _validate_lengths

class NdBPoly():
    def __init__(self, values, knots, odd_extrapolation=True, order=3, cval=0.0):
        self.order = order
        if isinstance(knots, np.ndarray): # mesh
            self.knots_mesh = knots
            if knots.ndim == 1:
                self.ndim = 1
            else:
                self.ndim = knots.ndim - 1
            self.knots_vectors = tuple(np.unique(self.knots_mesh[k,...])
                for k in range(self.ndim))
        elif not isinstance(knots, str) and len(knots): # vectors, or try
            self.knots_vectors = knots
            self.knots_mesh = self.meshgrid(knots, indexing='ij')
            self.ndim = len(knots)
        else: # use pixels
            raise ValueError("Don't know how to interpret dimension this")
        
        
        # self.odd_extrapolation = _validate_lengths(self.knots_mesh, odd_extrapolation)
        odd_extrapolation = np.array(odd_extrapolation, dtype=np.bool_)
        if odd_extrapolation.size == 1:
            odd_extrapolation = np.tile(odd_extrapolation, self.ndim)
        if odd_extrapolation.size != self.ndim:
            raise ValueError("Cannot broadcast odd_extrapolation to knot dimension")
        self.odd_extrapolation = odd_extrapolation
            

        self.pixels = np.ones_like(self.knots_mesh)
        for k in range(self.ndim):
            self.pixels[k,...] = np.cumsum(self.pixels[k,...], axis=k)-1

        self.diff_knots_vectors = tuple(np.diff(vec)
            for vec in self.knots_vectors)
        self.diff_knots_mesh = np.meshgrid(self.diff_knots_vectors, indexing='ij')
        
        if values.ndim == self.ndim:
            # how can you tell if values needs a [None, ...] or [...] ?
            # same question as to whether ndim is values.ndim or values.ndim-1
            self.values = values[None, ...]
        elif values.ndim == self.ndim+1:
            self.values = values
        else:
            raise ValueError("incompatible dimension size")
        
        self.coeffs = tuple(ndimage.spline_filter(self.values[k,...])
            for k in range(self.values.shape[0]))
    
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
            lt_test = self.knots_vectors[k][vec_sel] >= coords[k, ...]
            ge_test =  coords[k, ...] >= self.knots_vectors[k][vec_sel]
            total_test = np.r_['0,{}'.format(coords.ndim-1), 
                (self.knots_vectors[k][vec_sel] > 
                    coords[k, ...])[0,...][None,...],
                    
                lt_test[1:,...] & ge_test[:-1,...],
                
                (self.knots_vectors[k][vec_sel] <
                    coords[k, ...])[-1,...][None,...],
            ]
            
            where_out = np.nonzero(total_test)
            indices[(k,)+where_out[1:]] = where_out[0] - 1
        return indices

    def coord_edges(self, coords, in_place=False):
        coords = self.broadcast_coords(coords)
        if not in_place:
            coords = coords.copy()
        indices = self.indices_from_coords(coords)
        x0 = np.ones_like(coords)*np.NaN
        dx = np.ones_like(coords)
        aliasing = np.zeros_like(coords, dtype=np.int_)
        for k in range(self.ndim):
            internal_intervals = (k, 
                (indices[k,...]>-1) &
                (indices[k,...]<self.knots_vectors[k].size)
            )
            x0[internal_intervals] = \
                self.knots_vectors[k][indices[internal_intervals]]
            
            internal_intervals = (k, 
                (indices[k,...]>-1) &
                (indices[k,...]<self.knots_vectors[k].size-1)
            )
            dx[internal_intervals] = \
                self.diff_knots_vectors[k][indices[internal_intervals]-1]
            
            # I'm not sure why these are the opposite edge as what I think it 
            # should be, but it works!
            aliasing[k, indices[k,...]==-1] += 1
            aliasing[k, indices[k,...]==self.knots_vectors[k].size-1] += -1
        
        if np.any(aliasing != 0):
            alias_sel = np.any(aliasing != 0, axis=0)
            recurse_coords = coords[:, alias_sel].copy()
            for k in range(self.ndim):
                negative_sel = (k, aliasing[k, alias_sel] < 0)
                recurse_coords[negative_sel] = self.knots_vectors[k][0] - \
                    (recurse_coords[negative_sel] - self.knots_vectors[k][0])
                positive_sel = (k, aliasing[k, alias_sel] > 0)
                recurse_coords[positive_sel] = self.knots_vectors[k][-1] - \
                    (recurse_coords[positive_sel] - self.knots_vectors[k][-1])
            (recursed_indices, recursed_coords, recursed_x0,
                recursed_dx, recursed_alias) = self.coord_edges(
                                                    recurse_coords, True)
                                                    
            aliasing[:, alias_sel] = aliasing[:, alias_sel] - recursed_alias
            x0[:, alias_sel] = recursed_x0
            dx[:, alias_sel] = recursed_dx
            indices[:, alias_sel] = recursed_indices
            coords[:, alias_sel] = recursed_coords

        return indices, coords, x0, dx, aliasing
        
    def transform_coord_to_pixel(self, coords):
        indices, coords, x0, dx, aliasing = self.coord_edges(coords)
        pixel_coords = indices + (coords - x0)/dx
        return pixel_coords, aliasing, dx
        
    def evaluate(self, coords, derivative=0):
        output = np.ones((len(self.coeffs),) + coords.shape[1:])
        pixel_coords, aliasing_mask, dx = self.transform_coord_to_pixel(coords)
        remaining_axis_sel = tuple(np.arange(self.ndim)+1)
        for j in range(len(self.coeffs)):
            output[j,...] = ndimage.map_coordinates(self.coeffs[j],
                pixel_coords, prefilter=False, mode='nearest')

            # I'm not sure how to get odd extrapolation to work for anything besides 1 dimension, luckily that's all we need
            for k in np.nonzero((aliasing_mask!=0).any(axis=remaining_axis_sel))[0]:
                if self.odd_extrapolation[k]: # odd
                    left_x = pixel_coords.copy()
                    left_x[k] = 0
                    right_x = pixel_coords.copy()
                    right_x[k, ...] = self.knots_vectors[k].size
                    left_vals = ndimage.map_coordinates(self.coeffs[j],
                        left_x, prefilter=False, mode='nearest')
                    right_vals = ndimage.map_coordinates(self.coeffs[j],
                        right_x, prefilter=False, mode='nearest')
                    diff_vals = right_vals - left_vals

                    neg_sel = aliasing_mask[k]<0
                    output[j,neg_sel] = (
                        left_vals[neg_sel] 
                        + np.where((aliasing_mask[k]%2==1)[neg_sel], left_vals[neg_sel] - output[j,neg_sel], output[j,neg_sel]-right_vals[neg_sel])
                        + np.where((aliasing_mask[k]<=-2),aliasing_mask[k]+1,0)[neg_sel]*diff_vals[neg_sel]
                    )

                    pos_sel = aliasing_mask[k]>0
                    output[j,pos_sel] = (
                        right_vals[pos_sel] 
                        + np.where((aliasing_mask[k]%2==1)[pos_sel], right_vals[pos_sel] - output[j,pos_sel], output[j,pos_sel]- left_vals[pos_sel])
                        + np.where((aliasing_mask[k]>=2),aliasing_mask[k]-1,0)[pos_sel]*diff_vals[pos_sel]
                    )

                    # 2*right_vals[pos_sel] - output[j,pos_sel] + \
                    #     np.where((aliasing_mask[k]>=2),aliasing_mask[k]-1,0)[pos_sel]*diff_vals[pos_sel]
                
        return output