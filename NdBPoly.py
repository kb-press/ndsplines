from scipy import ndimage
import numpy as np

class NdBPoly():
    def __init__(self, values, knots, order=3, mode='even', cval=0.0):
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
        elif values == self.ndim+1:
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
                (self.knots_vectors[k][vec_sel] < coords[k, ...])[0,...][None,...],
                lt_test[1:,...] & ge_test[:-1,...],
                (self.knots_vectors[k][vec_sel] > coords[k, ...])[0,...][None,...],
            ]
            
            where_out = np.nonzero(total_test)
            indices[(k,)+where_out[1:]] = where_out[0] - 1
        return indices

    def transform_coord_to_pixel(self, coords, in_place=False):
        coords = self.broadcast_coords(coords)
        if not in_place:
            coords = coords.copy()
        indices = self.indices_from_coords(coords)
        x0 = np.ones_like(coords)*np.NaN
        dx = np.ones_like(coords)
        aliasing = np.zeros_like(coords, dtype=np.int_)
        for k in range(self.ndim):
            # internal coordinates
            internal_intervals = (k, 
                (indices[k,...]>-1) &
                (indices[k,...]<self.knots_vectors[k].size)
            )
            x0[internal_intervals] = self.knots_vectors[k][indices[internal_intervals]]
            
            internal_intervals = (k, 
                (indices[k,...]>-1) &
                (indices[k,...]<self.knots_vectors[k].size-1)
            )
            dx[internal_intervals] = self.diff_knots_vectors[k][indices[internal_intervals]-1]

            aliasing[k, indices[k,...]==-1] += 1
            aliasing[k, indices[k,...]==self.knots_vectors[k].size-1] += -1
        
        pixel_coords = indices + (coords - x0)/dx
        
        if np.any(aliasing != 0):
            alias_sel = np.any(aliasing != 0, axis=0)
            recurse_coords = coords[:, alias_sel].copy()
            for k in range(self.ndim):
                negative_sel = (k, aliasing[k, alias_sel] < 0)
                recurse_coords[negative_sel] = self.knots_vectors[k][0] - (recurse_coords[negative_sel] - self.knots_vectors[k][0])
                positive_sel = (k, aliasing[k, alias_sel] > 0)
                recurse_coords[positive_sel] = self.knots_vectors[k][-1] - (recurse_coords[positive_sel] - self.knots_vectors[k][-1])
            recursed_coords, recursed_alias = self.transform_coord_to_pixel(
                recurse_coords, True)
            aliasing[:, alias_sel] = aliasing[:, alias_sel] - recursed_alias
            pixel_coords[:, alias_sel] = recursed_coords
            
        return pixel_coords, aliasing
        return indices, coords, x0, dx
        # can I recurse on indices, x0, dx, and aliasing instead of pixel_coords?
        
    
    def evaluate(self, coords, derivative=0):
        output = np.ones((len(self.coords),) + self.coords.shape[1:])
        pixel_coords, aliasing_mask = self.transform_coord_to_pixel
        for k in range(self.ndim):
            output[k,...] = ndimage.map_coordinates(self.coeffs[k],
                pixel_coords, prefilter=False, mode='nearest')
        return output