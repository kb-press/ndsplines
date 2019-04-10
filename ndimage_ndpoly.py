from scipy import ndimage
import numpy as np

class NDBSpline():
    def __init__(self, values, knots, order=3, mode='constant', cval=0.0, prefilter=True):
        self.order = order
        if isinstance(knots, np.ndarray): # mesh
            
            if knots.ndim == 1:
                self.ndim = 1
                knots = knots[None,...]
            else:
                self.ndim = knots.ndim - 1
            self.knots_mesh = knots
            self.knots_vectors = tuple(np.unique(self.knots_mesh[k,...])
                for k in range(self.ndim))
        elif not isinstance(knots, str) and len(knots): # vectors, or try
            self.knots_vectors = knots
            self.knots_mesh = self.meshgrid(knots, indexing='ij')
            self.ndim = len(knots)
        else: # use pixels
            raise ValueError("Don't know how to interpret dimension this")

        self.mode = mode

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
        
        if prefilter:
            self.coeffs = tuple(ndimage.spline_filter(self.values[k,...])
                for k in range(self.values.shape[0]))
        else:
            self.coeffs = self.values
    
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

            total_test = np.r_['0', 
                lt_test[[0],...],
                np.diff(lt_test, axis=0)
                # (lt_test[2:,...] & ge_test[1:-1,...]),
                
                # (self.knots_vectors[k][vec_sel] <
                #     coords[k, ...])[-1,...][None,...],
            ]
            
            where_out = np.nonzero(total_test)
            indices[(k,)+where_out[1:]] = where_out[0]
        return indices

    def coord_edges(self, coords, in_place=False):
        if not in_place:
            coords = coords.copy()
        indices = self.indices_from_coords(coords)
        x0 = np.ones_like(coords)*np.NaN
        dx = np.ones_like(coords)
        for k in range(self.ndim):
            internal_intervals = (k, 
                (indices[k,...]>-1) &
                (indices[k,...]<self.knots_vectors[k].size)
            )
            x0[internal_intervals] = \
                self.knots_vectors[k][indices[internal_intervals]]
            
            # internal_intervals = (k, 
            #     (indices[k,...]>-1) &
            #     (indices[k,...]<self.knots_vectors[k].size-1)
            # )
            dx[internal_intervals] = \
                self.diff_knots_vectors[k][indices[internal_intervals]-1]

        return indices, coords, x0, dx
        
    def transform_coord_to_pixel(self, coords):
        indices, coords, x0, dx = self.coord_edges(coords)
        pixel_coords = indices + (coords - x0)/dx
        return pixel_coords, dx
        
    def evaluate(self, coords, derivative=0):
        coords = self.broadcast_coords(coords)
        output = np.ones((len(self.coeffs),) + coords.shape[1:])
        pixel_coords, dx = self.transform_coord_to_pixel(coords)
        for j in range(len(self.coeffs)):
            output[j,...] = ndimage.map_coordinates(self.coeffs[j],
                pixel_coords, prefilter=False, mode=self.mode)
        return output