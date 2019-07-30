"""
=========================================
Tutorial for basic usage 
=========================================
"""


import ndsplines
import numpy as np
import matplotlib.pyplot as plt

# generate grid of independent variables
x = np.array([-1, -7/8, -3/4, -1/2, -1/4, -1/8, 0, 1/8, 1/4, 1/2, 3/4, 7/8, 1])*np.pi
y = np.array([-1, -1/2, 0, 1/2, 1])
meshx, meshy = np.meshgrid(x, y, indexing='ij')
gridxy = np.stack((meshx, meshy), axis=-1)


# generate denser grid of independent variables to interpolate
sparse_dense = 2**7
xx = np.concatenate([np.linspace(x[i], x[i+1], sparse_dense) for i in range(x.size-1)]) # np.linspace(x[0], x[-1], x.size*sparse_dense)
yy = np.concatenate([np.linspace(y[i], y[i+1], sparse_dense) for i in range(y.size-1)]) # np.linspace(y[0], y[-1], y.size*sparse_dense)
gridxxyy = np.stack(np.meshgrid(xx, yy, indexing='ij'), axis=-1)

def plots(sparse_data, dense_data, ylabel='f(x,y)'):
    fig, axes = plt.subplots(1, 2, constrained_layout=True)
    for yidx in range(sparse_data.shape[1]):
        axes[0].plot(x, sparse_data[:, yidx], 'o', color='C%d'%yidx, label='y=%.2f'%y[yidx])
        axes[0].plot(xx, dense_data[:, np.clip(yidx*sparse_dense, 0, yy.size-1)], color='C%d'%yidx)# label='y=%.1f'%y[yidx])
        
    axes[0].legend()
    axes[0].set_xlabel('x')
    axes[0].set_ylabel(ylabel)
    for xidx in range(sparse_data.shape[0]//2):
        axes[1].plot(yy, dense_data[(xidx+3)*sparse_dense, :], '--', color='C%d'%xidx,)# label='x=%.1f'%x[xidx+3])
        axes[1].plot(y, sparse_data[xidx+3, :], 'o', color='C%d'%xidx, label='x=%.1f'%x[xidx+3],)
        
    axes[1].legend()
    axes[1].set_xlabel('y')
    plt.show()

# evaluate a function to interpolate over input grid
meshf = np.sin(meshx) * (meshy-3/8)**2 + 2

# create the interpolating splane
interp = ndsplines.make_interp_spline(gridxy, meshf)

# evaluate spline over denser grid
meshff = interp(gridxxyy)


plots(meshf, meshff)


##

# as subplots
fig, axes = plt.subplots(1,2, constrained_layout=True)

gridxxy = np.stack(np.meshgrid(xx, y, indexing='ij'), axis=-1)
meshff = interp(gridxxy)

for yidx in range(meshf.shape[1]):
    axes[0].plot(x, meshf[:, yidx], 'o', color='C%d'%yidx, label='y=%.1f'%y[yidx])
    axes[0].plot(xx, meshff[:, yidx], color='C%d'%yidx)
axes[0].legend()
axes[0].set_xlabel('$x$')
axes[0].set_ylabel('$f(x,y)$')

# y-dir plot
gridxyy = np.stack(np.meshgrid(x, yy, indexing='ij'), axis=-1)

meshff = interp(gridxyy)
for xidx in range(meshf.shape[0]//2):
    axes[1].plot(yy, meshff[xidx*1+3, :], '--', color='C%d'%xidx, label='x=%.1f'%x[xidx*1+3])
    axes[1].plot(y, meshf[xidx*1+3, :], 'o', color='C%d'%xidx)
    
axes[1].legend()
axes[1].set_xlabel('$y$')
# plt.ylabel(r'$\frac{\partial f(x,y)}{\partial y}$')
plt.show()

##

# we could also use tidy data format to make the grid

tidy_data = np.dstack((gridxy, meshf)).reshape((-1,3))
print(tidy_data)

tidy_interp = ndsplines.make_interp_spline_from_tidy(tidy_data, [0,1], [2])

print("\nCoefficients all same?", np.all(tidy_interp.coefficients == interp.coefficients))
print("Knots all same?", np.all([np.all(knot0 == knot1) for knot0, knot1 in zip(tidy_interp.knots, interp.knots)]))

# send to example of least squares
##
# two ways to evaluate derivative - y direction

deriv_interp = interp.derivative(1)
deriv1 = deriv_interp(gridxy)
deriv2 = interp(gridxxyy, nus=np.array([0,1]))

plots(deriv1, deriv2, r'$\frac{\partial f(x,y)}{\partial y}$')

##
# two ways to evaluate derivatives x-direction: create a derivative spline or call with nus:
deriv_interp = interp.derivative(0)
deriv1 = deriv_interp(gridxy)
deriv2 = interp(gridxxyy, nus=np.array([1,0]))

plots(deriv1, deriv2, r'$\frac{\partial f(x,y)}{\partial x}$')
##

# Calculus demonstration
interp1 = deriv_interp.antiderivative(0)
coeff_diff = interp1.coefficients - interp.coefficients
print("\nAntiderivative of derivative:\n","Coefficients differ by constant?", np.allclose(interp1.coefficients+2.0, interp.coefficients))
print("Knots all same?", np.all([np.all(knot0 == knot1) for knot0, knot1 in zip(interp1.knots, interp.knots)]))

antideriv_interp = interp.antiderivative(0)

interp2 = antideriv_interp.derivative(0)
print("\nDerivative of antiderivative:\n","Coefficients the same?", np.allclose(interp2.coefficients, interp.coefficients))
print("Knots all same?", np.all([np.all(knot0 == knot1) for knot0, knot1 in zip(interp2.knots, interp.knots)]))
