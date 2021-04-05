# This is a python 3 script for solving the Euler equations in 2D on a doubly periodic domain (torus)
# We solve the vorticity formulation

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time

def euler_td(omega):
  n = omega.shape[0];
 
  k_vec = np.zeros(n);
  k_vec[0:(n//2+1)] = np.arange(n/2 + 1)
  k_vec[(n//2+1):]  = np.arange(1-n//2,0)
  kx, ky = np.meshgrid( k_vec, k_vec )
  
  k_sq = np.multiply(kx, kx) +  np.multiply(ky, ky)
  k_sq[0,0] = 1

  u = np.fft.ifft2(  np.divide( np.multiply( np.complex(0,1)*ky, omega ), k_sq ) )
  v = np.fft.ifft2(  np.divide( np.multiply(-np.complex(0,1)*kx, omega ), k_sq ) )
  
  omega_x = np.fft.ifft2(  np.multiply( np.complex(0,1)*kx, omega ) )
  omega_y = np.fft.ifft2(  np.multiply( np.complex(0,1)*ky, omega ) )

  advec = np.multiply( u, omega_x ) + np.multiply( v, omega_y )
  
  td = -n*n*np.fft.fft2( advec )

  cutoff = 60.
  
  td[ k_sq > cutoff**2 ] = 0.
  td[0,0] = 0 #make sure the zero mode is constant (and zero)
  return td
  

def rk4_step( omega, dt ):
  #standard 4th order Runge-Kutta
  #I use o as my temporary omega
  k1 = euler_td(omega)
  o = omega + dt*k1/2
  k2 = euler_td(o)
  o = omega + dt*k2/2
  k3 = euler_td(o)
  o = omega + dt*k3
  k4 = euler_td(o)
  return omega + dt*(k1+2*k2+2*k3+k4)/6




#n is the number of grid points per side
n = 2**9
omega = np.zeros( (n,n) )

#Use meshgrid to define x and y
side_vector = np.arange(n)/n*2*np.pi
x,y = np.meshgrid(side_vector, side_vector)

omega_sp =  np.sin( 3*x )*np.sin(3*y) - 0.1*np.sin(x)*np.sin(y) + 0.001*np.cos(5*x)
omega = np.fft.ifft2(omega_sp)

#norm is used for plotting.
norm = mcolors.Normalize(vmin=-1, vmax=1)

out_every = 10
dt = 0.1
for timestep in range(1000):
  print(timestep)
  omega = rk4_step(omega, dt)

  #Take the Fourier transform back to space and plot
  omega_sp = np.fft.fft2( omega )
  plt.imshow( omega_sp.real, cmap=plt.get_cmap('bwr'), norm=norm )
  plt.title('vorticity    t=%.03f' % (timestep*dt) )
  plt.colorbar( ticks=[-1,0,1] )

  time.sleep(0.01)
  if timestep % out_every == 0:
    plt.gcf().savefig('frames/%05d.png' % timestep)
  plt.clf()
