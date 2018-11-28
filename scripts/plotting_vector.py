
print('starting')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap, shiftgrid
from matplotlib           import cm, colors

from scipy.integrate import simps
from scipy.sparse import linalg as spla
import os
import dedalus.public as de
import time
import pathlib
from mpi4py import MPI
import sphere as sph
import equations as eq

from mpi4py import MPI

Lmid = 10.0
kappa = 0.5
factor = 0.5
firstFrame = 1   #first frame to be plotted
lastFrame = 50   #last frame to be plotted


comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size
L_max = 255
S_max = 4  # spin order (leave fixed)
grid = sph.grid(L_max)
dt = factor/(100*Lmid**2)

# Make domain
phi_basis   = de.Fourier('phi'  , 2*(L_max+1), interval=(0,2*np.pi))
theta_basis = de.Fourier('theta', L_max+1, interval=(0,np.pi))
domain = de.Domain([phi_basis,theta_basis], grid_dtype=np.float64)
phi = domain.grids(1)[0]

# set up fields
v_th = domain.new_field()
v_ph = domain.new_field()
p    = domain.new_field()
om   = domain.new_field()
psi  = domain.new_field()
# set up sphere
p.require_coeff_space()
m_start = p.layout.start(1)[0]
m_len = p.layout.local_shape(1)[0]
S = sph.Sphere(L_max,S_max,m_min=m_start,m_max=m_start+m_len)

# Calculate theta grid
theta_slice = domain.distributor.layouts[-1].slices(1)[1]
theta_len = domain.local_grid_shape(1)[1]
theta = S.grid[theta_slice].reshape([1,theta_len])

figure, ax = plt.subplots(1,1)
figure.set_size_inches(6,6)
lon = np.linspace(0, 2*np.pi, 2*(L_max+1))
lat = grid - np.pi/2

meshed_grid = np.meshgrid(lon, lat)
lat_grid = meshed_grid[1]
lon_grid = meshed_grid[0]

mp = Basemap(projection='ortho', lat_0=33, lon_0=0, ax=ax)
mp.drawmapboundary()
mp.drawmeridians(np.arange(0, 360, 30),dashes = (None,None), linewidth=0.5)
mp.drawparallels(np.arange(-90, 90, 30),dashes = (None,None), linewidth=0.5)

x, y = mp(np.degrees(lon_grid), np.degrees(lat_grid))

for i in range(firstFrame+rank,lastFrame,size):

  output = np.load('dataFolder/output_%i.npz' %i)
  clim = 10000 # will plot clim*pressure/maximum pressure on log scale, to emphasize the bands of pressure higher than mean

  #calculate mean pressure on the sphere, and shift pressure to be deviation from this mean
  z = np.multiply( np.transpose(output['p']), np.sin(lat_grid+np.pi/2));
  intP = simps(simps(z, lon), lat)
  pToPlot = output['p'] - intP

  maxAbs = np.abs(pToPlot).max()
  maxThing = pToPlot.max()
  minThing = pToPlot.min()
  norm = matplotlib.colors.SymLogNorm(10,linscale=0.1,vmin=-clim,vmax=clim)
  im = mp.pcolor(x, y, 10000*np.transpose(pToPlot)/maxAbs, cmap='RdBu_r', norm=norm)
  if i == firstFrame+rank:
    plt.colorbar(im)
    title = figure.suptitle('t = %.4f' %output['t'][0])
  else:
    title.set_text('t = %.4f' %output['t'][0])

  vph,newlon = shiftgrid(180.,output['vph'].T,np.degrees(lon),start=False)
  vth,newlon = shiftgrid(180.,output['vth'].T,np.degrees(lon),start=False)

  uproj,vproj,xx,yy = mp.transform_vector(vph[::-1],vth[::-1],newlon,np.degrees(lat[::-1]),81,81,returnxy=True,masked=True)

  Q = mp.quiver(xx,yy,uproj,vproj,scale=10000,pivot = 'middle', headwidth=4, headlength=4, headaxislength=2)

  mypath = 'imagesRescaled_kappa' + str(kappa) + 'Lmid' + str(Lmid)+'dt'+str(dt)
  if not os.path.exists(mypath):
       os.makedirs(mypath)
  plt.savefig(mypath+'/pressure_%05i.png' %i,dpi=300)
  np.savetxt(mypath+'/maxparamsPressure',[maxThing, minThing])


