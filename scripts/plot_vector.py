
print('starting')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap, shiftgrid
from matplotlib           import cm, colors

import sphere as sph
import equations as eq

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

L_max = 255
grid = sph.grid(L_max)

print('made grid')

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

for i in range(1+rank,3,size):

  print(i)
  output = np.load('output_files/output_%i.npz' %i)

  print(output['p'].shape)

  clim = 500.
  norm = matplotlib.colors.SymLogNorm(10,linscale=0.1,vmin=-clim,vmax=clim)
  im = mp.pcolor(x, y, np.transpose(output['p']), cmap='RdBu_r', norm=norm)

  if i == 1+rank:
    plt.colorbar(im)
    title = figure.suptitle('t = %.4f' %output['t'][0])
  else:
    title.set_text('t = %.4f' %output['t'][0])

  vph,newlon = shiftgrid(180.,output['vph'].T,np.degrees(lon),start=False)
  vth,newlon = shiftgrid(180.,output['vth'].T,np.degrees(lon),start=False)

  uproj,vproj,xx,yy = mp.transform_vector(vph[::-1],vth[::-1],newlon,np.degrees(lat[::-1]),81,81,returnxy=True,masked=True)

  Q = mp.quiver(xx,yy,uproj,vproj,scale=1000,pivot = 'middle', headwidth=2, headlength=2, headaxislength=2)

  plt.savefig('images/vector_%05i.png' %i,dpi=300)


