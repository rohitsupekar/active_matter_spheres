import os
import sys
sys.path.append("../") # go to parent dir
import glob
import time
import logging
import numpy as np
from scipy.sparse import linalg as spla
import matplotlib.pyplot as plt
import logging
from mpl_toolkits import mplot3d
from mayavi import mlab
from scipy.special import sph_harm

#add path to data folder
input_folder = "/Volumes/ExtDrive/data"
output_folder = "plots"
dpi=300
cmap = "RdBu"

ind = 1500
f00 = "%s/sphere113/output_%i.npz" %(input_folder, ind)
f01 = "%s/sphere114/output_%i.npz" %(input_folder, ind)
f02 = "%s/sphere115/output_%i.npz" %(input_folder, ind)

f10 = "%s/sphere111/output_%i.npz" %(input_folder, ind)
f11 = "%s/sphere109/output_%i.npz" %(input_folder, ind)
f12 = "%s/sphere110/output_%i.npz" %(input_folder, ind)

f20 = "%s/sphere116/output_%i.npz" %(input_folder, ind)
f21 = "%s/sphere117/output_%i.npz" %(input_folder, ind)
f22 = "%s/sphere118/output_%i.npz" %(input_folder, ind)

fs = [[f00, f01, f02], [f10, f11, f12], [f20, f21, f22]]
om_list = [[None, None, None] for i in range(3)]

#load data
for i, ls in enumerate(fs):
    for j, str in enumerate(ls):
        with np.load(str) as file:
            print('Loaded %s' %(str))
            om_list[i][j] = file['om']
            phi = file['phi']
            theta = file['theta']
            time = file['t'][0]
            print('time = %f' %(time))

#change phi
phi = np.linspace(0, 2*np.pi, len(phi))

# Create a sphere
r = 0.3
pi = np.pi
cos = np.cos
sin = np.sin
phiphi, thth = np.meshgrid(np.pi-theta, phi-pi)

x = r * sin(phiphi) * cos(thth)
y = r * sin(phiphi) * sin(thth)
z = r * cos(phiphi)

#s = sph_harm(0, 10, theta, phi).real
m = mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(1, 1, 1), size=(1500, 1600))
mlab.clf()

cmin, cmax = -300, 300
dx = 0.7
for i, ls in enumerate(om_list):
    for j, om in enumerate(ls):
        om_max = np.max(om)
        if fs[i][j] == f02:
            mlab.mesh(x+j*dx, y, z-i*dx, scalars=om, colormap=cmap, vmax=0.3*om_max, vmin=-0.3*om_max)
        else:
            mlab.mesh(x+j*dx, y, z-i*dx, scalars=om, colormap=cmap, vmax=0.6*om_max, vmin=-0.6*om_max)

#plot the spherical harmonics
for j, ell in enumerate([6, 11, 21]):
    s = sph_harm(0, ell, thth, phiphi).real
    mlab.mesh(x + 3*dx , y, z-j*dx, scalars=s, colormap=cmap, vmax=0.4, vmin=-.4)

mlab.view(-90, 90, distance=4)
#mlab.savefig("%s/fig1_headon.pdf" %(output_folder), magnification=1)
mlab.show()
