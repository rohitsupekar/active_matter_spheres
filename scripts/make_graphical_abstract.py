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
dpi=600

ind = 2100
f00 = "%s/sphere113/output_%i.npz" %(input_folder, ind)
f01 = "%s/sphere114/output_%i.npz" %(input_folder, ind)
f02 = "%s/sphere115/output_%i.npz" %(input_folder, ind)

f10 = "%s/sphere111/output_%i.npz" %(input_folder, ind)
f11 = "%s/sphere109/output_%i.npz" %(input_folder, ind)
f12 = "%s/sphere110/output_%i.npz" %(input_folder, ind)

f20 = "%s/sphere116/output_%i.npz" %(input_folder, ind)
f21 = "%s/sphere117/output_%i.npz" %(input_folder, ind)
f22 = "%s/sphere118/output_%i.npz" %(input_folder, ind)

fs = [f12, f22]
om_list = [None for i in range(2)]

#load data
for i, str in enumerate(fs):
    with np.load(str) as file:
        print('Loaded %s' %(str))
        om_list[i] = file['om']
        phi = file['phi']
        theta = file['theta']
        time = file['t'][0]
        print('time = %f' %(time))

#change phi
phi = np.linspace(0, 2*np.pi, len(phi))
theta = np.flip(np.linspace(0, np.pi, len(theta)))

# Create a sphere
r = 0.3
pi = np.pi
cos = np.cos
sin = np.sin
phiphi, thth = np.meshgrid(theta, phi-pi)

x = r * sin(phiphi) * cos(thth)
y = r * sin(phiphi) * sin(thth)
z = r * cos(phiphi)

#s = sph_harm(0, 10, theta, phi).real
mlab.figure(1, bgcolor=(0, 0, 0), fgcolor=(1, 1, 1), size=(720, 600))
mlab.clf()

cmin, cmax = -300, 300

for i, om in enumerate(om_list):
    om_max = np.max(om)
    scale = 0.5
    spacing = 1.0
    mlab.mesh(x, y- spacing*i, z, scalars=om, colormap='coolwarm', vmax=scale*om_max, vmin=-scale*om_max)

    # Plot the equator and the tropics
    for angle in (-np.pi/3, -np.pi/6, 0., np.pi/6, np.pi/3):
        x_ = r*np.cos(phi) * np.cos(angle)
        y_ = r*np.sin(phi) * np.cos(angle)
        z_ = r*np.ones_like(phi) * np.sin(angle)

        mlab.plot3d(x_, y_-spacing*i, z_, color=(0, 0, 0),
                            opacity=1, tube_radius=0.003)

    th_ = np.linspace(-np.pi/3, np.pi/3, 100)
    for angle in np.linspace(0, 2*np.pi, 16):
        x_ = r*np.cos(angle) * np.cos(th_)
        y_ = r*np.sin(angle) * np.cos(th_)
        z_ = r*np.ones_like(angle) * np.sin(th_)

        mlab.plot3d(x_, y_-spacing*i, z_, color=(0, 0, 0),
                            opacity=1, tube_radius=0.003)

mlab.view(60, 63, distance=2.5)
mlab.savefig("%s/abstract.jpg" %(output_folder), magnification=2)
mlab.show()
