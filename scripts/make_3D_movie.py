import os
import sys
sys.path.append("../") # go to parent dir
import glob
import logging
from scipy.sparse import linalg as spla
import matplotlib.pyplot as plt
import logging
from mpl_toolkits import mplot3d
import time
from mayavi import mlab
import numpy as np


#add path to data folder
input_folder = "data/"
output_folder = "plots"
dpi=300
run_name = 'sphere110'
save_name = 'double_angle1'
offscreen = True

mlab.options.offscreen = offscreen

ind_start = 800
ind_end = 3000

with np.load(os.path.join(input_folder, 'sphere110/output_%i.npz' %(ind_start))) as file:
    om = file['om']
    time = file['t'][0]
    phi = file['phi']
    theta = file['theta']
    print('time=%f' %time)

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

mlab.figure(1, bgcolor=(0, 0, 0), fgcolor=(1, 1, 1), size=(1500, 1500))
mlab.clf()
plot = mlab.mesh(x, y, z, scalars=om, colormap='RdBu', resolution=20, vmin=-120, vmax=120)


start_angle = 70
end_angle = 10
start_dist = 1.7
end_dist = 1.2
degs_per_ind = 0.014
dist_per_ind = 0.0001

deg_arr = np.zeros(ind_end-ind_start+1)
n_inds = len(deg_arr)
n_half = int(n_inds/2)
deg_arr[0:n_half] = np.linspace(start_angle, end_angle, n_half)
deg_arr[n_half:] = np.linspace(end_angle, start_angle, n_inds-n_half)

dist_arr = np.zeros_like(deg_arr)
dist_arr[0:n_half] = np.linspace(start_dist, end_dist, n_half)
dist_arr[n_half:] = np.linspace(end_dist, start_dist, n_inds-n_half)

mlab.view(-90, start_angle, distance=start_dist)

# Plot the equator and the tropics
for angle in (-np.pi/3, -np.pi/6, 0., np.pi/6, np.pi/3):
    x_ = r*np.cos(phi) * np.cos(angle)
    y_ = r*np.sin(phi) * np.cos(angle)
    z_ = r*np.ones_like(phi) * np.sin(angle)

    mlab.plot3d(x_, y_, z_, color=(0, 0, 0),
                        opacity=1, tube_radius=None, representation='wireframe')

th_ = np.linspace(-np.pi/3, np.pi/3, 100)
for angle in np.linspace(0, 2*np.pi, 16):
    x_ = r*np.cos(angle) * np.cos(th_)
    y_ = r*np.sin(angle) * np.cos(th_)
    z_ = r*np.ones_like(angle) * np.sin(th_)

    mlab.plot3d(x_, y_, z_, color=(0, 0, 0),
                        opacity=1, tube_radius=None, representation='wireframe')

#@mlab.animate(delay=100)
#def anim():
for count, ind in enumerate(range(ind_start,ind_end)):
    with np.load(os.path.join(input_folder, 'sphere110/output_%i.npz' %(ind))) as file:
        om = file['om']
        time = file['t'][0]
        print('time=%f' %time)

    #if time>7:
    #    deg = start_angle - degs_per_ind*(count-count0)
    #    dist = start_dist - dist_per_ind*(count-count0)
    #else:
    #    deg = start_angle
    #    dist = start_dist
    #    count0 = count
    deg = deg_arr[count]
    dist = dist_arr[count]

    mlab.view(-90, deg, distance=dist)

    plot.mlab_source.set(x=x, y=y, z=z, scalars=om)
    #mlab.title('t=%0.3f' %(time), size=0.5)
    mlab.savefig('3d_plots/%s_%i.png' %(save_name, count), magnification=1)
    #yield

#anim()
mlab.show()
