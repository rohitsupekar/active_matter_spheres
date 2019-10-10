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

mlab.options.offscreen = True

#add path to data folder
input_folder = "data/"
output_folder = "plots"
dpi=300

ind = 4999 #time ind

with np.load(os.path.join(input_folder, 'sphere113/output_%i.npz' %(ind))) as file:
    om1 = file['om']
    time = file['t'][0]
    print('time=%f' %time)

with np.load(os.path.join(input_folder, 'sphere114/output_%i.npz' %(ind))) as file:
    om2 = file['om']
    time = file['t'][0]
    print('time=%f' %time)

with np.load(os.path.join(input_folder, 'sphere115/output_%i.npz' %(ind))) as file:
    om3 = file['om']
    time = file['t'][0]
    print('time=%f' %time)

with np.load(os.path.join(input_folder, 'sphere111/output_%i.npz' %(ind))) as file:
    om4 = file['om']
    time = file['t'][0]
    print('time=%f' %time)

with np.load(os.path.join(input_folder, 'sphere109/output_%i.npz' %(ind))) as file:
    om5 = file['om']
    time = file['t'][0]
    print('time=%f' %time)

with np.load(os.path.join(input_folder, 'sphere110/output_%i.npz' %(ind))) as file:
    om6 = file['om']
    time = file['t'][0]
    print('time=%f' %time)


with np.load(os.path.join(input_folder, 'sphere116/output_%i.npz' %(ind))) as file:
    om7 = file['om']
    time = file['t'][0]
    print('time=%f' %time)

with np.load(os.path.join(input_folder, 'sphere117/output_%i.npz' %(ind))) as file:
    om8 = file['om']
    time = file['t'][0]
    print('time=%f' %time)

with np.load(os.path.join(input_folder, 'sphere118/output_%i.npz' %(ind))) as file:
    phi = file['phi']
    theta = file['theta']
    om9 = file['om']
    time = file['t'][0]
    print('time=%f' %time)

#change phi
phi = np.linspace(0, 2*np.pi, len(phi))



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
mlab.figure(1, bgcolor=(0, 0, 0), fgcolor=(1, 1, 1), size=(800, 700))
mlab.clf()

cmin, cmax = -300, 300
dx = 0.7
m = mlab.mesh(x, y, z+2*dx, scalars=om1, colormap='bwr')
m = mlab.mesh(x+dx, y, z+2*dx, scalars=om2, colormap='bwr')
m = mlab.mesh(x+2*dx, y, z+2*dx, scalars=om3, colormap='bwr')

m = mlab.mesh(x, y, z+dx, scalars=om4, colormap='bwr')
m = mlab.mesh(x+dx, y, z+dx, scalars=om5, colormap='bwr')
m = mlab.mesh(x+2*dx, y, z+dx, scalars=om6, colormap='bwr')

m = mlab.mesh(x, y, z, scalars=om7, colormap='bwr')
m = mlab.mesh(x+dx, y, z, scalars=om8, colormap='bwr')
m = mlab.mesh(x+2*dx, y, z, scalars=om9, colormap='bwr')


mlab.view(-90, 90, distance=4)
#mlab.savefig("%s/mayavi.pdf" %(output_folder), magnification=100)
#mlab.show()

#mlab.figure(2, bgcolor=(0, 0, 0), fgcolor=(1, 1, 1), size=(700, 300))
#mlab.clf()
#m = mlab.mesh(x, y, z, scalars=om3, colormap='bwr')
#m = mlab.mesh(x+0.7, y, z, scalars=om6, colormap='bwr')
#m = mlab.mesh(x+1.4, y, z, scalars=om9, colormap='bwr')
#mlab.view(-90, 90, distance=1.5)

mlab.savefig("%s/mayavi_front.pdf" %(output_folder), magnification=100)


#mlab.show()
