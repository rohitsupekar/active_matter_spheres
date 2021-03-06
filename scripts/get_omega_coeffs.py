#get spectral coefficients for omega

import os
import sys
import glob
import time
import pathlib
import logging
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
from scipy.sparse import linalg as spla
from dedalus.tools.config import config
from simple_sphere import SimpleSphere, TensorField, TensorSystem
import equations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from dedalus.extras import plot_tools
import logging
logger = logging.getLogger(__name__)

#add path to data folder
#input_folder = "/Users/Rohit/Documents/research/active_matter_spheres/scripts/data/sphere3"
#output_folder = "/Users/Rohit/Documents/research/active_matter_spheres/scripts/garbage"
input_folder = "data/sphere22"
output_folder = "images/sphere22"
first_frame = 1
last_frame = len(glob.glob1("".join([input_folder,'/']),"*.npz"))
dpi = 256
fields = ['om']
ell_max = 10 #for plotting

# Setup output folder
if comm.rank == 0:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
comm.barrier()

max_vals = {key: 0 for key in fields}
clims = {key: 0 for key in fields}
for field in fields:
    for i in range(first_frame + comm.rank, last_frame + 1, comm.size):
        with np.load("".join([input_folder, '/output_%i.npz' %i])) as file:
            fieldval = file[field]
            max_vals[field] = max(max_vals[field], np.max(fieldval))

for field in fields:
    clims[field] = 0.75*max_vals[field]


fig, ax = plt.subplots(2,2,figsize=(10, 10), dpi=dpi)
fig.subplots_adjust(hspace=.4)
#plotting
plt.rc('font', size=15)

for ind in range(first_frame + comm.rank, last_frame + 1, comm.size):
    logger.info('Iterations: %i' %ind)

    with np.load(os.path.join(input_folder, 'output_%i.npz' %(ind))) as file:
        if ind == first_frame + comm.rank:
            phi = file['phi']
            theta = file['theta']
            L_max = len(theta)-1
            S_max = 4
            simplesphere = SimpleSphere(L_max, S_max)
            omega = TensorField(simplesphere, rank=0)

        om = file['om']
        time = file['t'][0]


    # assign loaded data
    omega.component_fields[0]['g'] = om
    # spectral transform
    omega.forward_phi()
    omega.forward_theta()
    coeffs = omega.coeffs

    #assign coeffs to a numpy array
    coeffs_arr = np.zeros([L_max+1, L_max+1], dtype=complex)
    for m in range(len(coeffs)):
        coeffs_arr[m,m:] = coeffs[m]

    m = np.arange(0,L_max+1)
    ell = np.arange(0,L_max+1)
    ellell, mm = np.meshgrid(ell, m)

    if ind == first_frame + comm.rank:
        title = fig.suptitle('t = %.4f' %time)

        image0 = ax[0,0].pcolormesh(phi, theta, om.T, cmap='RdBu_r')
        ax[0,0].set_ylabel("$\\theta$"), ax[0,0].set_xlabel("$\phi$")
        ax[0,0].set_title("$\omega (\\theta, \phi)$",fontsize=15)
        ax[0,0].invert_yaxis()
        fig.colorbar(image0, ax=ax[0,0])
        image0.set_clim(-clims['om'], clims['om'])

        image1 = ax[0,1].scatter(mm.flatten(), ellell.flatten(), 2*np.abs(coeffs_arr).flatten(), c='k', edgecolor='none')
        ax[0,1].set_xlim(-1, ell_max), ax[0,1].set_ylim(-1, ell_max)
        ax[0,1].set_xlabel('$m$'), ax[0,1].set_ylabel('$\ell$')
        ax[0,1].set_title('Abs $\hat{\omega}_{\ell, m}$',fontsize=15)
        ax[0,1].set_aspect('equal')

        isreal = (np.real(coeffs_arr) > 0)*1 + (np.real(coeffs_arr) <= 0)*-1
        image2 = ax[1,0].scatter(mm.flatten(), ellell.flatten(), 2*np.abs(np.real(coeffs_arr)).flatten(), c=isreal.flatten(), cmap='bwr', edgecolor='none')
        ax[1,0].set_xlim(-1, ell_max), ax[1,0].set_ylim(-1, ell_max)
        ax[1,0].set_xlabel('$m$'), ax[1,0].set_ylabel('$\ell$')
        ax[1,0].set_title('Real $\hat{\omega}_{\ell, m}$ (red=+ve, blue=-ve)', fontsize=15)

        isreal = (np.imag(coeffs_arr) > 0)*1 + (np.imag(coeffs_arr) <= 0)*-1
        image3 = ax[1,1].scatter(mm.flatten(), ellell.flatten(), 2*np.abs(np.imag(coeffs_arr)).flatten(), c=isreal.flatten(), cmap='bwr', edgecolor='none')
        ax[1,1].set_xlim(-1, ell_max), ax[1,1].set_ylim(-1, ell_max)
        ax[1,1].set_xlabel('$m$'), ax[1,1].set_ylabel('$\ell$')
        ax[1,1].set_title('Imag $\hat{\omega}_{\ell, m}$ (red=+ve, blue=-ve)',fontsize=15)
    else:
        title.set_text('t = %.4f' %time)
        image0.set_array(om.T.ravel())
        image1.set_sizes(2*np.abs(coeffs_arr).flatten())
        image2.set_sizes(2*np.abs(np.real(coeffs_arr)).flatten())
        image3.set_sizes(2*np.abs(np.imag(coeffs_arr)).flatten())

        isreal = (np.real(coeffs_arr) > 0)*1 + (np.real(coeffs_arr) <= 0)*-1
        image2.set_array(isreal.flatten())
        isreal = (np.imag(coeffs_arr) > 0)*1 + (np.imag(coeffs_arr) <= 0)*-1
        image3.set_array(isreal.flatten())
        #image1.set_array(1.5*np.abs(coeffs_arr).ravel())

    plt.savefig(os.path.join(output_folder, 'om_coeffs_%05i.png' %ind), dpi=dpi)
