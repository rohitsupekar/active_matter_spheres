#get spectral coefficients for omega
#script for plotting stuff directly from hard disk and not to be used with the bash script

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
from matplotlib.animation import FFMpegWriter
logger = logging.getLogger(__name__)
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

#add path to data folder
sim_number = 110
input_folder = "/Volumes/ExtDrive/data/sphere%i" %(sim_number)
output_folder = "videos"
first_frame = 1
last_frame = len(glob.glob1("".join([input_folder,'/']),"*.npz"))
last_frame = 3000
dpi = 300
FPS = 20
fields = ['om']
ell_max = 20 #for plotting
marker_size = 0.5
step = 5 #number of frames to skip
vphlim = 10
axs = [None for i in range(3)]

w, h = 0.4, 0.6

#plotting
#plt.rc('font', size=15)
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman'], 'size': 13})

fig = plt.figure(figsize=(8,4.5))
axs[0] = plt.axes((0.1, 0.2, 0.45, h))
axs[1] = plt.axes((0.63, 0.2, 0.33, h))

# Setup output folder
if comm.rank == 0:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
comm.barrier()

max_vals = {key: 0 for key in fields}
clims = {key: 0 for key in fields}
#for field in fields:
#    for i in range(first_frame + comm.rank+1, last_frame + 1, comm.size):
#        with np.load("".join([input_folder, '/output_%i.npz' %i])) as file:
#            fieldval = file[field]
#            max_vals[field] = max(max_vals[field], np.max(fieldval))

#for field in fields:
#    clims[field] = 0.75*max_vals[field]

clims['om'] = 150.0

metadata = dict(title='Movie', artist='Matplotlib', comment='Movie support!')
writer = FFMpegWriter(fps=FPS, metadata=metadata)

with writer.saving(fig, "%s/sphere%i_om_coeffs.mp4" %(output_folder, sim_number), dpi):

    for ind in range(first_frame + comm.rank + 1, last_frame + 1, step):
        if ind%1==0: logger.info("Frame: %i" %(ind))

        with np.load(os.path.join(input_folder, 'output_%i.npz' %(ind))) as file:
            if ind == first_frame + comm.rank +1:
                phi = file['phi']
                theta = file['theta']
                L_max = len(theta)-1
                S_max = 4
                simplesphere = SimpleSphere(L_max, S_max)
                omega = TensorField(simplesphere, rank=0)

            om = file['om']
            vph = np.mean(file['v_ph'], axis=0)
            print(np.max(vph))
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

        mag = np.abs(coeffs_arr)
        phase = np.angle(coeffs_arr)

        if ind == first_frame + comm.rank +1:
            mag_fac = marker_size/np.max(mag)

        m = np.arange(0,L_max+1)
        ell = np.arange(0,L_max+1)
        ellell, mm = np.meshgrid(ell, m)

        if ind == first_frame + comm.rank +1:
            title = fig.suptitle(r'$t/\tau = %.4f$' %time, usetex=True)

            ax = axs[0]
            img0 = ax.pcolormesh(phi, np.pi/2-theta, om.T, cmap='RdBu_r', shading='garoud', rasterized=True)
            ax.set_ylabel(r'Latitude $(\pi/2-\theta)$', usetex=True);
            ax.set_yticks([-np.pi/2, 0, np.pi/2])
            ax.set_yticklabels([r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$'])
            ax.set_xlabel(r'Longitude $\phi$', usetex=True)
            ax.set_xticks([0, np.pi, 2*np.pi])
            ax.set_xticklabels([r'$0$', r'$\pi$', r'$2 \pi$'])
            img0.set_clim([-clims['om'], clims['om']])

             #add colorbar
            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes("top", size="7%", pad="4%")
            cb = plt.colorbar(img0, cax=cax, orientation="horizontal")
            cax.xaxis.set_ticks_position("top")
            cb.set_ticks([-clims['om'], 0, clims['om']])

            #add axis for v_ph
            vph_ax = ax_divider.append_axes("right", size="30%", pad="17%")
            line, = vph_ax.plot(vph, np.pi/2-theta, 'k', linewidth=1)
            vph_ax.set_yticks([]); vph_ax.set_ylim([-np.pi/2, np.pi/2])
            vph_ax.set_xlim([-vphlim, vphlim]);
            vph_ax.set_xticks([-vphlim, 0, vphlim]);
            vph_ax.axvline(linestyle='--',color='k',linewidth=0.5)
            vph_ax.set_xlabel(r'$\langle v_\phi\rangle_\phi/(R/\tau)$')

            ax = axs[1]
            img1 = ax.scatter(mm.flatten(), ellell.flatten(), mag_fac*mag.flatten(), c=phase.flatten(), \
                    cmap='hsv', edgecolor='none')
            rect = Rectangle((-1, 10.4), ell_max+1, 4, facecolor='k', alpha=0.2)
            ax.add_patch(rect)
            ax.set_xlim(-1, ell_max), ax.set_ylim(-1, ell_max)
            ax.set_xlabel('$m$', usetex=True), ax.set_ylabel('$\ell$', usetex=True, rotation=0)
            img1.set_clim(0, 2*np.pi)
            ax_divider = make_axes_locatable(ax)
            # add an axes above the main axes.
            cax = ax_divider.append_axes("top", size="7%", pad="4%")
            cb = plt.colorbar(img1, cax=cax, orientation="horizontal")
            cb.set_ticks([0, np.pi, 2*np.pi])
            cb.set_ticklabels(['$0$', r'$\pi$', r'$2 \pi$'])
            cax.xaxis.set_ticks_position("top")

            #fig.tight_layout()

        else:
            title.set_text(r'$t/\tau = %.4f$' %time)
            img0.set_array(om.T.ravel())
            img1.set_sizes(mag_fac*mag.flatten())
            img1.set_array(phase.flatten())
            line.set_xdata(vph)

        writer.grab_frame()
