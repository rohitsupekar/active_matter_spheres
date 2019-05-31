import os
import sys
sys.path.append("../") # go to parent dir
import glob
import time
import pathlib
import logging
import numpy as np
from scipy.sparse import linalg as spla
from dedalus.tools.config import config
from simple_sphere import SimpleSphere, TensorField, TensorSystem
import equations
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from dedalus.extras import plot_tools
import logging
from mpl_toolkits import mplot3d
import pickle
logger = logging.getLogger(__name__)

#sphere_inds = [48, 49, 50, 51, 55]
sphere_inds = [48, 49, 50, 51]
load_raw_data = False

output_folder = "plots"


plt.figure(figsize=(6,4), dpi=200)

ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)

for sphere_ind in sphere_inds:

    if load_raw_data:

        input_folder = "data/sphere%i" %(sphere_ind)

        logger.info('sphere %i' %(sphere_ind))
        first_frame = 1
        last_frame = len(glob.glob1("".join([input_folder,'/']),"*.npz"))
        dpi = 256
        fields = ['om']

        # Setup output folder
        if not os.path.exists(output_folder):
                os.makedirs(output_folder)

        t_arr = np.zeros(last_frame)

        for ind in range(first_frame, last_frame + 1, 1):
            if np.mod(ind,100)==0: logger.info('Frame: %i' %ind)

            with np.load(os.path.join(input_folder, 'output_%i.npz' %(ind))) as file:
                if ind == first_frame:
                    phi = file['phi']
                    theta = file['theta']
                    L_max = len(theta)-1
                    S_max = 4
                    simplesphere = SimpleSphere(L_max, S_max)
                    omega = TensorField(simplesphere, rank=0)
                    coeffs_all = np.zeros((last_frame,L_max+1, L_max+1), dtype=complex)

                om = file['om']
                time = file['t'][0]
                t_arr[ind-1] = time

            # assign loaded data
            omega.component_fields[0]['g'] = om
            # spectral transform
            omega.forward_phi()
            omega.forward_theta()
            coeffs = omega.coeffs

            for m in range(len(coeffs)):
                coeffs_all[ind-1, m, m:] = coeffs[m]

        E = np.zeros(t_arr.shape)
        enst = np.zeros(t_arr.shape)
        L_max = len(theta)-1

        #calculate energy
        for m in range(L_max+1):
            for ell in range(L_max+1):
                if ell!=0:
                    E = E + (np.abs(coeffs_all[:,m,ell])**2)/(ell*(ell+1))
                    enst = enst + (np.abs(coeffs_all[:,m,ell])**2)

        with open('plots/sphere%i.pkl' %(sphere_ind), 'wb') as ff:
            pickle.dump([t_arr, E, enst], ff)


    else:
        with open('plots/sphere%i.pkl' %(sphere_ind), 'rb') as ff:
            logger.info('Loaded sphere%i' %(sphere_ind))
            t_arr, E, enst = pickle.load(ff)


    ax1.plot(t_arr, E, linewidth=1); ax1.set_title('Energy')

    t_start = 20; dt = t_arr[1] - t_arr[0]
    ind_start = int(np.floor(t_start/dt))

    #DFT of the energy
    E_fft = np.fft.fftshift(np.fft.fft(E[ind_start:] - np.mean(E[ind_start:])))
    freq = np.fft.fftshift(np.fft.fftfreq(t_arr[ind_start:].shape[-1], d = t_arr[2]-t_arr[1]))

    ax2.plot(freq, np.log10(np.abs(E_fft)), linewidth=1); ax2.set_title('log(DFT of Energy)')
    ax2.set_xlim([-3, 3]); ax2.set_ylim([0, 4])
    ax3.plot(t_arr, enst, linewidth=1); ax3.set_title('Enstrophy')

    #DFT of the enstrophy
    enst_fft = np.fft.fftshift(np.fft.fft(enst[ind_start:] - np.mean(enst[ind_start:])))

    ax4.plot(freq, np.log10(np.abs(enst_fft)), linewidth=1); ax4.set_title('log(DFT of Enstrophy)')
    ax4.set_xlim([-3, 3]); ax4.set_ylim([2, 6])


#ax1.legend(['dt=0.5', 'dt=0.1', 'dt=0.05', 'dt=0.025', 'dt=0.1(L=512)'], prop={'size': 5})
ax1.legend(['dt=0.5', 'dt=0.1', 'dt=0.05', 'dt=0.025'], prop={'size': 5})
plt.tight_layout()
plt.savefig("%s/energy_plot.pdf" %(output_folder))
plt.show()
