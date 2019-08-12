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
logger = logging.getLogger(__name__)

sphere_inds = [3]

plt.figure(figsize=(4,3), dpi=200)

for sphere_ind in sphere_inds:

    input_folder = "data/sphere%i" %(sphere_ind)
    output_folder = "plots"

    print('sphere%i' %(sphere_ind))
    first_frame = 1
    last_frame = len(glob.glob1("".join([input_folder,'/']),"*.npz"))
    dpi = 256
    fields = ['om']

    # Setup output folder
    if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    t_arr = np.zeros(last_frame)

    for ind in range(first_frame, last_frame + 1, 1):
        logger.info('Frame: %i' %ind)

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
    L_max = len(theta)-1

    #calculate energy
    for m in range(L_max+1):
        for ell in range(L_max+1):
            if ell!=0:
                E = E + (np.abs(coeffs_all[:,m,ell])**2)/(ell*(ell+1))

    plt.plot(t_arr, E)

plt.savefig("%s/energy_plot.eps" %(output_folder))
plt.show()
