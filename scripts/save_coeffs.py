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
from dedalus.tools.config import config
from simple_sphere import SimpleSphere, TensorField, TensorSystem
import equations
import pickle as pkl
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

#add path to data folder
input_folder = "data"
output_folder = "coeffs_data"
dpi=300

first_frame = 1
last_frame = 1000

sphere_list = ['sphere%i' %(i) for i in range(141, 150)]
fs = ["%s/%s" %(input_folder, sphere_list[i]) for i in range(len(sphere_list))]

t_arr = np.zeros(last_frame - first_frame + 1)

#create tensor objects
L_max = 255
S_max = 4
simplesphere = SimpleSphere(L_max, S_max)
omega = TensorField(simplesphere, rank=0)

om_coeffs = np.zeros((last_frame-first_frame+1, 256, 256), dtype=complex)
om = np.zeros((512, 256))

# load omega data
for i, str in enumerate(fs):

    for j, ind in enumerate(range(first_frame, last_frame+1, 1)):

        with np.load("%s/output_%i.npz" %(fs[i], ind)) as file:
            if ind%100==0: print("Loaded %s/output_%i.npz" %(fs[i], ind))

            om = file['om']
            phi = file['phi']
            theta = file['theta']
            t_arr[j] = file['t'][0]

        # assign loaded data
        omega.component_fields[0]['g'] = om
        # spectral transform
        omega.forward_phi()
        omega.forward_theta()
        for m in range(len(omega.coeffs)):
            om_coeffs[j, m, m:] = omega.coeffs[m]

    save_dict = {'t': t_arr, 'om_coeffs': om_coeffs}
    with open('%s/%s.pkl' %(output_folder, sphere_list[i]), 'wb') as file:
        pkl.dump(save_dict, file, protocol=pkl.HIGHEST_PROTOCOL)

    print('\nSAVED!!!!\n')
