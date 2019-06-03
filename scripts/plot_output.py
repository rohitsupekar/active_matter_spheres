import os
import sys
import glob
import time
import pathlib
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


# Parameters
first_frame = 1
#last_frame = 8
figsize = (3, 3)
dpi = 300
show_time = True
gridlines = True
coastlines = False
edgecolor = 'k'
proj = ccrs.PlateCarree(central_longitude=0)#, central_latitude=30)
proj = ccrs.Mollweide(central_longitude=0)
proj = ccrs.Orthographic(central_longitude=0, central_latitude=30)
#fields = ['p','om','vth','vph']
fields = ['v_ph', 'om', 'v_th']
sim_number = int(sys.argv[1])
input_folder = sys.argv[2]
output_folder = sys.argv[3]
FPS = int(sys.argv[4])

#count files in the input folder
last_frame = len(glob.glob1("".join([input_folder,'/']),"*.npz"))
logger.info('Total number of frames: %i' %(last_frame))

#set clims for all the fields
max_vals = {key: 0 for key in fields}
clims = {key: 0 for key in fields}
logger.info('Find max values..')
for i in range(first_frame + comm.rank+1, last_frame + 1, comm.size):
    with np.load("".join([input_folder, '/output_%i.npz' %i])) as file:
        for field in fields:
            fieldval = file[field]
            max_vals[field] = max(max_vals[field], np.max(fieldval))

for field in fields:
    clims[field] = 0.75*max_vals[field]

metadata = dict(title='Movie', artist='Matplotlib', comment='Movie support!')
writer = FFMpegWriter(fps=FPS, metadata=metadata)

for field in fields:

    fig = plt.figure(figsize=figsize)
    axes = plt.axes((0.1, 0.1, 0.8, 0.8), projection=proj)

    with writer.saving(fig, "%s/sphere%i_%s.mp4" %(output_folder, sim_number, field), dpi):
        for i in range(first_frame + comm.rank+1, last_frame + 1, comm.size):

            if i%10==0: logger.info('Frame: %i' %(i))

            with np.load("".join([input_folder, '/output_%i.npz' %i])) as file:
                if i == first_frame + comm.rank+1:
                    phi = file['phi']
                    theta = file['theta']
                time = file['t'][0]
                thth, phiphi = np.meshgrid(theta, phi)

                data = file[field]

                # Create plot
                if i == first_frame + comm.rank + 1:
                    lon = (phi + phi[1]/2 - np.pi) * 180 / np.pi
                    lat = (np.pi/2 - theta) * 180 / np.pi
                    xmesh, ymesh = plot_tools.quad_mesh(lon, lat)
                    image = axes.pcolormesh(xmesh, ymesh, data.T, cmap='RdBu_r', transform=ccrs.PlateCarree())
                    if show_time:
                        title = fig.suptitle('t = %.4f' %time)
                    if gridlines:
                        axes.gridlines(xlocs=np.arange(0, 361, 30), ylocs=np.arange(-60, 61, 30), color='k')

                    axes.set_global()
                    axes.outline_patch.set_edgecolor(edgecolor)
                    fig.colorbar(image, ax=axes)

                # Update plot
                else:
                    image.set_array(data.T.ravel())
                    if show_time:
                        title.set_text('t = %.4f' %time)


                if clims:
                    image.set_clim(-clims[field], clims[field])
                else:
                    clim_i = np.max(np.abs(data))
                    image.set_clim(-clim_i, clim_i)

            writer.grab_frame()
