
import glob
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from mpi4py import MPI
comm = MPI.COMM_WORLD
import cartopy.crs as ccrs
from dedalus.extras import plot_tools
import logging
logger = logging.getLogger(__name__)


# Parameters
first_frame = 1
#last_frame = 8
figsize = (6, 6)
dpi = 256
show_time = True
gridlines = True
coastlines = False
edgecolor = 'k'
proj = ccrs.PlateCarree(central_longitude=0)#, central_latitude=30)
proj = ccrs.Mollweide(central_longitude=0)
proj = ccrs.Orthographic(central_longitude=0, central_latitude=30)
#fields = ['p','om','vth','vph']
fields = ['v_ph', 'om']
input_folder = sys.argv[1]
output_folder = sys.argv[2]

#count files in the input folder
last_frame = len(glob.glob1("".join([input_folder,'/']),"*.npz"))

# Setup output folder
if comm.rank == 0:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
comm.barrier()

# Setup figure projection
fig = plt.figure(figsize=figsize)
axes = plt.axes((0, 0, 1, 1), projection=proj)

#set clims for all the fields
max_vals = {key: 0 for key in fields}
clims = {key: 0 for key in fields}
for field in fields:
    for i in range(first_frame + comm.rank, last_frame + 1, comm.size):
        with np.load("".join([input_folder, '/output_%i.npz' %i])) as file:
            fieldval = file[field]
            max_vals[field] = max(max_vals[field], np.max(fieldval))

for field in fields:
    clims[field] = 0.75*max_vals[field]

# Plot outputs
for i in range(first_frame + comm.rank, last_frame + 1, comm.size):
    # Load data
    logger.info('Frame: %i' %(i))

    with np.load("".join([input_folder, '/output_%i.npz' %i])) as file:
        if i == first_frame + comm.rank:
            phi = file['phi']
            theta = file['theta']
        time = file['t'][0]
        for field in fields:
            data = file[field]
            # Create plot
            if i == first_frame + comm.rank:
                lon = (phi + phi[1]/2 - np.pi) * 180 / np.pi
                lat = (np.pi/2 - theta) * 180 / np.pi
                xmesh, ymesh = plot_tools.quad_mesh(lon, lat)
                image = axes.pcolormesh(xmesh, ymesh, data.T, cmap='RdBu_r', transform=ccrs.PlateCarree())
                if show_time:
                    title = fig.suptitle('t = %.4f' %time)
                if gridlines:
                    axes.gridlines(xlocs=np.arange(0, 361, 30), ylocs=np.arange(-60, 61, 30), color='k')
                if coastlines:
                    axes.coastlines()
                axes.set_global()
                axes.outline_patch.set_edgecolor(edgecolor)
            # Update plot
            else:
                image.set_array(data.T.ravel())
                if show_time:
                    title.set_text('t = %.4f' %time)
            # Save
            if clims:
                image.set_clim(-clims[field], clims[field])
            else:
                clim_i = np.max(np.abs(data))
                image.set_clim(-clim_i, clim_i)
            plt.savefig(os.path.join(output_folder, '%s_%05i.png' %(field, i)), dpi=dpi)
