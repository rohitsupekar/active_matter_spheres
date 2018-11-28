

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from mpi4py import MPI
comm = MPI.COMM_WORLD
import cartopy.crs as ccrs
from dedalus.extras import plot_tools


# Parameters
first_frame = 1
last_frame = 20
figsize = (8, 4)
dpi = 256
show_time = False
gridlines = False
coastlines = False
edgecolor = 'none'
proj = ccrs.PlateCarree(central_longitude=0)#, central_latitude=30)
proj = ccrs.Mollweide(central_longitude=0)
fields = ['p','om','vth','vph']
output_folder = 'images_pc'
clim = None

# Setup output folder
if comm.rank == 0:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
comm.barrier()

# Setup figure projection
fig = plt.figure(figsize=figsize)
axes = plt.axes((0, 0, 1, 1), projection=proj)

# Plot outputs
for i in range(first_frame + comm.rank, last_frame + 1, comm.size):
    # Load data
    with np.load('output_files/output_%i.npz' %i) as file:
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
            if clim:
                image.set_clim(-clim, clim)
            else:
                clim_i = np.max(np.abs(data))
                image.set_clim(-clim_i, clim_i)
            plt.savefig(os.path.join(output_folder, '%s_%05i.png' %(field, i)), dpi=dpi)

