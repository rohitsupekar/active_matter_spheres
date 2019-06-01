"""Simulation script."""

import sys
import os
import time
import pathlib
import logging
import numpy as np
from mpi4py import MPI
from scipy.sparse import linalg as spla
from dedalus.tools.config import config
from simple_sphere import SimpleSphere
import equations

# Logging and config
logger = logging.getLogger(__name__)
STORE_LU = config['linear algebra'].getboolean('store_LU')
PERMC_SPEC = config['linear algebra']['permc_spec']
USE_UMFPACK = config['linear algebra'].getboolean('use_umfpack')

# Discretization parameters
L_max = 127  # Spherical harmonic order
S_max = 4  # Spin order (leave fixed)

# Model parameters
Lmid = float(sys.argv[1])   #gives 1/10 as characteristic diameter for the vortices
kappa = float(sys.argv[2])  #spectral injection bandwidth
fspin = float(sys.argv[3])  #rotation
gamma = 1  # surface mass density
isInertialRot = True #set to True for an initial rotation of f_inertial/2 in the inertial frame
f_inertial = 50

logger.info('Simulation params: Lmid = %.3f, kappa = %.3f, f = %.3f' %(Lmid, kappa, fspin))

### calculates e0, e1, e2 from Lmid and kappa
a = 0.25*(Lmid**2*kappa**2 - 0.5*(2*np.pi*Lmid+1)**2)**2 + 17*17/16 - (34/16)*(2*np.pi*Lmid+1)**2
b = (17/4 - 0.25*(2*np.pi*Lmid+1)**2)**2
c = 1/(17/4 - 0.25*(2*np.pi*Lmid + 1)**2 - 2)
e0 = a*c/(a-b)
e1 = 2*np.sqrt(b)*c/(a-b)
e2 = c/(a-b)

params = [gamma, e0, e1, e2, fspin]

# Integration parameters
Amp = 1e-2  # initial noise amplitude
#factor = 0.5   #controls the time step below to be 0.5/(100), which is 0.5/100 of characteristic vortex dynamics time
factor = float(sys.argv[4])
dt = factor/(100)
n_iterations = int(2000/factor)# total iterations. Change 10000 to higher number for longer run!
n_output = int(5/factor)  # data output cadence
n_clean = 10
output_folder = sys.argv[5]  # data output folder

# Prevent running from dropbox
path = pathlib.Path(__file__).resolve()
if 'dropbox' in str(path).lower():
    raise RuntimeError("It looks like you're running this script inside a dropbox folder. This has been disallowed to prevent spamming other shared-folder users.")

# Find MPI rank
comm = MPI.COMM_WORLD
rank = comm.rank

# Domain
start_init_time = time.time()
simplesphere = SimpleSphere(L_max, S_max)
domain = simplesphere.domain

# Model
model = equations.ActiveMatterModel(simplesphere, params)
state_system = model.state_system

# Matrices
# Combine matrices and perform LU decompositions for constant timestep
A = []
for dm, m in enumerate(simplesphere.local_m):
    # Backward Euler for LHS
    Am = model.M[dm] + dt*model.L[dm]
    if STORE_LU:
        Am = spla.splu(Am.tocsc(), permc_spec=PERMC_SPEC)
    A.append(Am)

phi_flat = simplesphere.phi_grid.ravel()
theta_flat = simplesphere.global_theta_grid.ravel()

# Initial conditions
v = model.v

if isInertialRot==True:
    theta, phi = np.meshgrid(theta_flat, phi_flat)
    #set v_phi to Omega*sin(theta)
    v.component_fields[1]['g'] = (f_inertial/2)*np.sin(theta)
    v.forward_phi()
    v.forward_theta()

# Add random perturbations to the velocity coefficients
seed0 = np.random.randint(0, 1000)
logger.info("seed0 = %i" %(seed0))
rand = np.random.RandomState(seed=seed0+rank)
for dm, m in enumerate(simplesphere.local_m):
    shape = v.coeffs[dm].shape
    noise = rand.standard_normal(shape)
    phase = rand.uniform(0,2*np.pi,shape)
    v.coeffs[dm] += Amp * noise*np.exp(1j*phase)

state_system.pack_coeffs()

# Setup outputs
file_num = 1
if not os.path.exists(output_folder):
    os.mkdir(output_folder)


# Main loop
end_init_time = time.time()
logger.info('Initialization time: %f' %(end_init_time-start_init_time))
logger.info('Starting loop')
start_run_time = time.time()

t = 0
RHS = model.RHS_system.coeffs
X = model.state_system.coeffs
for i in range(n_iterations):

    # Timestep
    model.compute_RHS()
    for dm, m in enumerate(simplesphere.local_m):
        # Forward Euler for RHS
        RHS[dm] = model.M[dm].dot(X[dm]) + dt*(RHS[dm])
        if STORE_LU:
            X[dm] = A[dm].solve(RHS[dm])
        else:
            X[dm] = spla.spsolve(A[dm], RHS[dm])
    t += dt

    # Imagination cleaning
    if i % n_clean == 0:
        # Zero imaginary part by performing full transform loop
        state_system.unpack_coeffs()
        state_system.backward_theta()
        state_system.backward_phi()
        state_system.forward_phi()
        state_system.forward_theta()
        state_system.pack_coeffs()

    # Output
    if i % n_output == 0:
        # Gather full data to output
        model.compute_analysis()
        output = {}
        for task in model.analysis:
            output[task] = comm.gather(model.analysis[task], root=0)
        # Save data
        if rank == 0:
            for task in output:
                output[task] = np.hstack(output[task])
            np.savez(os.path.join(output_folder, 'output_%i.npz' %file_num),
                     t=np.array([t]), phi=phi_flat, theta=theta_flat, **output)
            logger.info('Iter: %i, Time: %f, KE max: %f' %(i, t, np.max(np.abs(output['KE']))))
        file_num += 1

end_run_time = time.time()
logger.info('Iterations: %i' %(i+1))
logger.info('Sim end time: %f' %t)
logger.info('Run time: %.2f sec' %(end_run_time-start_run_time))
logger.info('Run time: %f cpu-hr' %((end_run_time-start_run_time)/60/60*domain.dist.comm_cart.size))
