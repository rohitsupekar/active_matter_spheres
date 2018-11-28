"""
Higher-order Navier Stokes on the sphere.
"""

import numpy as np
from scipy.sparse import linalg as spla
import sphere_wrapper as sph
import equations as eq
import os
import dedalus.public as de
import time
import pathlib
from mpi4py import MPI

# Load config options
from dedalus.tools.config import config
STORE_LU = config['linear algebra'].getboolean('store_LU')
PERMC_SPEC = config['linear algebra']['permc_spec']
USE_UMFPACK = config['linear algebra'].getboolean('use_umfpack')

# Discretization parameters
L_max = 255  # spherical harmonic order
S_max = 4  # spin order (leave fixed)

# Physical parameters
gamma = 1  # surface mass density
Lmid = 50
e0 = 0.1  # friction term
e1 = 0.21 / Lmid**2  # forcing term
e2 = 0.1 / Lmid**4  # damping term
fspin = 0  # rotation
Amp = 1e-2  # initial noise amplitude

# Integration parameters
dt = 0.0004  # timestep
n_iterations = 10000  # total iterations
n_output = 10  # data output cadence
output_folder = 'output_files'  # data output folder

# Prevent running from dropbox
#path = pathlib.Path(__file__).resolve()
#if 'dropbox' in str(path).lower():
#    raise RuntimeError("It looks like you're running this script inside a dropbox folder. This has been disallowed to prevent spamming other shared-folder users.")

# Find MPI rank
comm = MPI.COMM_WORLD
rank = comm.rank

# Make domain
phi_basis   = de.Fourier('phi'  , 2*(L_max+1), interval=(0,2*np.pi))
theta_basis = de.Fourier('theta', L_max+1, interval=(0,np.pi))
domain = de.Domain([phi_basis,theta_basis], grid_dtype=np.float64)
phi = domain.grids(1)[0]

# set up fields
v_th = domain.new_field()
v_ph = domain.new_field()
p    = domain.new_field()
om   = domain.new_field()

# work arrays
v_thth = domain.new_field()
v_thph = domain.new_field()
v_phth = domain.new_field()
v_phph = domain.new_field()

F_th = domain.new_field()
F_ph = domain.new_field()

# set up sphere
p.require_coeff_space()
m_start = p.layout.start(1)[0]
m_len = p.layout.local_shape(1)[0]
m_end = m_start + m_len - 1
S = sph.Sphere(L_max,S_max,m_min=m_start,m_max=m_end)

# Calculate theta grid
theta_slice = domain.distributor.layouts[-1].slices(1)[1]
theta_len = domain.local_grid_shape(1)[1]
theta = S.grid[theta_slice].reshape([1,theta_len])

# Grid initial conditions

# Right now we specify initial conditions in terms of spectral coefficients
# If you want to specify initial conditions in terms of theta & phi
# do that here.

# Move initial conditions into coefficient space

# move data into (m,theta):
for f in [v_th,v_ph,p]:
        # Move data to (m, theta), dist over m
        f.require_layout(domain.distributor.layouts[1])

# move workspace into (m,theta):
for f in [v_thth,v_thph,v_phth,v_phph,F_th,F_ph]:
        # Setup temp fields to be in (m, theta), dist over m
        f.layout = domain.distributor.layouts[1]

# Transform theta -> ell, and put data into state vector
state_vector = []
for m in range(m_start,m_end+1):
    md = m - m_start
    v_c = S.forward(m,1,[v_th.data[md],v_ph.data[md]])
    p_c = S.forward(m,0,p.data[md])
    state_vector.append(eq.packup(v_c,p_c))

# Add random perturbations to the spectral coefficients
rand = np.random.RandomState(seed=42+rank)
for m in range(m_start,m_end+1):
    md = m - m_start
    v_c, p_c = eq.unpack(S,m,state_vector[md])
    shape = v_c.shape
    noise = rand.standard_normal(shape)
    phase = rand.uniform(0,2*np.pi,shape)
    v_c = Amp * noise*np.exp(1j*phase)
    state_vector[md] = eq.packup(v_c,p_c)

# allocating more work arrays
RHS, Dv_c = [], []
for m in range(m_start,m_end+1):
    md = m - m_start
    v_c = S.forward(m,1,[v_th.data[md],v_ph.data[md]])
    p_c = S.forward(m,0,p.data[md])
    RHS.append(eq.packup(v_c,p_c))
    Dv_c.append(S.forward(m,2,[p.data[md],p.data[md],p.data[md],p.data[md]]))

# build matrices
P,M,L = [],[],[]
for m in range(m_start,m_end+1):
    Mm,Lm = eq.advection(S,m,[gamma,e0,e1,e2,fspin])
    M.append(Mm.astype(np.complex128))
    L.append(Lm.astype(np.complex128))
    P.append(0.*Mm.astype(np.complex128))

# calculate RHS nonlinear terms from state_vector
def nonlinear(state_vector,RHS):

    for f in [v_th,v_ph,v_thth,v_thph,v_phth,v_phph]:
        # Setup temp fields to be in (m, theta), dist over m
        f.layout = domain.distributor.layouts[1]

    for m in range(m_start,m_end+1):
        md = m - m_start

        # get v and p in coefficient space
        v_c,p_c = eq.unpack(S,m,state_vector[md])

        # calculate grad v
        S.grad(m,1,v_c,Dv_c[md])

        # transform v and grad v into (m, theta)
        v_g    = S.backward(m,1,v_c)
        Dv_g   = S.backward(m,2,Dv_c[md])

        # unpack into theta and phi directions
        v_th.data[md] = v_g[0]
        v_ph.data[md] = v_g[1]
        v_thth.data[md]  = Dv_g[0]
        v_phth.data[md]  = Dv_g[1]
        v_thph.data[md]  = Dv_g[2]
        v_phph.data[md]  = Dv_g[3]

    # transform to grid space
    for f in [v_th,v_ph,v_thth,v_thph,v_phth,v_phph,F_th,F_ph]: f.require_grid_space()

    # calculate nonlinear terms
    F_th['g'] = -(v_th['g']*v_thth['g'] + v_ph['g']*v_thph['g'])
    F_ph['g'] = -(v_th['g']*v_phth['g'] + v_ph['g']*v_phph['g'])

    # move workspace into (m,theta):
    for f in [F_th,F_ph]:
        # Change fields to be in (m, theta), dist over m
        f.require_layout(domain.distributor.layouts[1])

    # move workspace into (ell,m)
    for m in range(m_start,m_end+1):
        md = m - m_start
        F_c = S.forward(m,1,[F_th.data[md],F_ph.data[md]])
        G_c = S.forward(m,0,0.*F_th.data[md])
        RHS[md] = eq.packup(F_c,G_c)

# Setup outputs
#file_num = 1
#if not os.path.exists(output_folder):
#    os.mkdir(output_folder)
t = 0

# Combine matrices and perform LU decompositions for constant timestep
for m in range(m_start,m_end+1):
    md = m - m_start
    Pmd = M[md] + dt*L[md]
    if STORE_LU:
        P[md] = spla.splu(Pmd.tocsc(), permc_spec=PERMC_SPEC)
    else:
        P[md] = Pmd

# Main loop
for i in range(n_iterations):

    nonlinear(state_vector,RHS)
    for m in range(m_start,m_end+1):
        md = m - m_start
        RHS[md] = M[md].dot(state_vector[md]) + dt*(RHS[md])
        if STORE_LU:
            state_vector[md] = P[md].solve(RHS[md])
        else:
            state_vector[md] = spla.spsolve(P[md],RHS[md])

    t += dt

    # imposing that the m=0 mode of v_th, v_ph, p are purely real
    if i % 100 == 1 and rank == 0:
        v_c,p_c = eq.unpack(S,0,state_vector[0])
        v_g = S.backward(0,1,v_c)
        v_g.imag = 0.
        (start_index,end_index,spin) = S.tensor_index(0,1)
        v_c = S.forward(0,1,v_g)
        p_c.imag = 0.

        state_vector[0] = eq.packup(v_c,p_c)

    if i % n_output == 0:

        # transform back to grid space for output
        for f in [v_th,v_ph,p,om]:
            # Setup temp fields to be in (m, theta), dist over m
            f.layout = domain.distributor.layouts[1]

        for m in range(m_start,m_end+1):
            md = m - m_start

            v_c,p_c = eq.unpack(S,m,state_vector[md])

            # calculating omega = km.up - kp.um
            (start_index,end_index,spin) = S.tensor_index(m,1)
            om_c = 1j*(S.op('k-',m,1).dot(v_c[start_index[0]:end_index[0]]) - S.op('k+',m,-1).dot(v_c[start_index[1]:end_index[1]]))

            # transform to m, theta
            v_g   = S.backward(m,1,v_c)
            p_g   = S.backward(m,0,p_c)
            om_g  = S.backward(m,0,om_c)

            v_th.data[md] = v_g[0]
            v_ph.data[md] = v_g[1]
            p.data[md]    = p_g
            om.data[md]   = om_g

        # go to grid space
        for f in [v_th,v_ph,p,om]: f.require_grid_space()

        # gather full data to output
        vph_global = comm.gather(v_ph['g'], root=0)
        vth_global = comm.gather(v_th['g'], root=0)
        p_global = comm.gather(p['g'], root=0)
        om_global = comm.gather(om['g'], root=0)

        if rank == 0:
            # Save data
            vph_global = np.hstack(vph_global)
            vth_global = np.hstack(vth_global)
            p_global = np.hstack(p_global)
            om_global = np.hstack(om_global)
#            np.savez(os.path.join(output_folder, 'output_%i.npz' %file_num),
#                     p=p_global, om=om_global, vph=vph_global, vth=vth_global,
#                     t=np.array([t]), phi=phi_basis.grid(1), theta=S.grid)
#            file_num += 1

            # Print iteration and maximum vorticity
            print('Iter:', i, 'Time:', t, 'om max:', np.max(np.abs(om_global)))

