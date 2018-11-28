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
import logging
logger = logging.getLogger(__name__)

# Load config options
from dedalus.tools.config import config
STORE_LU = config['linear algebra'].getboolean('store_LU')
PERMC_SPEC = config['linear algebra']['permc_spec']
USE_UMFPACK = config['linear algebra'].getboolean('use_umfpack')

# Discretization parameters
L_max = 127  # spherical harmonic order
S_max = 4  # spin order (leave fixed)

Lmid = 10.0    #gives 1/10 as characteristic diameter for the vortices
kappa = 1.5    #spectral injection bandwidth
factor = 0.5   #controls the time step below to be 0.5/(100*Lmid^2), which is 0.5/100 of characteristic vortex dynamics time

# Physical parameters
gamma = 1  # surface mass density
fspin = 0

### calculates e0, e1, e2 from Lmid and kappa
a = 0.25*( 4*kappa**2*Lmid**2 - 2*(2*np.pi*Lmid + 1)**2 )**2 - 34*(2*np.pi*Lmid + 1)**2 + 17**2
b = ( 17/4 - 0.25*(2*np.pi*Lmid+1)**2 )**2
c = Lmid**2/(  (17/4 - 0.25*(2*np.pi*Lmid + 1)**2 -2/(Lmid**2) ) )
e0 = a*c/(a-16*b)
e2 = c/(a/16 - b)
e1 = - 2*(17/4 - 0.25*(2*np.pi*Lmid+1)**2 )*e2
Amp = 1e-2  # initial noise amplitude

# Integration parameters
dt = factor/(100*Lmid**2)

n_iterations = int(100/factor)# total iterations. Change 10000 to higher number for longer run!
n_output = int(10/factor)  # data output cadence
output_folder = 'output_files'  # data output folder

# Prevent running from dropbox
path = pathlib.Path(__file__).resolve()
if 'dropbox' in str(path).lower():
    raise RuntimeError("It looks like you're running this script inside a dropbox folder. This has been disallowed to prevent spamming other shared-folder users.")

# Find MPI rank
comm = MPI.COMM_WORLD
rank = comm.rank

# Domain
start_init_time = time.time()
phi_basis = de.Fourier('phi', 2*(L_max+1), interval=(0,2*np.pi))
theta_basis = de.Fourier('theta', L_max+1, interval=(0,np.pi))
domain = de.Domain([phi_basis, theta_basis], grid_dtype=np.float64)

# Sphere
layout0 = domain.distributor.layouts[0]
m_start = layout0.start(1)[0]
m_len = layout0.local_shape(1)[0]
m_end = m_start + m_len - 1
S = sph.Sphere(L_max, S_max, m_min=m_start, m_max=m_end)

# Grids
phi = domain.grids(1)[0]
theta_slice = domain.distributor.layouts[-1].slices(1)[1]
theta_len = domain.local_grid_shape(1)[1]
theta = S.grid[theta_slice].reshape([1,theta_len])


class TensorField:

    def __init__(self, domain, rank):
        self.domain = domain
        self.rank = rank
        self.ncomp = 2**rank
        self.component_fields = [domain.new_field() for i in range(self.ncomp)]
        self.coeffs = [None for i in range(m_len)]
        self._layout1 = domain.distributor.layouts[1]
        # Forward transform to initialize coeffs
        self.forward_phi()
        self.forward_theta()

    def set_layout(self, layout):
        for f in self.component_fields:
            f.layout = layout

    def forward_phi(self):
        """Transform field from (phi, theta) to (m, theta)."""
        for f in self.component_fields:
            f.require_layout(self._layout1)

    def backward_phi(self):
        """Transform from (m, theta) to (phi, theta)."""
        for f in self.component_fields:
            f.require_grid_space()

    def forward_theta(self):
        """Transform from (m, theta) to (m, ell)."""
        for m in range(m_start, m_end+1):
            dm = m - m_start
            m_data = [f.data[dm] for f in self.component_fields]
            # Unpack for rank 0 to counteract shortcut bug in sphere_wrapper
            if self.rank == 0:
                m_data, = m_data
            self.coeffs[dm] = S.forward(m, self.rank, m_data)

    def backward_theta(self):
        """Transform from (m, ell) to (m, theta)."""
        self.set_layout(self._layout1)
        for m in range(m_start, m_end+1):
            dm = m - m_start
            m_data = S.backward(m, self.rank, self.coeffs[dm])
            if self.rank == 0:
                m_data = [m_data]
            for i, f in enumerate(self.component_fields):
                f.data[dm] = m_data[i]


class TensorSystem:

    def __init__(self, tensors):
        self.tensors = tensors
        self.coeffs = [None for i in range(m_len)]
        # Pack to initialize data
        self.pack_coeffs()

    def forward_phi(self):
        for t in self.tensors:
            t.forward_phi()

    def backward_phi(self):
        for t in self.tensors:
            t.backward_phi()

    def forward_theta(self):
        for t in self.tensors:
            t.forward_theta()

    def backward_theta(self):
        for t in self.tensors:
            t.backward_theta()

    def pack_coeffs(self):
        """Pack tensor coefficients into system vectors."""
        for m in range(m_start, m_end+1):
            dm = m - m_start
            m_coeffs = [t.coeffs[dm] for t in self.tensors]
            self.coeffs[dm] = np.hstack(m_coeffs)

    def unpack_coeffs(self):
        """Unpack system vectors into tensor coefficients."""
        for m in range(m_start, m_end+1):
            dm = m - m_start
            i0 = 0
            for t in self.tensors:
                i1 = i0 + len(t.coeffs[dm])
                t.coeffs[dm] = self.coeffs[dm][i0:i1]
                i0 = i1

# Problem fields
v = TensorField(domain, rank=1)
p = TensorField(domain, rank=0)
state_system = TensorSystem([v, p])

# Work fields
om = TensorField(domain, rank=0)
grad_v = TensorField(domain, rank=2)
Fv = TensorField(domain, rank=1)
Fp = TensorField(domain, rank=0)
RHS_system = TensorSystem([Fv, Fp])

# Unpack components
v_th, v_ph = v.component_fields
p0, = p.component_fields
om0, = om.component_fields
v_thth, v_phth, v_thph, v_phph = grad_v.component_fields
Fv_th, Fv_ph = Fv.component_fields

# Add random perturbations to the velocity coefficients
v.forward_phi()
v.forward_theta()
rand = np.random.RandomState(seed=42+rank)
for m in range(m_start, m_end+1):
    dm = m - m_start
    shape = v.coeffs[dm].shape
    noise = rand.standard_normal(shape)
    phase = rand.uniform(0,2*np.pi,shape)
    v.coeffs[dm] = Amp * noise*np.exp(1j*phase)

# Build matrices
P, M, L = [], [], []
for m in range(m_start, m_end+1):
    logger.info("Building matrix %i in %i-%i" %(m, m_start, m_end))
    Mm, Lm = eq.advection(S, m, [gamma,e0,e1,e2,fspin])
    M.append(Mm.astype(np.complex128))
    L.append(Lm.astype(np.complex128))
    P.append(0.*Mm.astype(np.complex128))


def compute_RHS(state_system, RHS_system):
    """Calculate RHS terms from state vector."""

    # Unpack state system
    state_system.unpack_coeffs()

    # Calculate grad v
    for m in range(m_start, m_end+1):
        dm = m - m_start
        S.grad(m, 1, v.coeffs[dm], grad_v.coeffs[dm])

    # Transform to grid
    v.backward_theta()
    v.backward_phi()
    grad_v.backward_theta()
    grad_v.backward_phi()

    # Calculate nonlinear terms
    Fv_th['g'] = -(v_th['g']*v_thth['g'] + v_ph['g']*v_thph['g'])
    Fv_ph['g'] = -(v_th['g']*v_phth['g'] + v_ph['g']*v_phph['g'])

    # Forward transform F
    Fv.forward_phi()
    Fv.forward_theta()

    # Pack RHS system
    RHS_system.pack_coeffs()

# Setup outputs
file_num = 1
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
t = 0

# Combine matrices and perform LU decompositions for constant timestep
for m in range(m_start, m_end+1):
    dm = m - m_start
    Pdm = M[dm] + dt*L[dm]
    if STORE_LU:
        P[dm] = spla.splu(Pdm.tocsc(), permc_spec=PERMC_SPEC)
    else:
        P[dm] = Pdm

# Main loop
end_init_time = time.time()
logger.info('Initialization time: %f' %(end_init_time-start_init_time))
logger.info('Starting loop')
start_run_time = time.time()

state_system.pack_coeffs()
for i in range(n_iterations):

    # Compute RHS
    compute_RHS(state_system, RHS_system)
    RHS = RHS_system.coeffs
    X = state_system.coeffs
    # Solve LHS
    for m in range(m_start, m_end+1):
        dm = m - m_start
        RHS[dm] = M[dm].dot(X[dm]) + dt*(RHS[dm])
        if STORE_LU:
            X[dm] = P[dm].solve(RHS[dm])
        else:
            X[dm] = spla.spsolve(P[dm],RHS[dm])
    t += dt

    # Impose that m=0 mode of state fields are purely real
    if i % 10 == 1 and rank == 0:
        state_system.unpack_coeffs()
        state_system.backward_theta()
        for tensor in state_system.tensors:
            for field in tensor.component_fields:
                field.data[0].imag = 0
        state_system.forward_theta()
        state_system.pack_coeffs()

    if i % n_output == 0:
        # Unpack state system
        state_system.unpack_coeffs()
        # Compute vorticity
        for m in range(m_start, m_end+1):
            dm = m - m_start
            v_c = v.coeffs[dm]
            # calculating omega = km.up - kp.um
            start_index, end_index, spin = S.tensor_index(m, 1)
            om.coeffs[dm] = 1j*(S.op('k-',m,1).dot(v_c[start_index[0]:end_index[0]]) - S.op('k+',m,-1).dot(v_c[start_index[1]:end_index[1]]))
        # Transform to grid
        for f in [v, p, om]:
            f.backward_theta()
            f.backward_phi()
        # Gather full data to output
        vph_global = comm.gather(v_ph['g'], root=0)
        vth_global = comm.gather(v_th['g'], root=0)
        p_global = comm.gather(p0['g'], root=0)
        om_global = comm.gather(om0['g'], root=0)

        if rank == 0:
            # Save data
            vph_global = np.hstack(vph_global)
            vth_global = np.hstack(vth_global)
            p_global = np.hstack(p_global)
            om_global = np.hstack(om_global)
            np.savez(os.path.join(output_folder, 'output_%i.npz' %file_num),
                     p=p_global, om=om_global, vph=vph_global, vth=vth_global,
                     t=np.array([t]), phi=phi_basis.grid(1), theta=S.grid)
            file_num += 1

            # Print iteration and maximum vorticity
            logger.info('Iter: %i, Time: %f, om max: %f' %(i, t, np.max(np.abs(om_global))))

end_run_time = time.time()
logger.info('Iterations: %i' %(i+1))
logger.info('Sim end time: %f' %t)
logger.info('Run time: %.2f sec' %(end_run_time-start_run_time))
logger.info('Run time: %f cpu-hr' %((end_run_time-start_run_time)/60/60*domain.dist.comm_cart.size))

