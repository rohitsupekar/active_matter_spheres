"""High-level class for spheres."""

import numpy as np
import dedalus.public as de
import sphere_wrapper


class SimpleSphere:

    def __init__(self, L_max, S_max):
        self.L_max = L_max
        self.S_max = S_max
        # Domain
        phi_basis = de.Fourier('phi', 2*(L_max+1), interval=(0,2*np.pi))
        theta_basis = de.Fourier('theta', L_max+1, interval=(0,np.pi))
        self.domain = de.Domain([phi_basis, theta_basis], grid_dtype=np.float64)
        # Local m
        layout0 = self.domain.distributor.layouts[0]
        self.m_start = layout0.start(1)[0]
        self.m_len = layout0.local_shape(1)[0]
        self.m_end = self.m_start + self.m_len - 1
        self.local_m = np.arange(self.m_start, self.m_end+1)
        # Sphere wrapper
        self.sphere_wrapper = sphere_wrapper.Sphere(L_max, S_max, m_min=self.m_start, m_max=self.m_end)
        # Grids
        self.phi_grid = self.domain.grid(0)
        self.global_theta_grid = self.sphere_wrapper.grid[None,:]
        theta_slice = self.domain.distributor.layouts[-1].slices(1)[1]
        self.local_theta_grid = self.global_theta_grid[:,theta_slice]


class TensorField:

    def __init__(self, simplesphere, rank):
        self.simplesphere = simplesphere
        self.rank = rank
        self.ncomp = 2**rank
        self.component_fields = [simplesphere.domain.new_field() for i in range(self.ncomp)]
        self.coeffs = [None for m in simplesphere.local_m]
        self._layout1 = simplesphere.domain.distributor.layouts[1]
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
        SW = self.simplesphere.sphere_wrapper
        for dm, m in enumerate(self.simplesphere.local_m):
            m_data = [f.data[dm] for f in self.component_fields]
            # Unpack for rank 0 to counteract shortcut bug in sphere_wrapper
            if self.rank == 0:
                m_data, = m_data
            self.coeffs[dm] = SW.forward(m, self.rank, m_data)

    def backward_theta(self):
        """Transform from (m, ell) to (m, theta)."""
        SW = self.simplesphere.sphere_wrapper
        self.set_layout(self._layout1)
        for dm, m in enumerate(self.simplesphere.local_m):
            m_data = SW.backward(m, self.rank, self.coeffs[dm])
            if self.rank == 0:
                m_data = [m_data]
            for i, f in enumerate(self.component_fields):
                f.data[dm] = m_data[i]


class TensorSystem:

    def __init__(self, simplesphere, tensors):
        self.simplesphere = simplesphere
        self.tensors = tensors
        self.coeffs = [None for m in simplesphere.local_m]
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
        for dm, m in enumerate(self.simplesphere.local_m):
            m_coeffs = [t.coeffs[dm] for t in self.tensors]
            self.coeffs[dm] = np.hstack(m_coeffs)

    def unpack_coeffs(self):
        """Unpack system vectors into tensor coefficients."""
        for dm, m in enumerate(self.simplesphere.local_m):
            i0 = 0
            for t in self.tensors:
                i1 = i0 + len(t.coeffs[dm])
                t.coeffs[dm] = self.coeffs[dm][i0:i1]
                i0 = i1

