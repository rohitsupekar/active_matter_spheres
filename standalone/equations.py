#change made
import logging
import numpy as np
import scipy.sparse as sparse
from simple_sphere import TensorField, TensorSystem

logger = logging.getLogger(__name__)


class ActiveMatterModel:
    """
    Model for active matter equations on the sphere.

    Parameters:
        g: (gamma) surface mass density
        e0: friction term
        e1: forcing laplacian term
        e2: small scale damping term
        f: 2 Omega_z, rotation

    Equations:
        g*dt(up) + kp*p - 2*km*eps*kp*up + i*f*Cp*up = - (u.grad u)_p
        g*dt(um) + km*p - 2*kp*eps*km*um - i*f*Cm*um = - (u.grad u)_m
        kp*um + km*up = 0

    where
        eps = e0 + e1*(kp*km + km*kp) + e2*(kp*km + km*kp)*(kp*km + km*kp)

    Variable order: up, um, p

    """

    def __init__(self, simplesphere, params):
        self.simplesphere = simplesphere
        self.params = params
        # Problem fields
        self.v = v = TensorField(simplesphere, rank=1)
        self.p = p = TensorField(simplesphere, rank=0)
        self.state_system = TensorSystem(simplesphere, [self.v, self.p])
        # Work fields
        self.grad_v = TensorField(simplesphere, rank=2)
        self.Fv = TensorField(simplesphere, rank=1)
        self.Fp = TensorField(simplesphere, rank=0)
        self.RHS_system = TensorSystem(simplesphere, [self.Fv, self.Fp])
        # Analysis fields
        self.analysis = {}
        self.om = TensorField(simplesphere, rank=0)
        self.KE = TensorField(simplesphere, rank=0)
        # Build matrices
        self.M, self.L = [], []
        for m in simplesphere.local_m:
            logger.info("Building matrix %i" %m)
            Mm, Lm = self.build_matrices(m)
            self.M.append(Mm)
            self.L.append(Lm)

    def compute_RHS(self):
        """Calculate RHS terms from state vector."""
        # Local references
        simplesphere = self.simplesphere
        v = self.v
        grad_v = self.grad_v
        Fv = self.Fv
        v_th, v_ph = v.component_fields
        v_thth, v_phth, v_thph, v_phph = grad_v.component_fields
        Fv_th, Fv_ph = Fv.component_fields
        # Unpack state system
        self.state_system.unpack_coeffs()
        # Calculate grad v
        for dm, m in enumerate(simplesphere.local_m):
            simplesphere.sphere_wrapper.grad(m, 1, v.coeffs[dm], grad_v.coeffs[dm])
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
        self.RHS_system.pack_coeffs()

    def build_matrices(self, m):
        """Build LHS matrices."""

        S = self.simplesphere.sphere_wrapper
        g, e0, e1, e2, f = self.params

        M00 = (g)*S.op('I',m,1)
        M01 = S.zeros(m,1,-1)
        M02 = S.zeros(m,1,0)

        M10 = S.zeros(m,-1,1)
        M11 = (g)*S.op('I',m,-1)
        M12 = S.zeros(m,-1,0)

        M20 = S.zeros(m,0,1)
        M21 = S.zeros(m,0,-1)
        M22 = S.zeros(m,0,0)

        M = sparse.bmat([[M00, M01, M02],
                         [M10, M11, M12],
                         [M20, M21, M22]])

        # (+,+)
        L00 = -2*S.op('k-',m,2).dot(
                    ( e0*S.op('I',m,2) + e1*( S.op('k+',m,1).dot(S.op('k-',m,2))
                                            +S.op('k-',m,3).dot(S.op('k+',m,2)))
                                    + e2*( S.op('k+',m,1).dot(S.op('k-',m,2))
                                            +S.op('k-',m,3).dot(S.op('k+',m,2))).dot
                                            ( S.op('k+',m,1).dot(S.op('k-',m,2))
                                            +S.op('k-',m,3).dot(S.op('k+',m,2)))
                                                                                ).dot(S.op('k+',m,1)) )
        L00 += 1j*f*S.op('C',m,1)

        # (+,-)
        L01 = S.zeros(m,1,-1)

        # (+,p)
        L02 = S.op('k+',m,0)

        # (-,+)
        L10 = S.zeros(m,-1,1)

        # (-,-)
        L11 = -2*S.op('k+',m,-2).dot(
                    ( e0*S.op('I',m,-2) + e1*( S.op('k+',m,-3).dot(S.op('k-',m,-2))
                                            +S.op('k-',m,-1).dot(S.op('k+',m,-2)))
                                        + e2*( S.op('k+',m,-3).dot(S.op('k-',m,-2))
                                            +S.op('k-',m,-1).dot(S.op('k+',m,-2))).dot
                                            ( S.op('k+',m,-3).dot(S.op('k-',m,-2))
                                            +S.op('k-',m,-1).dot(S.op('k+',m,-2)))
                                                                                ).dot(S.op('k-',m,-1)) )
        L11 += -1j*f*S.op('C',m,-1)

        # (-,p)
        L12 = S.op('k-',m,0)

        # (p,+)
        L20 = S.op('k-',m,1)

        # (p,-)
        L21 = S.op('k+',m,-1)

        # (p,p)
        L22 = S.zeros(m,0,0)
        if m == 0:
            L22[0,0] = 1.

        L = sparse.bmat([[L00, L01, L02],
                        [L10, L11, L12],
                        [L20, L21, L22]])

        return M.astype(np.complex128), L.astype(np.complex128)

    def compute_analysis(self):
        """Compute analysis fields."""
        v = self.v
        p = self.p
        om = self.om
        KE = self.KE
        S = self.simplesphere.sphere_wrapper
        # Unpack state system
        self.state_system.unpack_coeffs()
        # Compute vorticity
        for dm, m in enumerate(self.simplesphere.local_m):
            v_c = v.coeffs[dm]
            # omega = km.up - kp.um
            start_index, end_index, spin = S.tensor_index(m, 1)
            om.coeffs[dm] = 1j*(S.op('k-',m,1).dot(v_c[start_index[0]:end_index[0]]) - S.op('k+',m,-1).dot(v_c[start_index[1]:end_index[1]]))
        # Compute kinetic energy
        for dm, m in enumerate(self.simplesphere.local_m):
            v_c = v.coeffs[dm]
            # omega = km.up - kp.um
            start_index, end_index, spin = S.tensor_index(m, 1)
            KE.coeffs[dm] = 1j*(S.op('k-',m,1).dot(v_c[start_index[0]:end_index[0]]) - S.op('k+',m,-1).dot(v_c[start_index[1]:end_index[1]]))
        # Transform output to grid
        for tensor in [v, p, om, KE]:
            tensor.backward_theta()
            tensor.backward_phi()
        # Copy component fields
        v_th, v_ph = v.component_fields
        p0, = p.component_fields
        om0, = om.component_fields
        KE0, = KE.component_fields
        self.analysis['v_th'] = v_th['g'].copy()
        self.analysis['v_ph'] = v_ph['g'].copy()
        self.analysis['p'] = p0['g'].copy()
        self.analysis['om'] = om0['g'].copy()
        self.analysis['KE'] = KE0['g'].copy()

