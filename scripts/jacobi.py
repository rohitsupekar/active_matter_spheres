import numpy             as np
import scipy             as sp
import scipy.sparse      as sparse
import scipy.linalg      as linear
import scipy.special     as fun
import matplotlib.pyplot as plt


dtype = np.float64


def sparse_symm_to_banded(matrix):
    """Convert sparse symmetric to upper-banded form."""
    diag = matrix.todia()
    B = max(abs(diag.offsets))
    I = diag.data.shape[1]
    banded = np.zeros((B+1, I), dtype=diag.dtype)
    for i, b in enumerate(diag.offsets):
        if b >= 0:
            banded[B-b] = diag.data[i]
    return banded


class OPBasis:
    """For manuplating orthogonal polynomials"""

    def grid_guess(self):
        J = self.op['J']
        if self.symmetric:
            J_banded = sparse_symm_to_banded(J)
            self.grid = np.sort(linear.eigvals_banded(J_banded).real)
        else:
            J_dense = J.todense()
            self.grid = np.sort(linear.eigvals(J_dense).real)
        self.ngrid = len(self.grid)

    def three_term_recursion(self,init='flat',remainder=False):

        J, z, N = sparse.dia_matrix(self.op['J']).data, self.grid, self.max_degree+1

        z = z.astype(dtype)
        self.poly = np.zeros((N,self.ngrid), dtype=dtype)

        if init == 'flat':
            self.poly[0] = np.ones(self.ngrid)/np.sqrt(self.consts['mu'])
        else:
            self.poly[0] = init

        if np.shape(J)[0] == 3:
            for n in range(1,N):
                self.poly[n] = ((z-J[1][n-1])*self.poly[n-1] - J[0][n-2]*self.poly[n-2])/J[-1][n]
            if remainder:
                return ((z-J[1][N-1])*self.poly[N-1] - J[0][N-2]*self.poly[N-2])
        else:
            for n in range(1,N):
                self.poly[n] = (z*self.poly[n-1] - J[0][n-2]*self.poly[n-2])/J[-1][n]
            if remainder:
                return (z*self.poly[N-1] - J[0][N-2]*self.poly[N-2])

    def three_term_deriv(self,init_deriv='flat',remainder=False):

        J, z, N = sparse.dia_matrix(self.op['J']).data, self.grid, self.max_degree+1
        
        z = z.astype(dtype)
        deriv = np.zeros((N,self.ngrid), dtype=dtype)

        if init_deriv != 'flat': deriv[0] = init_deriv
       
        if np.shape(J)[0] == 3:
            for n in range(1,N):
                deriv[n] = ((z-J[1][n-1])*deriv[n-1] - J[0][n-2]*deriv[n-2])/J[-1][n] + self.poly[n-1]/J[-1][n]
            if remainder:
                return (z-J[1][N-1])*deriv[N-1] - J[0][N-2]*deriv[N-2] + self.poly[N-1]
            else:
                    return deriv
        else:
            for n in range(1,N):
                deriv[n] = (z*deriv[n-1] - J[0][n-2]*deriv[n-2])/J[-1][n] + self.poly[n-1]/J[-1][n]
            if remainder:
                return z*deriv[N-1] - J[0][N-2]*deriv[N-2] + self.poly[N-1]
            else:
                return deriv

    def compute_weights(self):
        self.weights = 0
        for n in range(self.max_degree+1):
            self.weights = self.weights + self.poly[n]**2
        self.weights = 1/self.weights

    def inner_product(self,k1,k2):
        return np.sum(self.poly[k1]*self.poly[k2]*self.weights)

    def polish_norm(self):
        for k in range(self.max_degree+1):
            self.poly[k] = self.poly[k]/np.sqrt(self.inner_product(k,k))

    def make_grid(self):
        self.grid_guess()
        remainder = self.three_term_recursion(remainder=True)
        self.grid = self.grid - remainder/self.three_term_deriv(remainder=True)
        self.three_term_recursion()
        self.compute_weights()
        self.polish_norm()

    def transform_error(self):
        err=0
        for j in range(self.max_degree+1):
            for k in range(self.max_degree+1):
                if j == k :
                    i = 1
                else:
                    i = 0
        err = np.max((np.abs(self.inner_product(j,k)-i),err))
        print(err)

    def show(self,k):
        plt.plot(np.arccos(self.grid),self.poly[k])


class Jacobi(OPBasis):
    """Defines basic aspects of Jacobi polynomials. Requires a,b > -1."""

    def __init__(self,max_degree,a=-1/2,b=-1/2,symmetric=True,grid_basis=None,envlope=True):
        self.max_degree  = max_degree
        self.consts      = {'a':a,'b':b}
        self.symmetric   = symmetric
        self.op          = {'I':sparse.identity(max_degree+1)}

        self.consts['mu'] = np.exp( (a+b+1)*np.log(2) + fun.gammaln(a+1) + fun.gammaln(b+1) - fun.gammaln(a+b+2) )
        self.make_ops()
        if grid_basis == None:
            self.make_grid()
        else:
            self.ngrid, self.grid, self.weights = grid_basis.ngrid, grid_basis.grid, grid_basis.weights
            a0, b0 = grid_basis.consts['a'], grid_basis.consts['b']
            if envlope:
                init = np.exp( ((a-a0)/2)*np.log(1-self.grid) + ((b-b0)/2)*np.log(1+self.grid) )/np.sqrt(self.consts['mu'])
                self.three_term_recursion(init=init)
                self.polish_norm()

        self.convert_ops()

    def push(self,op,data):
        return self.op[op].dot(data)

    def pull(self,op,data):
        return (self.op[op].transpose()).dot(data)

    def make_ops(self):

        def diag(bands,locs):
            return sparse.dia_matrix((bands,locs),shape=(len(bands[0]),len(bands[0])))

        a,b, N = self.consts['a'],self.consts['b'], self.max_degree+1
        n = np.arange(0,N)
        na = n+a
        nb = n+b
        nab = n+a+b
        nnab = 2*n+a+b

        # (1-z) <a,b| = <a-1,b| A-
        if a > 0:
            if a+b==0:
                middle = na/(2*n+1)
                lower  = (nb+1)/(2*n+1)
                middle[0]  = 2*a
            else:
                middle = 2*na*nab/(nnab*(nnab+1))
                lower  = 2*(n+1)*(nb+1)/((nnab+1)*(nnab+2))
            self.op['A-'] = diag([-np.sqrt(lower),np.sqrt(middle)],[-1,0])

        # <a,b| = <a+1,b| A+
        if a+b == 0 or a+b == -1:
            middle = (na+1)/(2*n+1)
            upper  = nb/(2*nab+1)
            middle[0], upper[0] = (1+a)*(1-(a+b)), 0
        else:
            middle = 2*(na+1)*(nab+1)/((nnab+1)*(nnab+2))
            upper  = 2*n*nb/(nnab*(nnab+1))
        self.op['A+'] = diag([np.sqrt(middle),-np.sqrt(upper)],[0,+1])

        # (1+z) <a,b| = <a,b-1| B-
        if b > 0:
            if a+b == 0:
                middle = nb/(2*n+1)
                lower  = (na+1)/(2*n+1)
                middle[0] = 2*b
            else:
                middle = 2*nb*nab/(nnab*(nnab+1))
                lower  = 2*(n+1)*(na+1)/((nnab+1)*(nnab+2))
            self.op['B-'] = diag([np.sqrt(lower),np.sqrt(middle)],[-1,0])

        # <a,b| = <a,b+1| B+
        if a+b == 0 or a+b == -1:
            middle = (nb+1)/(2*n+1)
            upper  = na/(2*nab+1)
            middle[0], upper[0] = (1+b)*(1-(a+b)), 0
        else:
            middle = 2*(nb+1)*(nab+1)/((nnab+1)*(nnab+2))
            upper  = 2*n*na/(nnab*(nnab+1))
        self.op['B+'] = diag([np.sqrt(middle),np.sqrt(upper)],[0,+1])

        # ( a - (1-z)*d/dz ) <a,b| = <a-1,b+1| C-
        if a > 0:
            self.op['C-'] = diag([np.sqrt(na*(nb+1))],[0])

        # ( b + (1+z)*d/dz ) <a,b| = <a+1,b-1| C+
        if b > 0:
            self.op['C+'] = diag([np.sqrt((na+1)*nb)],[0])

        # ( a(1+z) - b(1-z) - (1-z^2)*d/dz ) <a,b| = <a-1,b-1| D-
        if a > 0 and b > 0:
            self.op['D-'] = diag([np.sqrt((n+1)*nab)],[-1])

        # d/dz <a,b| = <a+1,b+1| D+
        self.op['D+'] = diag([np.sqrt(n*(nab+1))],[+1])

        # z <a,b| = <a,b| J
        self.op['J'] = 0.5*(self.pull('B+',self.op['B+']) - self.pull('A+',self.op['A+']))

        self.op['z=+1'] = np.sqrt(nnab+1)*np.sqrt(fun.binom(na,a))*np.sqrt(fun.binom(nab,a))
        self.op['z=-1'] = ((-1)**n)*np.sqrt(nnab+1)*np.sqrt(fun.binom(nb,b))*np.sqrt(fun.binom(nab,b))
        if a+b==-1:
            self.op['z=+1'][0] = np.sqrt(np.sin(np.pi*np.abs(a))/np.pi)
            self.op['z=-1'][0] = np.sqrt(np.sin(np.pi*np.abs(b))/np.pi)

    def convert_ops(self,format='crs'):

        if format == 'crs':
            for op in self.op:
                if op != 'z=+1' and op != 'z=-1': self.op[op] = sparse.csr_matrix(self.op[op])

        if format == 'dia':
            for op in self.op:
                if op != 'z=+1' and op != 'z=-1': self.op[op] = sparse.dia_matrix(self.op[op])

    def print_op(self,op):
        if op == 'z=-1' or op == 'z=+1':
            print(self.op[op])
        else:
            print(self.op[op].todense())
        print('-----------------------------------------------------------------------------------------------')



