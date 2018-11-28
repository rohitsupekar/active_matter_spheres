import numpy             as np
import scipy             as sp
import scipy.sparse      as sparse
import scipy.linalg      as linear
import scipy.special     as fun
import jacobi            as jac
from dedalus.tools.array import reshape_vector

def grid(L_max):
    P00 = jac.Jacobi(L_max,a=0,b=0)
    cos = P00.grid
    return np.arccos(cos)

class Sphere:
    """Defines Spin-Weighted Spherical Harmonics, and their assosciated operators"""

    def __init__(self,L_max,S_max=0,m_min=None,m_max=None):
        self.L_max, self.S_max  = L_max, S_max

        if m_min == None: m_min = -L_max
        if m_max == None: m_max = L_max+1

        # grid and weights for the all transforms
        P00 = jac.Jacobi(L_max,a=0,b=0)

        self.cos,  self.sin     = P00.grid, np.sqrt(1-P00.grid**2)
        self.grid, self.weights = np.arccos(self.cos), P00.weights

        # [v(+),v(-)] = U.[v(theta),v(phi)]
        self.U = np.sqrt(0.5)*np.array([[1,1j],[1,-1j]])
        # [v(theta),v((phi)] = Udag.[v(+),v(-)]
        self.Udag = np.sqrt(0.5)*np.array([[1,1],[-1j,1j]])

        self.Y  = {(0,0):P00.poly}
        self.op = {(0,0):{'I':P00.op['I']}}

        for s in range(-S_max,S_max+1):
            for m in range(m_min,m_max):

                a,b,N = abs(m+s),abs(m-s), L_max-max(abs(m),abs(s))
                P = jac.Jacobi(N,a=a,b=b,grid_basis=P00)

                self.Y[(m,s)]  = P.poly

                # identity & cosine multplication
                self.op[(m,s)] = {'I':P.op['I']}
                self.op[(m,s)]['C'] = P.op['J']

                # s -> s+1 derivative
                da,db = abs(m+s+1)-a, abs(m-s-1)-b
                if (da== 1) and (db==-1): self.op[(m,s)]['k+'] =   -np.sqrt(0.5)*P.op['C+']
                if (da==-1) and (db== 1): self.op[(m,s)]['k+'] =    np.sqrt(0.5)*P.op['C-']
                if (da== 1) and (db== 1): self.op[(m,s)]['k+'] =   -np.sqrt(0.5)*P.op['D+'][:-1,:]

                # s -> s-1 derivative
                da,db = abs(m+s-1)-a, abs(m-s+1)-b
                if (da== 1) and (db==-1): self.op[(m,s)]['k-'] =  -np.sqrt(0.5)*P.op['C+']
                if (da==-1) and (db== 1): self.op[(m,s)]['k-'] =   np.sqrt(0.5)*P.op['C-']
                if (da== 1) and (db== 1): self.op[(m,s)]['k-'] =  -np.sqrt(0.5)*P.op['D+'][:-1,:]

                # s -> s+1 sin(theta) {first pass}
                da,db = abs(m+s+1)-a, abs(m-s-1)-b
                if (da== 1) and (db==-1): self.op[(m,s)]['S+'] =   P.op['A+']
                if (da==-1) and (db== 1): self.op[(m,s)]['S+'] =   P.op['B+']
                if (da== 1) and (db== 1): self.op[(m,s)]['S+'] =   P.op['A+'][:-1,:]
                if (da==-1) and (db==-1): self.op[(m,s)]['S+'] =   P.op['A-']

                # s -> s-1 sin(theta) {first pass}
                da,db = abs(m+s-1)-a, abs(m-s+1)-b
                if (da== 1) and (db==-1): self.op[(m,s)]['S-'] =   P.op['A+']
                if (da==-1) and (db== 1): self.op[(m,s)]['S-'] =   P.op['B+']
                if (da== 1) and (db== 1): self.op[(m,s)]['S-'] =   P.op['A+'][:-1,:]
                if (da==-1) and (db==-1): self.op[(m,s)]['S-'] =   P.op['A-']

        # Trust us (this really works):
        for s in range(-S_max,S_max+1):
            for m in range(m_min,m_max):

                a,b = abs(m+s),abs(m-s)

                # s -> s+1 derivative & sin(theta) {second pass}
                da,db,N = abs(m+s+1)-a, abs(m-s-1)-b, L_max-max(abs(m),abs(s+1))
                P = jac.Jacobi(N,a=a+da,b=b+db,grid_basis=P00)
                if s > -S_max :
                    if (da== 1) and (db==-1): self.op[(m,s)]['S+'] = P.pull('B+',self.op[(m,s)]['S+'])
                    if (da==-1) and (db== 1): self.op[(m,s)]['S+'] = P.pull('A+',self.op[(m,s)]['S+'])
                    if (da== 1) and (db== 1): self.op[(m,s)]['S+'] = P.pull('B-',self.op[(m,s)]['S+'])
                    if (da==-1) and (db==-1):
                        P.op['B+'] = P.op['B+'][:-1,:]
                        self.op[(m,s)]['S+'] =  P.pull('B+',self.op[(m,s)]['S+'])
                        self.op[(m,s)]['k+'] = -self.op[(m,s+1)]['k-'].transpose()

                # s -> s-1 derivative & sin(theta) {second pass}
                da,db,N = abs(m+s-1)-a, abs(m-s+1)-b, L_max-max(abs(m),abs(s-1))
                P = jac.Jacobi(N,a=a+da,b=b+db,grid_basis=P00)
                if s <  S_max :
                    if (da== 1) and (db==-1): self.op[(m,s)]['S-'] = P.pull('B+',self.op[(m,s)]['S-'])
                    if (da==-1) and (db== 1): self.op[(m,s)]['S-'] = P.pull('A+',self.op[(m,s)]['S-'])
                    if (da== 1) and (db== 1): self.op[(m,s)]['S-'] = P.pull('B-',self.op[(m,s)]['S-'])
                    if (da==-1) and (db==-1):
                        P.op['B+'] = P.op['B+'][:-1,:]
                        self.op[(m,s)]['S-'] =  P.pull('B+',self.op[(m,s)]['S-'])
                        self.op[(m,s)]['k-'] = -self.op[(m,s-1)]['k+'].transpose()

        P00,P = 0,0

    def forward_spin(self,m,s,data):
        # grid --> coefficients
        return self.Y[(m,s)].dot(reshape_vector(self.weights,dim=len(data.shape),axis=0)*data)

    def backward_spin(self,m,s,data):
        # coefficients --> grid
        return (data.T.dot(self.Y[(m,s)])).T

    def tensor_index(self,m,rank):
        num = np.arange(2**rank)
        spin = (-1)**num
        for k in range(2,rank+1):
            spin += ((-1)**(num//2**(k-1))).astype(np.int64)

        if rank == 0: spin = [0]

        start_index = [0]
        end_index = []
        for k in range(2**rank):
            end_index.append(start_index[k]+self.L_max-self.L_min(m,spin[k])+1)
            if k < 2**rank-1:
                start_index.append(end_index[k])

        return (start_index,end_index,spin)

    def forward(self,m,rank,data):

        if rank == 0:
            return self.forward_spin(m,0,data)

        (start_index,end_index,spin) = self.tensor_index(m,rank)

        unitary = self.U
        for k in range(rank-1):
            unitary = np.kron(unitary,self.U)

        #data = unitary.dot(data)
        data = np.einsum("ij,j...->i...",unitary,data)

        shape = np.array(np.array(data).shape[1:])
        shape[0] = end_index[-1]

        data_c = np.zeros(shape,dtype=np.complex128)

        for i in range(2**rank):
            data_c[start_index[i]:end_index[i]] = self.forward_spin(m,spin[i],data[i])
        return data_c

    def backward(self,m,rank,data):

        if rank == 0:
            return self.backward_spin(m,0,data)

        (start_index,end_index,spin) = self.tensor_index(m,rank)

        unitary = self.Udag
        for k in range(rank-1):
            unitary = np.kron(unitary,self.Udag)

        shape = np.array(np.array(data).shape)
        shape = np.concatenate(([2**rank],shape))
        shape[1] = self.L_max+1

        data_g = np.zeros(shape,dtype=np.complex128)

        for i in range(2**rank):
            data_g[i] = self.backward_spin(m,spin[i],data[start_index[i]:end_index[i]])
        #return unitary.dot(data_g)
        return np.einsum("ij,j...->i...",unitary,data_g)

    def grad(self,m,rank_in,data,data_out):
        # data and data_out are in coefficient space

        (start_index_in,end_index_in,spin_in) = self.tensor_index(m,rank_in)
        rank_out = rank_in+1
        (start_index_out,end_index_out,spin_out) = self.tensor_index(m,rank_out)

        half = 2**(rank_out-1)
        for i in range(2**(rank_out)):
            if i//half == 0:
                op = self.op[(m,spin_in[i%half])]['k+']
            else:
                op = self.op[(m,spin_in[i%half])]['k-']

            np.copyto( data_out[start_index_out[i]:end_index_out[i]],
                       op.dot(data[start_index_in[i%half]:end_index_in[i%half]]) )

    def contract(self,rank_L,data_L,rank_R,data_R):
        # returns sum_i data_L_{a,b,...,i} data_R_{i,c,d, ...}

        rank_out = rank_L + rank_R - 2
        data_out = np.zeros([2**rank_out,self.L_max+1],dtype=np.complex128)

        for i in range(2**(rank_L-1)):
            for j in range(2**(rank_R-1)):
                data_out[j+2**(rank_R-1)*i] += data_L[2*i]*data_R[j] + data_L[2*i+1]*data_R[j+2**(rank_R-1)]

        return data_out


    def L_min(self,m,s):
        return max(abs(m),abs(s))

    def zeros(self,m,s_out,s_in):
        return np.zeros([self.L_max+1-self.L_min(m,s_out),self.L_max+1-self.L_min(m,s_in)], dtype=np.complex128)








