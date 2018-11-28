import numpy             as np
import scipy.sparse      as sparse
import matplotlib.pyplot as plt
from scipy.linalg         import eig
#from mpl_toolkits.basemap import Basemap

# parameters:
# g: (gamma) surface mass density
# e0: friction term
# e1: forcing laplacian term
# e2: small scale damping term

# equations:

# g*dt(up) - kp*p - 2*km*eps*kp*up = - (u.grad u)_p
# g*dt(um) - km*p - 2*kp*eps*km*um = - (u.grad u)_m
# kp*um + km*up = 0
# where
# eps = e0 + e1*(kp*km + km*kp) + e2*(kp*km + km*kp)*(kp*km + km*kp)

# variable order: + - p

def advection(S,m,params):
    """Defines M, L matrices for advection"""
        
    g,e0,e1,e2  = params[0],params[1],params[2],params[3]
    
    # (+,+)    
    L00 = -2*S.op('k-',m,2).dot(
                ( e0*S.op('I',m,2) + e1*( S.op('k+',m,1).dot(S.op('k-',m,2)) 
                                         +S.op('k-',m,3).dot(S.op('k+',m,2)))
                                   + e2*( S.op('k+',m,1).dot(S.op('k-',m,2)) 
                                         +S.op('k-',m,3).dot(S.op('k+',m,2))).dot
                                        ( S.op('k+',m,1).dot(S.op('k-',m,2))
                                         +S.op('k-',m,3).dot(S.op('k+',m,2)))
                                                                            ).dot(S.op('k+',m,1)) )

    # (+,-)
    L01 = S.zeros(m,1,-1)

    # (+,p)
    L02 = -S.op('k+',m,0)

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

    # (-,p)
    L12 = -S.op('k-',m,0)

    # (p,+)
    L20 = S.op('k-',m,1)

    # (p,-)
    L21 = S.op('k+',m,-1)

    # (p,p)
    L22 = S.zeros(m,0,0)
    if m == 0:
        L22[0,0] = 1.

    L = sparse.bmat([[L00,L01,L02],[L10,L11,L12],[L20,L21,L22]])

    R00 = (g)*S.op('I',m,1)
    R01 = S.zeros(m,1,-1)
    R02 = S.zeros(m,1,0)

    R10 = S.zeros(m,-1,1)
    R11 = (g)*S.op('I',m,-1)
    R12 = S.zeros(m,-1,0)

    R20 = S.zeros(m,0,1)
    R21 = S.zeros(m,0,-1)
    R22 = S.zeros(m,0,0)

    R = sparse.bmat([[R00,R01,R02],[R10,R11,R12],[R20,R21,R22]])

    return R,L

def unpack(S,m,vec):
    
    (start_index,end_index,spins) = S.tensor_index(m,1)
    v    = vec[0:end_index[-1]]
    p    = vec[end_index[-1]:end_index[-1]+S.L_max-S.L_min(m,0)+1]

    return v,p

def packup(v_c,p_c):
    return np.hstack([v_c,p_c])

def eigensolve(R,L):
    vals, vecs = eig(L,b=-R)
    i = np.argsort(-vals.real)
    vals, vecs = vals[i], vecs.transpose()[i]
    vecs = vecs[np.isfinite(vals)]
    vals = vals[np.isfinite(vals)]
    return vals,vecs

def show_ball(S, field, index, longitude=0, latitude=0, mp = None):
    
    if mp == None:
        figure, ax = plt.subplots(1,1)
        figure.set_size_inches(3,3)

    lon = np.linspace(0, 2*np.pi, 2*(S.L_max+1))
    lat = S.grid - np.pi/2
    
    meshed_grid = np.meshgrid(lon, lat)
    lat_grid = meshed_grid[1]
    lon_grid = meshed_grid[0]
    
    if mp == None:
        mp = Basemap(projection='ortho', lat_0=latitude, lon_0=longitude, ax=ax)
        mp.drawmapboundary()
        mp.drawmeridians(np.arange(0, 360, 30))
        mp.drawparallels(np.arange(-90, 90, 30))

    x, y = mp(np.degrees(lon_grid), np.degrees(lat_grid))
    im = mp.pcolor(x, y, np.transpose(field), cmap='RdYlBu_r')
    
    
    plt.savefig('images/om_%05i.png' %index)
    return im,mp
