import numpy as np
from numpy.linalg import pinv, norm, svd
from scipy.optimize import fmin

import numpy as np
from numpy.linalg import svd, pinv

def null_matlab(A, rtol=1e-12):
    # Equivalent to MATLAB null(A) using SVD
    U, s, Vt = svd(A)
    rank = np.sum(s > rtol * s[0])      # same rank logic as MATLAB
    return Vt[rank:].T                  # null space basis (4×k)

def prepare_multilateration(emitters):
    emitters = np.asarray(emitters)
    xe, ye, ze = emitters[:, 0], emitters[:, 1], emitters[:, 2]
    N = len(emitters)

    # Build A = [1, -2xe, -2ye, -2ze]
    A = np.column_stack([np.ones(N), -2*xe, -2*ye, -2*ze])

    # Pseudo-inverse
    pinvA = pinv(A)

    # Null-space identical to MATLAB null(A)
    xh = null_matlab(A)

    # Geometric constant
    b_const = -(xe**2 + ye**2 + ze**2)

    return dict(A=A, pinvA=pinvA, xh=xh, b_const=b_const)
def multilateration_fast(precomp, R):
    R = np.asarray(R).flatten()
    pinvA = precomp["pinvA"]
    xh = precomp["xh"]
    b_const = precomp["b_const"]

    b = R**2 + b_const
    xp = pinvA @ b

    # Full-rank case → unique
    if xh.shape[1] == 0:
        x = xp
    else:
        h = xh[:,0]

        def q(t):
            X0 = xp + t*h
            return (X0[0] - X0[1]**2 - X0[2]**2 - X0[3]**2)**2

        t_opt = fmin(q, 0, disp=False)
        x = xp + t_opt*h

    R0 = x[0]
    X, Y, Z = x[1], x[2], x[3]
    return np.array([X, Y, Z])




# Emitters coordinates [mm]
xe = np.array([7.64121, 12.36373,0 , -12.36373, -7.64121])
ye = np.array([10.51722, -4.01722, -13.0, -4.01722, 10.517])
ze = np.array([0, 0, 0, 0, 0])
#
#
xe = np.array([-12.36373, 0, 12.36373])
ye = np.array([-4.01722, -13.0, -4.01722])
ze = np.array([0, 0, 0])

# Stack into (N,3) for the Python function
emitters = np.column_stack([xe, ye, ze])

# Receiver coordinate [mm]
xr, yr, zr = 20, -5, 50

# Compute distances from receiver to each emitter
th_Dist_em_reco = np.sqrt((xr - xe)**2 + (yr - ye)**2 + (zr - ze)**2)
th_Dist_em_reco = [72,69,64]

print(emitters)
# Precompute geometry for multilateration
precomp = prepare_multilateration(emitters)


# Compute receiver position
pos_receiver = multilateration_fast(precomp, th_Dist_em_reco)
X, Y, Z = pos_receiver

print(f"Receiver position: X={X:.2f}, Y={Y:.2f}, Z={Z:.2f}")
