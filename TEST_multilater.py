import numpy as np
from numpy.linalg import pinv, norm, svd
from scipy.optimize import fmin


def prepare_multilateration(emitters):
    emitters = np.asarray(emitters)
    xe, ye, ze = emitters[:,0], emitters[:,1], emitters[:,2]
    N = len(emitters)

    A = np.column_stack([np.ones(N), -2*xe, -2*ye, -2*ze])
    pinvA = pinv(A)

    # null-space of A
    U, S, Vt = svd(A)
    tol = 1e-12
    null_mask = (S < tol)
    if np.all(~null_mask):
        xh = np.zeros((4,0))
    else:
        xh = Vt.T[:, null_mask]

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
xe = np.array([7.64121, 12.36373, 0, -12.36373, -7.64121])
ye = np.array([10.51722, -4.01722, -13.0, -4.01722, 10.517])
ze = np.array([0, 0, 0, 0, 0])

# Stack into (N,3) for the Python function
emitters = np.column_stack([xe, ye, ze])

# Receiver coordinate [mm]
xr, yr, zr = -0.16, -0.1, 34

# Compute distances from receiver to each emitter
th_Dist_em_reco = np.sqrt((xr - xe)**2 + (yr - ye)**2 + (zr - ze)**2)

print(emitters)
# Precompute geometry for multilateration
precomp = prepare_multilateration(emitters)

# Compute receiver position
pos_receiver = multilateration_fast(precomp, th_Dist_em_reco)
X, Y, Z = pos_receiver

print(f"Receiver position: X={X:.2f}, Y={Y:.2f}, Z={Z:.2f}")
