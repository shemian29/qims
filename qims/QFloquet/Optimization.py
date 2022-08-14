%matplotlib inline
import matplotlib.pyplot as plt

import qutip as qt
import numpy as np
import scipy as sc
import scqubits as scq

import datetime
from tqdm.notebook import tqdm

s0 = qt.identity(2)
sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()


def ht(t, args):
    # T, prms_dc, prms_ac, prms_ext, ops_ext

    T = args[0]
    prms_dc = args[1]
    prms_ac = args[2]
    prms_ext = args[3]
    ops_ext = args[4]

    prms_fixed = args[5]

    N_c = args[6]
    omega = 2 * np.pi / T

    cost_sint = np.array(np.cos(omega * np.arange(1, Nc + 1) * t).tolist() \
                         + np.sin(omega * np.arange(1, Nc + 1) * t).tolist())

    f_t = []
    for r in range(len(prms_ac)):
        f_t.append(np.dot(prms_ac[r], cost_sint))

    V_t = 0
    for r in range(len(prms_ext)):
        V_t = V_t + np.dot(prms_ext[r], cost_sint) * ops_ext[r]

    h0 = h_static(prms_dc + f_t, prms_fixed) + V_t

    return h0


def Spectrum(prms, args):
    prms_dc = args[0]
    ops_ext = args[1]

    N_c = args[2]
    T = prms[-1]

    prms = prms[0:len(prms) - 1]
    prms_ac = np.reshape(prms[0:2 * len(prms_dc) * Nc], (len(prms_dc), 2 * Nc))
    prms_ext = np.reshape(prms[2 * len(prms_dc) * Nc:len(prms)], (len(ops_ext), 2 * Nc))
    # print(prms_ac.shape)
    scan = []

    for phix in tqdm(np.linspace(0.5 - 0.02, 0.56, 50)):
        p_fx = np.array([phix])
        evecs, evals = qt.floquet_modes(ht, T, [T, prms_dc, prms_ac, prms_ext, ops_ext, p_fx, N_c])
        # Return the
        scan.append(np.sort(evals))

    return np.array(scan)


def Error(prms, args):
    prms_dc = args[0]
    ops_ext = args[1]

    N_c = args[2]
    T = prms[-1]
    prms = prms[0:len(prms) - 1]
    prms_ac = np.reshape(prms[0:2 * len(prms_dc) * Nc], (len(prms_dc), 2 * Nc))
    prms_ext = np.reshape(prms[2 * len(prms_dc) * Nc:len(prms)], (len(ops_ext), 2 * Nc))
    scanT = []

    for phix in np.linspace(0.5 - 0.02, 0.56, 10):
        #             ty = prms_dc[0]
        #             tx = prms_dc[1]
        #             m = prms_dc[2]

        #             kx = prms_fixed[0]
        #             ky = prms_fixed[1]

        p_fx = np.array([phix])
        evecs, evals = qt.floquet_modes(ht, T, [T, prms_dc, prms_ac, prms_ext, ops_ext, p_fx, N_c])
        # Return the
        scanT.append(np.sum(np.abs(evals)))

    return np.sum(np.abs(np.gradient(np.array(scanT))))


