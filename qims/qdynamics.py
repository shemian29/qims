import scqubits as scq
import qutip as qt
import numpy as np

sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()


def qubit_hamiltonian(qb, prms):
    """
        a
    :param qb:
    :return:
    """
    if qb == 'fluxonium':
        fluxonium = scq.Fluxonium(EJ=4,
                                  EC=0.5,
                                  EL=1.3,
                                  flux=0.5,
                                  cutoff=prms[0])

        # Setup of fluxonium qubit
        evals, evecs = fluxonium.eigensys()
        gst = qt.Qobj(evecs.T[0])
        est = qt.Qobj(evecs.T[1])
        Delta = evals[1] - evals[0]

        ph_ext = prms[1]
        B = 2 * 2 * np.pi * (ph_ext - 0.5) * fluxonium.EL * np.abs(
            (est.dag() * qt.Qobj(fluxonium.phi_operator()) * gst).full()[0, 0])
        H0 =0.5*( (Delta / 2) * sx + (B / 2) * sz)

    return H0


def Kick_X(t, args):
    """

    :param t:
    :param args:
    :return:
    """
    xloc = args['period']*(np.where(args['U_total'] == "X")[0]/len(args['U_total']))
    sm = 0
    for tx in xloc:
        sm = sm + args['amplitude'] * (1 / (np.sqrt(2 * np.pi))) * np.e ** \
    (-((t - tx - np.floor(t / args['period']) * args['period']) ** 2) / (
            2 * args['width'] ** 2))
    return sm


def Kick_Y(t, args):
    """

    :param t:
    :param args:
    :return:
    """
    xloc = args['period']*(np.where(args['U_total'] == "Y")[0]/len(args['U_total']))
    sm = 0
    for tx in xloc:
        sm = sm + args['amplitude'] * (1 / (np.sqrt(2 * np.pi))) * np.e ** \
    (-((t - tx - np.floor(t / args['period']) * args['period']) ** 2) / (
            2 * args['width'] ** 2))
    return sm


def Kick_Z(t, args):
    """

    :param t:
    :param args:
    :return:
    """
    xloc = args['period']*(np.where(args['U_total'] == "Z")[0]/len(args['U_total']))
    sm = 0
    for tx in xloc:
        sm = sm + args['amplitude'] * (1 / (np.sqrt(2 * np.pi))) * np.e ** \
    (-((t - tx - np.floor(t / args['period']) * args['period']) ** 2) / (
            2 * args['width'] ** 2))
    return sm

def FloquetSpectrum(qb, q_prms, d_prms):
    """
    :param prms:
    :param qb:
    :return:
    """
    args = {'amplitude': d_prms[0], 'width': d_prms[1], 'period': d_prms[2],'U_total':d_prms[3]}
    # print(args)
    HFloquet = [(2 * np.pi) * qubit_hamiltonian(qb, q_prms), \
                [sx, lambda t, args: Kick_X(t, args)],\
                [sy, lambda t, args: Kick_Y(t, args)], \
                [sz, lambda t, args: Kick_Z(t, args)]]
    options = qt.Options()
    options.nsteps = 1000000000
    return qt.floquet_modes(HFloquet, args['period'], args=args)







# sol = {}
#
# H = qop.pxp_hamiltonian(bs ,bs_ind ,12)
# SZ = qop.sz_neel(size = 12, basis = bs)
#
# wscar = 2* 0.636
# wmax = 3 * wscar
# dw = wscar / 20
# tmax = (2 * wmax / (2 * wmax + dw)) * (2 * np.pi / (dw))
# dt = 2 * np.pi / (2 * wmax + dw)
# t_list = np.linspace(0, tmax, int(tmax / dt))
# print(dt)
#
# for n in tqdm(range(len(bs[nx]))):
#     ProdSt = qt.basis(len(bs[nx]), n)
#     sol[nx, n] = qt.krylovsolve(HQ, ProdSt, t_list, krylov_dim=20, e_ops=[SZ])
#     qt.qsave(sol, 'sol_0')
#     np.savetxt('Monitor_0.txt', [100 * n / len(bs[nx])], fmt='%1.2f')