
import scqubits as scq
import qutip as qt
import numpy as np

sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()

for r in range(len(args['U_ops'])):
    HFloquet.append([args['U_ops'][r], \
                     lambda t, args: Pulse(t - args['t_pulse'][r], args)])

def FloquetSpectrum(H0, V_drive, nsteps):
    """
    :param prms:
    :param qb:
    :return:
    """
    args = {'amplitude': d_prms[0], 'width': d_prms[1], 'period': d_prms[2],'U_ops':d_prms[3], 't_pulse':d_prms[4]}

    HFloquet = [(2 * np.pi) * qubit_hamiltonian(qb, q_prms)]



    # HFloquet = [(2 * np.pi) * qubit_hamiltonian(qb, q_prms), \
    #             [sx, lambda t, args: Kick_X(t, args)],\
    #             [sy, lambda t, args: Kick_Y(t, args)], \
    #             [sz, lambda t, args: Kick_Z(t, args)]]
    options = qt.Options()
    options.nsteps = nsteps
    return qt.floquet_modes(HFloquet, args['period'], args=args, options=options)


def flat_dev(spectrum):

    return np.sum(np.abs(np.diff(spectrum)))