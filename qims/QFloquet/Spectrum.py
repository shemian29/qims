import numpy as np
import qutip as qt
import qims as qims

sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()


def FloquetSpectrum(H0, V_drives, nsteps):
    """

    :param H0: qutip static Hamiltonian
    :param V_drives: list of dictionary containing type of drive operator, drive profile and drive parameters
    :param nsteps: number of steps used in the qutip QFloquet integrator
    :return:
    """
    args = {'amplitude': d_prms[0], 'width': d_prms[1], 'period': d_prms[2],'U_ops':d_prms[3], 't_pulse':d_prms[4]}

    HFloquet = [H0]
    for r in range(len(V_drives)):
        HFloquet.append([V_drives[r]["operator"], \
                         lambda t, args=args[r]: V_drives[r]["drive_profile"](t, args)])



    options = qt.Options()
    options.nsteps = nsteps
    return qt.floquet_modes(HFloquet, args['period'], args=args, options=options)


def flat_dev(spectrum):

    return np.sum(np.abs(np.diff(spectrum)))