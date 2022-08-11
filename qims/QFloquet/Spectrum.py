import numpy as np
import qutip as qt
import qims as qims

sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()


def FloquetSpectrum(H0, args, nsteps):
    """

    :param H0: function that produces qutip static Hamiltonian
    :param args: list of dictionary containing type of drive operator, drive profile and drive parameters
    :param nsteps: number of steps used in the qutip QFloquet integrator
    :return:
    """
    # Extract position of driving operators in input sequence of operators
    for op_types in args["op_data"]["op_types"]:
        args[op_types[0]] = qims.extract_oploc(op_types[1], args)

    # If input of a given drive parameter is a single number then the assumption is that the parameter applies to all driving operators
    for term in list(args["drive_prms"].keys()):
        chk = type(args["drive_prms"][term])
        if ( type(chk) != list):
            args["drive_prms"][term] = args["drive_prms"][term] * np.ones(
                len(args["op_data"]["op_sequence"]))

    # Full time-dependent Hamiltonian
    HFloquet = [H0(args["static_prms"]), [sx, qims.drive_sx],
                [sy, qims.drive_sy],
                [sz, qims.drive_sz]]

    # Set number of integration steps
    options = qt.Options()
    options.nsteps = nsteps

    # Return eigenstates and quasi-energies
    return qt.floquet_modes(HFloquet, args['period'], args=args, options=options), HFloquet


def FloquetSpectrum_Sweep(H0, args, nsteps, sweeps):


    for st_sw in list(sweeps.keys()):
        for dr_sw in sweeps["drive_sweep"]:
            evecs, evals = FloquetSpectrum(H0, args, nsteps)



def flat_dev(spectrum):
    return np.sum(np.abs(np.diff(spectrum)))
