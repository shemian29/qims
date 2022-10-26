
import numpy as np
import qutip as qt
import qims as qims
from scipy.sparse import block_diag
import scipy as scp



def error(prms, args):
    # print("-> Generate PXP operators")
    Sx, Sy, Sz, S2, Sz_alt, S2_alt, OP = qims.pxp_operators(args['bs'], args['bs_ind'], args['Nx'], prms)

    # print("-> Generate momentum-space Hamiltonians and basis transformation")
    Hs, U = qims.Hk(hamiltonian=Sx,
                    basis=args['bs'],
                    basis_ind=args['bs_ind'],
                    size=args['Nx'],
                    check_spect=False,
                    check_symm=False)

    # print("-> Calculate momentum eigenspectrum")
    evals, evecs, Ukevecs = qims.MomentumEigensystem(Hs, U, S2, args['Nx'])
    evecs_ordered, evals_ordered, towers = qims.Towers(evals, evecs, Hs, U, Sz, Sy, args['Nx'])

    sm = 0

    for q in range(0, int(args['Nx'] / 2)):
        # for q in [0]:
        ws = []
        for r in range(len(towers[q])):
            scan = []
            for ind in towers[q][r]:
                scan.append(evals[ind[0]][ind[1]])
            if len(scan) > 1:
                ws.append((np.mean(np.diff(scan)) * qt.identity(len(scan))).data)
            else:
                ws.append(0)
        Omega = qt.Qobj(block_diag(ws).tocsr())

        Tp = (evecs_ordered[q].dag() * ((Sz - 1j * Sy) / 2) * evecs_ordered[q])
        EE = (evecs_ordered[q].dag() * (Sx) * evecs_ordered[q])

        sm = sm + np.sum(np.abs(((EE * Tp - Tp * EE) - Omega * Tp).full()))
    return sm


def opt_towers(bs, bs_ind, Nx,xinit = np.zeros(9)):
    # print("Finding solution ("+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+")")

    args = {
        'bs': bs,
        'bs_ind': bs_ind,
        'Nx': Nx
    }


    print('Starting point: ',qims.error(xinit,args),xinit)
    x0 = xinit
    bnds = [[-1, 1] for r in range(9)]
    sol = scp.optimize.minimize(qims.error, x0, args=args, bounds=bnds, method='SLSQP', options={'disp': True})

    # print("Obtained solution  ("+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+")")
    print('Optimzied:', qims.error(sol.x, args), sol.x)

    return sol


def GenScarData(bs, bs_ind, Nx, prms = np.zeros(9)):
    args = {
        'bs': bs,
        'bs_ind': bs_ind,
        'Nx': Nx
    }

    # print("-> Generate PXP operators")
    Sx, Sy, Sz, S2, Sz_alt, S2_alt, OP = qims.pxp_operators(args['bs'], args['bs_ind'], args['Nx'], prms)

    # print("-> Generate momentum-space Hamiltonians and basis transformation")
    Hs, U = qims.Hk(hamiltonian=Sx,
                    basis=args['bs'],
                    basis_ind=args['bs_ind'],
                    size=args['Nx'],
                    check_spect=False,
                    check_symm=False)

    # print("-> Calculate momentum eigenspectrum")
    evals, evecs, Ukevecs = qims.MomentumEigensystem(Hs, U, S2, args['Nx'])
    evecs_ordered, evals_ordered, towers = qims.Towers(evals, evecs, Hs, U, Sz, Sy, args['Nx'])


    return Sx, Sy, Sz, S2, Sz_alt, S2_alt, OP, Hs, U, evals, evecs, Ukevecs, evecs_ordered, evals_ordered, towers