import numpy as np
import qutip as qt
import scipy.sparse
import tqdm.notebook

import qims as qims

#adding test comment
eta = 0.636

def constr(st, r, size):
    return (1 - qims.occ(st, (r - 1) % size, size)) * (1 - qims.occ(st, (r + 1) % size, size))


def pxp(st, r, size):
    j = size - r - 1
    b = 1
    b <<= j  # compute a "mask" integer b which is 1 on site j and zero elsewhere
    st ^= b
    return st


def pxp_hamiltonian(basis, basis_ind, size):
    """
    PXP Hamiltonian
    :param basis: dictionary with mapped basis indices
    :param basis_ind: dictionary inverse of basis dictionary
    :param size: system size
    :return: Hamiltonian as a qutip object
    """
    h = 0
    for r in tqdm.notebook.tqdm(range(size)):
        scan = []
        for st in range(len(basis)):

            if constr(basis[st], r, size) == 1:
                scan.append([basis_ind[pxp(basis[st], r, size)], st, 1])

        elms = np.array(scan)

        row = elms.T[0]
        col = elms.T[1]
        data = elms.T[2]
        h = h + scipy.sparse.csr_matrix((data, (row, col)), shape=(len(basis), len(basis)))

    return qt.Qobj(h)


def sp(st, r, size):
    j = size - r - 1
    b = 1
    b <<= j  # compute a "mask" integer b which is 1 on site j and zero elsewhere
    st ^= b
    return st


def pxp_operators(basis, basis_ind, size, prms):
    """
    PXP Hamiltonian
    :param basis: dictionary with mapped basis indices
    :param basis_ind: dictionary inverse of basis dictionary
    :param size: system size
    :return: Hamiltonian as a qutip object
    """
    SP = 0
    for r in tqdm.notebook.tqdm(range(size)):
        scan = []
        for st in range(len(basis)):
            # print()

            if constr(basis[st], r, size) == 1:

                if (r % 2)==0 and 0 == qims.occ(basis[st], r, size):
                    # print(r, qims.ind2state(st, size))
                    scan.append([basis_ind[pxp(basis[st], r, size)], st, 1])
                elif (r % 2)==1 and 1 == qims.occ(basis[st], r, size):
                    scan.append([basis_ind[pxp(basis[st], r, size)], st, 1])
                else:
                    scan.append([basis_ind[pxp(basis[st], r, size)], st, 0])

        elms = np.array(scan)

        row = np.array(elms.T[0])
        col = np.array(elms.T[1])
        data = np.array(elms.T[2])
        SP = SP + scipy.sparse.csr_matrix((data, (row, col)), shape=(len(basis), len(basis)))
    SP = qt.Qobj(SP)
    Sx = (SP + SP.dag()) / (2*eta)
    Sy = (SP - SP.dag()) / (2 * 1j * eta)
    Sz = 0.5 * (SP * SP.dag() - SP.dag() * SP)/(eta**2)
    S2 = Sx * Sx + Sy * Sy + Sz * Sz

    return Sx, Sy, Sz, S2


def sz_neel(basis, size):
    row = np.arange(0, len(basis))
    col = np.arange(0, len(basis))
    neel = 2 * np.array([np.ones(int(size / 2)), np.zeros(int(size / 2))]).T.flatten() - 1
    data = np.array([np.sum(neel * (2 * qims.ind2state(basis[r], size) - 1)) for r in range(len(basis))])

    return qt.Qobj(scipy.sparse.csr_matrix((data, (row, col)), shape=(len(basis), len(basis))))
