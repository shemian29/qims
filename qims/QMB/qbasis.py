from numpy import array, arange, dot, roll
import numpy as np
from tqdm.notebook import tqdm
import qutip as qt
from scipy.sparse import hstack

def ind2occ(s, r, size):
    """
    Compute the occupation number at a site given a product state index

    :type size: int
    :type s: int
    :type r: int
    :param s: base index of the product state
    :param r: site for which one seeks occupation number
    :param size: system size
    :return: occupation number
    """

    return (s // (2 ** (size - r - 1))) % 2


def ind2state(s, size):
    """

    :param s: base index
    :param size: system size
    :return: state
    """
    return array([ind2occ(s, r, size) for r in range(size)])


def state2ind(state):
    sps = 2 ** (len(state) - arange(len(state)) - 1)
    return dot(state, sps)


def idstates(r, size):
    st = ind2state(r, size)
    inds = dot(st, roll(st, 1))
    if inds == 0.0:
        return r
    else:
        return -1

def GenerateBasis(size, bc = "periodic", parallel = True):



    if parallel == True:
        print("parallel")
        if bc == "periodic":
            Basis = qt.parfor(idstates, range(2 ** size), size = size)
            # Basis = qt.parallel_map(idstates, range(2 ** size), task_kwargs={'size':size}, progress_bar=True)
            Basis = [Basis[r] for r in range(len(Basis)) if Basis[r] > -0.5]
        elif bc == "open":
            Basis = []
            for r in range(2 ** size):
                st = ind2state(r, size)

                inds = dot(st[0:size-1], st[1:size])
                if inds == 0.0:
                    Basis.append(r)

    elif parallel == False:
        if bc == "periodic":
            Basis = []
            for r in tqdm(range(2 ** size)):
                st = ind2state(r, size)
                inds = dot(st, roll(st, 1))
                if inds == 0.0:
                    Basis.append(r)

        elif bc == "open":
            Basis = []
            for r in range(2 ** size):
                st = ind2state(r, size)

                inds = dot(st[0:size-1], st[1:size])
                if inds == 0.0:
                    Basis.append(r)
    return Basis





def occ(st: int, r: int, size: int) -> int:
    """

    :param st:
    :param r:
    :param size:
    :return:
    :rtype: int
    """
    j = size - r - 1  # compute lattice site in reversed bit configuration (cf QuSpin convention for mapping from bits to sites)
    occ_var = (st >> j) & 1
    return occ_var


def basis(size, bc = "periodic", parallel = True):
    """
    Generate basis using mapped indices
    :param size:
    :return: basis mapping, and inverse dictionary
    """

    bstemp = GenerateBasis(size, bc, parallel)
    bs_ind = {}
    bs = {}
    for r in range(len(bstemp)):
        bs_ind[bstemp[r]] = r
        bs[r] = bstemp[r]
    return bs, bs_ind

def Towers(evals,evecs, Hs, U, Sz, Sy, Nx):
    tower = {}
    k_list = np.arange(0, Nx) / Nx
    TP = (Sz - 1j * Sy) / 2
    for q in tqdm(range(0, int(Nx / 2))):
        tower[q] = {}
        ind = 0
        it = 0

        k0 = k_list[q]  # k_list[0]
        ks = k_list[(q + int(Nx / 2)) % Nx]

        mtr = {}
        vs = qt.Qobj([evecs[k0, n].full().T[0] for n in range(Hs[k0].shape[0])]).dag()
        vs2 = qt.Qobj([evecs[ks, n].full().T[0] for n in range(Hs[ks].shape[0])]).dag()

        mtr[k0] = np.abs((vs2.dag() * qt.Qobj(U[ks]).dag() * TP * qt.Qobj(U[k0]) * vs).full())
        mtr[ks] = np.abs((vs.dag() * qt.Qobj(U[k0]).dag() * TP * qt.Qobj(U[ks]) * vs2).full())

        indlist = {}
        indlist[k0] = np.arange(U[k0].shape[1])
        indlist[ks] = np.arange(U[ks].shape[1])

        sm = {}
        sm[k0] = len(indlist[k0])
        sm[ks] = len(indlist[ks])

        while sm[k0] > 0 and sm[ks] > 0:

            E0 = evals[k0, indlist[k0][0]]
            E1 = evals[ks, indlist[ks][0]]

            # print(k0,ks)
            if E0 < E1:
                Eold = E0
                kit = k0
                # print(kit)
            else:
                Eold = E1
                kit = ks
                # print('a',kit)

            ind = indlist[kit][0]
            tower[q][it] = [[kit, ind]]
            indlist[kit] = list(set(indlist[kit]) - set([ind]))

            ind = np.argmax(mtr[kit][:, ind])
            kit = ((int(kit * Nx) + int(Nx / 2)) % Nx) / Nx
            # print(q,kit,ind)
            Enew = evals[kit, ind]

            while Enew > Eold:

                if Enew > Eold and (ind in indlist[kit]):
                    tower[q][it].append([kit, ind])
                Eold = Enew
                indlist[kit] = list(set(indlist[kit]) - set([ind]))

                ind = np.argmax(mtr[kit][:, ind])
                kit = ((int(kit * Nx) + int(Nx / 2)) % Nx) / Nx

                Enew = evals[kit, ind]

            sm[k0] = len(indlist[k0])
            sm[ks] = len(indlist[ks])
            it = it + 1



    evecs_ordered = {}

    for q in tqdm(range(0, int(Nx / 2))):
        for tw in range(len(tower[q])):
            for r in range(len(tower[q][tw])):
                k, n = tower[q][tw][r]
                if r == 0 and tw == 0:
                    vtemp = (qt.Qobj(U[k]) * evecs[k, n]).data
                else:
                    vtemp = hstack((vtemp, (qt.Qobj(U[k]) * evecs[k, n]).data))
        evecs_ordered[q] = qt.Qobj(vtemp)

    return evecs_ordered, tower