import qims as qims
import numpy as np
from tqdm.notebook import tqdm
import scipy.sparse
import qutip as qt
from scipy.sparse import hstack

def TransInd(s, size):
    """

    :param s: basis mapping index
    :param size: system size
    :return:
    """
    return int(qims.ind2occ(s, size - 1, size) * (2 ** (size - 1)) - (qims.ind2occ(s, size - 1, size) / (2)) + s / (2))


def GenerateMomentumBasis(size, basis):
    """

    :param size: system size
    :return: lists of basis mapped indices that form cycles
    """
    dms = (2) ** size
    lst = np.array(list(basis.items()))[:,1].tolist()
    vbs = []

    while len(lst) > 0:

        np.savetxt("Monitor_" + str([size]) + ".txt", [(100 * len(lst) / dms)], fmt='%1.4f')

        bas = [lst[0]]
        tmp = TransInd(bas[-1], size)

        while (tmp - bas[0]) != 0:
            bas.append(tmp)
            tmp = TransInd(bas[-1], size)

        vbs.append(bas)

        lst = list(set(lst) - set(bas))

    return vbs


def Hk(hamiltonian, basis, basis_ind,size, check_symm = False, check_spect = False):


    print("Setup momentum basis")

    k_list = np.arange(0, size) / size
    scan = {}
    for kk in k_list:
        scan[kk] = []
    vbs = GenerateMomentumBasis(size, basis)
    for n in tqdm(range(len(vbs))):

        ktemp_list = np.arange(0, len(vbs[n])) / len(vbs[n])
        inds = list(map(basis_ind.get, vbs[n]))

        for kk in ktemp_list:
            phs = np.exp(-2 * np.pi * kk * 1j * np.arange(0, len(inds))) / np.sqrt(len(inds))
            scan[kk].append([inds, phs])


    print("Calculating k-basis transformations")
    U = {}
    for kk in tqdm(k_list):
        # print('---------------------------')
        U[kk] = 0
        for n in range(len(scan[kk])):
            row = n * np.ones(len(scan[kk][n][0]))
            col = np.array(scan[kk][n][0])
            data = np.array(scan[kk][n][1])

            U[kk] = U[kk] + scipy.sparse.csr_matrix((data, (row, col)), shape=(len(scan[kk]), \
                                                                               len(basis))).T

    print("Calculate momentum Hamiltonians")
    Hs = {}
    scan = []
    for k in tqdm(k_list):
        U[k] = qt.Qobj(U[k])
        Hs[k]=(U[k].dag()*hamiltonian*U[k])
        scan.append(Hs[k].eigenstates()[0])
    scan = [x for xs in scan for x in xs]


    if (check_symm == True):
        if hamiltonian.shape[0]<4000:
            print("Run symmetry check:")
            Trn = 0
            for n in range(len(vbs)):
                inds = list(map(basis_ind.get, vbs[n]))
                row = inds
                col = np.roll(inds, 1)
                data = np.ones(len(np.roll(inds, 1)))

                Trn = Trn + scipy.sparse.csr_matrix((data, (row, col)), shape=(len(basis), \
                                                                               len(basis))).T
            Trn = qt.Qobj(Trn)
            commutator =  np.sum(np.abs((Trn*hamiltonian-hamiltonian*Trn).full()))
            if commutator == 0.0:
                print(" -> Passed: Hamiltonian is translationally invariant")
            else:
                print("WARNING: Hamiltonian is NOT translationally invariant, failure amount: ",commutator)
        else:
            print("Hilbert space is too big to perform symmetry check")

    if check_spect == True:
        if hamiltonian.shape[0]<4000:
            print("Run spectrum check:")
            de = hamiltonian.eigenenergies() - np.sort(scan)
            dE = np.sum(np.abs(de))
            if dE < 10**(-10):
                print(" -> Passed: Spectra of full H and the H(k) match")
            else:
                print("WARNING: Spectra of full H and the H(k) do NOT match, failure amount: ",dE)
        else:
            print("Hilbert space is too big to perform spectrum check")


    return Hs, U


def MomentumEigensystem(Hs, U, S2, size):
    evecs = {}
    evals = {}
    Ukevecs = {}
    k_list = np.arange(0, size) / size
    for k in tqdm(k_list):

        # print(k*size,Hs[k].shape[0]/2)
        evals_temp, evecs_temp = Hs[k].eigenstates()

        print("Reorder at: ", k)
        zrs = evecs_temp[np.ix_(np.where(np.abs(evals_temp) < 10 ** (-10))[0])]
        print(len(zrs))
        if len(zrs)!= 0:
            vtemp = zrs[0]
            for n in range(1, len(zrs)):
                vtemp = hstack((vtemp, zrs[n].data))
            evecs_set = qt.Qobj(vtemp)

            S2_proj = (U[k] * evecs_set).dag() * S2 * U[k] * evecs_set
            reord = S2_proj.eigenstates()[1]

            vtemp = reord[0]
            for n in range(1, len(reord)):
                vtemp = hstack((vtemp, reord[n].data))
            transf = qt.Qobj(vtemp)

            tmp = [qt.Qobj((evecs_set * transf)[:, r]) for r in range(len(zrs))]

            evecs_temp[np.ix_(np.where(np.abs(evals_temp) < 10 ** (-10))[0])] = tmp

            # chk = np.sum(np.abs(evals_temp - [
            #     ((qt.Qobj(U[k]) * evecs_temp[r]).dag() * Sx * qt.Qobj(U[k]) * evecs_temp[r])[0, 0] for r in
            #     range(len(evals_temp))]))


        scan = []
        for n in range(len(evals_temp)):
            scan.append(qt.Qobj(U[k])*evecs_temp[n])
        Ukevecs[k] = scan

        evecs[k] = evecs_temp

        evals[k] = evals_temp

    return evals, evecs, Ukevecs