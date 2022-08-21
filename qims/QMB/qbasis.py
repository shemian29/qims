from numpy import array, arange, dot, roll
from tqdm.notebook import tqdm
import qutip as qt


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
