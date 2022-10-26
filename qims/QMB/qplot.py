from matplotlib import pyplot as plt
import numpy as np


def PlotTowers(OP, kevec):

    plt.imshow(np.abs(((kevec.dag() * (OP) * kevec)).full()))


def PlotS2(evals,Ukevecs,size):
    k_list = np.arange(0, size) / size
    for k in k_list:
        tmp = np.array([[evals[k][n], \
                         ((Ukevecs[k][n]).dag() * S2_alt * Ukevecs[k][n]).full()[0, 0]] for n in
                        range(Hs[k].shape[0])])
        plt.plot(np.real(tmp.T[0]), (np.sqrt(4 * np.abs(tmp.T[1]) + 1) - 1) / 2, '.')


def PlotZ2overlap():
    k_list = np.arange(0, size) / size
    for k in k_list:
        tmp = np.array([[evals[k][n], np.abs((U[k] * evecs[k][n]).full().T[0][-1])] for n in range(Hs[k].shape[0])])
        plt.plot(tmp.T[0], tmp.T[1], '.')
        plt.yscale('log')
        plt.ylim([0.0001, 1])

    plt.show()


def PlotEntropy():
    k_list = np.arange(0, size) / size
    for k in k_list:
        for n in range(Hs[k].shape[0]):
            plt.plot([evals[k][n] for n in range(Hs[k].shape[0])], \
                     [np.array(S[k, n, int(Nx / 2)]) for n in range(Hs[k].shape[0])], '.');

    plt.show()