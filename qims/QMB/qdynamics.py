import qutip as qt
import numpy as np
import qims as qims
from tqdm.notebook import tqdm


def SytSy_alt(Sy, bs, Nx, Ukevecs, evals):
    Corr3 = {}
    Corrrand = {}
    wscar = 1
    wmax = 5 * wscar
    dw = wscar / 20
    tmax = (2 * wmax / (2 * wmax + dw)) * (2 * np.pi / (dw))
    dt = 2 * np.pi / (2 * wmax + dw)
    tlist = np.linspace(0, tmax, int(tmax / dt) + 1)
    k_list = np.arange(0, Nx) / Nx
    wlist = np.linspace(-wmax, wmax, len(tlist))

    itmax = 500
    tmp = [qt.rand_ket(len(bs)) for n in range(itmax)]
    sts_rand = qims.npqt2qtqt(tmp)

    for k in tqdm(k_list[0:int(Nx / 2)]):
        kit = ((k * Nx + int(Nx / 2)) % Nx) / Nx

        aux_k = qims.npqt2qtqt(Ukevecs[k])
        aux_kp = qims.npqt2qtqt(Ukevecs[kit])
        Skkp = aux_k.dag() * Sy * aux_kp
        Skkpaux_k = Skkp.dag() * aux_k.dag()
        Skkpaux_kp = Skkp * aux_kp.dag()

        rand_aux_k = (sts_rand.dag()) * aux_k
        rand_aux_kp = (sts_rand.dag()) * aux_kp
        rand_Skkpaux_k = Skkp.dag() * aux_k.dag() * (sts_rand)
        rand_Skkpaux_kp = Skkp * aux_kp.dag() * (sts_rand)

        for t in tqdm(tlist):
            phase_k = qt.Qobj(np.diag(np.exp(1j * evals[k] * t)))
            phase_kp = qt.Qobj(np.diag(np.exp(-1j * evals[kit] * t)))

            phkkp = phase_k * Skkp * phase_kp
            t1 = aux_k.full()
            t2 = (phkkp * Skkpaux_k).full()
            Corr3[k, t] = np.sum(t1.T * t2, axis=0)
            t1 = aux_kp.full()
            t2 = (phkkp.dag() * Skkpaux_kp).full()
            Corr3[kit, t] = np.sum(t1.T * t2, axis=0)

            t1 = (rand_aux_k).full()
            t2 = (phkkp * rand_Skkpaux_k).full()
            Corrrand[k, t] = np.sum(t1.T * t2, axis=0)
            t1 = (rand_aux_kp).full()
            t2 = (phkkp.dag() * rand_Skkpaux_kp).full()
            Corrrand[kit, t] = np.sum(t1.T * t2, axis=0)

    return Corr3, Corrrand, tlist, wlist

def SytSy(Sy, bs, Nx, Ukevecs, evals):
    Corr3 = {}
    Corrrand = {}
    wscar = 1
    wmax = 5 * wscar
    dw = wscar / 20
    tmax = (2 * wmax / (2 * wmax + dw)) * (2 * np.pi / (dw))
    dt = 2 * np.pi / (2 * wmax + dw)
    tlist = np.linspace(0, tmax, int(tmax / dt) + 1)
    k_list = np.arange(0, Nx) / Nx
    wlist = np.linspace(-wmax, wmax, len(tlist))

    itmax = 500
    tmp = [qt.rand_ket(len(bs)) for n in range(itmax)]
    sts_rand = qims.npqt2qtqt(tmp)

    for k in tqdm(k_list[0:int(Nx / 2)]):
        kit = ((k * Nx + int(Nx / 2)) % Nx) / Nx

        kwargs = {}

        kwargs["evalsk"] = evals[k]
        kwargs["evalskit"] = evals[kit]


        kwargs["aux_k"] = qims.npqt2qtqt(Ukevecs[k])
        kwargs["aux_kp"] = qims.npqt2qtqt(Ukevecs[kit])
        kwargs["Skkp"] = kwargs["aux_k"].dag() * Sy * kwargs["aux_kp"]

        kwargs["Skkpaux_k"] = kwargs["Skkp"].dag() * kwargs["aux_k"].dag()
        kwargs["Skkpaux_kp"] = kwargs["Skkp"] * kwargs["aux_kp"].dag()
        Corr3[k] = qt.parallel_map(Corr, tlist, task_kwargs=kwargs, progress_bar=True)


        rkwargs = {}
        rkwargs["aux_k"] = (sts_rand.dag()) * kwargs["aux_k"]
        rkwargs["aux_kp"] = (sts_rand.dag()) * kwargs["aux_kp"]
        rkwargs["Skkpaux_k"] = kwargs["Skkp"].dag() * kwargs["aux_k"].dag() * (sts_rand)
        rkwargs["Skkpaux_kp"] = kwargs["Skkp"] * kwargs["aux_kp"].dag() * (sts_rand)
        Corrrand[k] = qt.parallel_map(Corr, tlist, task_kwargs=kwargs, progress_bar=True)


        # for t in tqdm(tlist):
        #     phase_k = qt.Qobj(np.diag(np.exp(1j * evals[k] * t)))
        #     phase_kp = qt.Qobj(np.diag(np.exp(-1j * evals[kit] * t)))
        #
        #     phkkp = phase_k * Skkp * phase_kp

        #     t1 = aux_k.full()
        #     t2 = (phkkp * Skkpaux_k).full()
        #     Corr3[k, t] = np.sum(t1.T * t2, axis=0)
        #     t1 = aux_kp.full()
        #     t2 = (phkkp.dag() * Skkpaux_kp).full()
        #     Corr3[kit, t] = np.sum(t1.T * t2, axis=0)
        #
        #     t1 = (rand_aux_k).full()
        #     t2 = (phkkp * rand_Skkpaux_k).full()
        #     Corrrand[k, t] = np.sum(t1.T * t2, axis=0)
        #     t1 = (rand_aux_kp).full()
        #     t2 = (phkkp.dag() * rand_Skkpaux_kp).full()
        #     Corrrand[kit, t] = np.sum(t1.T * t2, axis=0)

    return Corr3, Corrrand, tlist, wlist


def Corr(t, **kwargs):
    phase_k = qt.Qobj(np.diag(np.exp(1j * kwargs["evalsk"] * t)))
    phase_kp = qt.Qobj(np.diag(np.exp(-1j * kwargs["evalskit"] * t)))

    phkkp = phase_k * kwargs["Skkp"] * phase_kp
    t1 = kwargs["aux_k"].full()
    t2 = (phkkp * kwargs["Skkpaux_k"]).full()
    Corr3k = np.sum(t1.T * t2, axis=0)
    t1 = kwargs["aux_kp"].full()
    t2 = (phkkp.dag() * kwargs["Skkpaux_kp"]).full()
    Corr3kit = np.sum(t1.T * t2, axis=0)
    return [Corr3k, Corr3kit]

