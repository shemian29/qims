

import numpy as np
from tqdm import tqdm
import qutip as qt


sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()

# Parameters from Floquet qubit paper
hbar = 6.62*(10**(-34))/(2*np.pi)
hplanck = 6.62*(10**(-34))
TandC = 1.1*(10**(-6))
df = 1.8*(10**(-6))
phige = 2
EC = 0.5*(10**9)*hplanck
EL = 1.3*(10**9)*hplanck
Ad = (np.pi**2)*TandC*phige**2/EC
Af = 2*np.pi*df*EL*np.abs(phige)

def Sf(w):
    return ((Af**2)/(hbar**2))*np.abs(((2*np.pi)/w))


def filter_coefficients(f_modes_table_t, m, tlist, prms):
    sm = 0
    for i in range(len(f_modes_table_t)):
        evecs_t = f_modes_table_t[i]
        sm = sm +np.exp(1j*m*tlist[i]*2*np.pi/prms["period"])*((sz/2)*(evecs_t[0]*evecs_t[0].dag()-evecs_t[1]*evecs_t[1].dag())).tr()
    return np.mean(np.diff(tlist))*sm/prms["period"]

def dephasing_rate(f_modes_table_t, prms, tlist):
    sm = 0
    for m in tqdm(range(-int(len(tlist)/2),int(len(tlist)/2))):
        if m ==0:
            sm = sm + 4*Af*2*np.abs(filter_coefficients(f_modes_table_t, 0, tlist, prms))
        if m != 0:
            sm = sm + 2*Sf(m*(2*np.pi/(prms["period"]*(10**-9))))*(np.abs(filter_coefficients(f_modes_table_t, m, tlist, prms))**2)
    return sm