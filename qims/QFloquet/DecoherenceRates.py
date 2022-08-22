





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


sm = 0
for m in tqdm(range(-int(len(tlist)/2),int(len(tlist)/2))):
    if m ==0:
        sm = sm + 4*Af*2*np.abs(flz(0))
    if m != 0:
        sm = sm + 2*Sf(m*(2*np.pi/(prms["period"]*(10**-9))))*(np.abs(flz(m))**2)