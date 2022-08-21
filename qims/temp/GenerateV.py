import qutip as qt
from qims.temp import cQED as qc
import sys




N = int(sys.argv[1])
Ncut = int(sys.argv[2])
    
print("Starting: "+str([N,Ncut]))

V = qc.ChargeToTranslation(N,Ncut)

qt.qsave(V, 'ChargeToTranslation_(N,Ncut)_'+str((N,Ncut)))