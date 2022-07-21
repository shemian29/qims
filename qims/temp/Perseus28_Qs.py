import numpy as np
import PerseusQuantumScars as qs
import os
import sys


Nx = 28
rmax = 2
QList = np.array(np.loadtxt('QList'+str(Nx)+'.txt'),dtype=int)
Qvec = QList[int(sys.argv[1])][0]

loaded = np.load('Perseus_vbsQ_Nx_'+str(Nx)+'_Q_'+str(Qvec)+'.npz',allow_pickle=True)


vbsQ = loaded['vbsQ']
ChOps = qs.OperatorSet(rmax)
OpSel = ChOps[QList[int(sys.argv[1])][1]]


# Qab03, Qab12 = qs.BuildQMatrices_Resolved(vbsQ, Nx, OpSel[0])
# if len(OpSel)==2:
#     Qab03inv, Qab12inv = qs.BuildQMatrices_Resolved(vbsQ, Nx, OpSel[1])
#     Qab03s = np.array([Qab03[0]+Qab03inv[0],Qab03[1]+Qab03inv[1]], dtype = "object")
#     Qab12s = np.array([Qab12[0]+Qab12inv[0],Qab12[1]+Qab12inv[1]], dtype = "object")
# else:
#     Qab03s = np.array(Qab03,dtype = "object")
#     Qab12s = np.array(Qab12,dtype = "object")


Qab03, Qab12 = qs.BuildQMatrices_Resolved(vbsQ, Nx, OpSel[0])
if len(OpSel)==2:
    Qab03inv, Qab12inv = qs.BuildQMatrices_Resolved(vbsQ, Nx, OpSel[1])
    Qab03m = Qab03[0]+Qab03inv[0]
    Qab03p = Qab03[1]+Qab03inv[1]
    Qab12m = Qab12[0]+Qab12inv[0]
    Qab12p = Qab12[1]+Qab12inv[1]
else:
    Qab03m = Qab03[0]
    Qab03p = Qab03[1]
    Qab12m = Qab12[0]
    Qab12p = Qab12[1]

    
np.savez_compressed('Perseus_Qs_Nx_'+str(Nx)+'_Q_'+str(Qvec)+'_OpSel_'+str(QList[int(sys.argv[1])][1]), 
                        Qab03p = Qab03p,
                        Qab03m = Qab03m,
                        Qab12p = Qab12p,
                        Qab12m = Qab12m,
                        ChOps = ChOps,
                        OpSel = OpSel,
                        QList = QList)