
import numpy as np
from numpy.linalg import multi_dot
from scipy.linalg import null_space
import scipy as sc
import matplotlib.pyplot as plt
import itertools
from timeit import timeit
import datetime


#------------------------------------------------------------------------------------------------------------

def QubitForm(State, Size):
    return np.frombuffer(np.binary_repr(State).zfill(Size).encode('ASCII'),\
                         dtype='S1').astype(int)


def IndexForm(v):
    return np.dot(v,(2**(np.arange(len(v),0,-1)-1)))


def T1(v, Size):
    return IndexForm(np.roll(QubitForm(v, Size), -1))


def Translate(vecQ):
    a = np.array([T1(vecQ[1][r], Size) for r in range(len(vecQ[1]))])
    return [vecQ[0], a]

def Invert(v, Size):
    return IndexForm(QubitForm(v, Size)[::-1])


def Inversion(vecQ, Size):
    a = np.array([Invert(vecQ[1][r], Size) for r in range(len(vecQ[1]))])
    return [vecQ[0], a]


def overlap(v1,v2):
    ltemp, il, ir = np.intersect1d(v1[1], v2[1], return_indices=True,assume_unique=True)

    return np.dot(np.conjugate(v1[0][il]),v2[0][ir])



#------------------------------------------------------------------------------------------------------------


def GenerateBasis(Size):
    Basis = []
    for r in range(2**Size):
        st = QubitForm(r, Size)
        inds = np.dot(st,np.roll(st,1))
        if(inds==0.0):
            Basis.append(r)
    return Basis


#------------------------------------------------------------------------------------------------------------

#Generates basis for two momenta

def GenerateSymmetricBasis(Q, Size,Basis):
    mms = np.array([n/Size for n in range(Size)])
    vbs = [[[] for r in range(Size)],[[] for r in range(Size)]]
    lst = Basis
    it = 1
    while len(lst)>0:
        #Choose first index in list
        a0 = lst[0]
        aux1 = [a0]

        #Apply translation recurrently until the cycle closes
        while len(np.where(aux1==a0)[0])<2:
            aux1 = np.append(aux1, T1(aux1[-1],Size))
        N0 = len(aux1)-1
        bas = aux1[0:N0]

        #Define chiral index
        ChiralIndex = int((np.prod(2*QubitForm(bas[0], Size)-1)+1)/2)

        #Define amplitudes for corresponding eigenstates
        seq = np.arange(0,len(bas))


        [vbs[ChiralIndex][np.where(mms == n/N0)[0][0]]\
         .append([(1/np.sqrt(N0))*np.exp(-1j*np.array((2*np.pi*n/N0)*seq)),bas] ) for n in range(N0)]

        it = it + 1

        lst = np.setdiff1d(lst, bas)



    vbsQ = [[[ [],[]  ],[ [],[]  ]],
            [[ [],[]  ],[ [],[]  ]]]

    dm = 0
    if Q != 0 and Q != int(Size/2) and Q != Size/4:
        Qp = int(np.abs(Size/2-Q))
        

        for qind in [0,1]:
            q = [Q,Qp][qind]
                
            for chiral in [0,1]:
                for m in range(len(vbs[chiral][q])):

                    Chl = vbs[chiral][q][m]
                    Chr = Inversion(vbs[chiral][q][m], Size)

                    ltemp = np.sort(np.unique(np.concatenate((Chl[1], Chr[1]))))

                    vl = np.zeros(len(ltemp), dtype = complex)
                    vl[[np.where(ltemp==Chl[1][r])[0][0] for r in range(len(Chl[1]))]] = Chl[0]

                    vr = np.zeros(len(ltemp), dtype = complex)
                    vr[[np.where(ltemp==Chr[1][r])[0][0] for r in range(len(Chr[1]))]] = Chr[0]

                    vbsQ[1][chiral][qind].append([(1/np.sqrt(2))*(vl+vr),ltemp])
                    dm = dm + 1
                    vbsQ[0][chiral][qind].append([(1/np.sqrt(2))*(vl-vr),ltemp])
                    dm = dm + 1


    if Q == Size/4:
        q = Q
        qind = 0
        for chiral in [0,1]:
            for m in range(len(vbs[chiral][q])):

                Chl = vbs[chiral][q][m]
                Chr = Inversion(vbs[chiral][q][m], Size)

                ltemp = np.sort(np.unique(np.concatenate((Chl[1], Chr[1]))))

                vl = np.zeros(len(ltemp), dtype = complex)
                vl[[np.where(ltemp==Chl[1][r])[0][0] for r in range(len(Chl[1]))]] = Chl[0]

                vr = np.zeros(len(ltemp), dtype = complex)
                vr[[np.where(ltemp==Chr[1][r])[0][0] for r in range(len(Chr[1]))]] = Chr[0]

                vbsQ[1][chiral][qind].append([(1/np.sqrt(2))*(vl+vr),ltemp])
                dm = dm + 1
                vbsQ[0][chiral][qind].append([(1/np.sqrt(2))*(vl-vr),ltemp])
                dm = dm + 1        


#     print('0,pi momenta')
    if Q == 0 or Q == int(Size/2):
        Qp = int(np.abs(Size/2-Q))
        
        for q in [Q,Qp]:
            if q == Q:
                qind = 0
            elif q == Qp:
                qind = 1
                
                
            for chiral in [0,1]:

                for m in range(len(vbs[chiral][q])):
                    Chl = vbs[chiral][q][m]
                    Chr = Inversion(vbs[chiral][q][m], Size)


                    ovrl0 = overlap(Chl,Chr)
                    ovrl = np.round(np.real(ovrl0),decimals = 4)
                    if ovrl == 1.0:
                        vbsQ[1][chiral][qind].append(Chl)
                        dm = dm + 1
                    elif ovrl == -1.0:
                        vbsQ[0][chiral][qind].append(Chl)
                        dm = dm + 1
                    else:
                        ltemp = np.sort(np.unique(np.concatenate((Chl[1], Chr[1]))))

                        vl = np.zeros(len(ltemp), dtype = complex)
                        vl[[np.where(ltemp==Chl[1][r])[0][0] for r in range(len(Chl[1]))]] = Chl[0]

                        vr = np.zeros(len(ltemp), dtype = complex)
                        vr[[np.where(ltemp==Chr[1][r])[0][0] for r in range(len(Chr[1]))]] = Chr[0]

                        vp = [(1/np.sqrt(2))*(vl+vr),ltemp]
                        vm = [(1/np.sqrt(2))*(vl-vr),ltemp]


                        riter = 0
                        tempaux = 0.0
                        while tempaux == 0.0 and riter < len(vbsQ[1][chiral][qind]):
                            tempaux = np.abs(overlap(vbsQ[1][chiral][qind][riter],vp))
                            riter = riter + 1
                        if tempaux == 0.0:
                            vbsQ[1][chiral][qind].append(vp)
                            dm = dm + 1
                            
                        riter = 0
                        tempaux = 0.0
                        while tempaux == 0.0 and riter < len(vbsQ[0][chiral][qind]):
                            tempaux = np.abs(overlap(vbsQ[0][chiral][qind][riter],vm))
                            riter = riter + 1
                        if tempaux == 0.0:
                            vbsQ[0][chiral][qind].append(vm)  
                            dm = dm + 1
            
    return vbsQ, dm




def CheckInversionSymmetry(Q, SymBas, Size):
    scan = []
    Qp = int(np.abs(Size/2-Q))

        
    for q in [Q,Qp]:
        
        if q == Q:
            qind = 0
        elif q == Qp:
            qind = 1
            
        if q == Size/4:
            qind = 0
            
        for chiral in [0,1]:
            for lmbd in [0,1]:
                for m in range(len(SymBas[lmbd][chiral][qind])):
#                     print(SymBas[lmbd][chiral][qind])
                    scan.append(np.sum((2*lmbd-1)-np.real(np.round(overlap(SymBas[lmbd][chiral][qind][m],Inversion(SymBas[lmbd][chiral][qind][m],Size)),decimals = 3))))
    if(np.sum(np.abs(scan))==0.0): 
        tmp = 'Inversion symmetric!'
    else: 
        tmp = "NOT inversion symmetric WARNING!"
    print(tmp)

        
    
def CheckOrthonormality(Q, SymBas, Size, Basis):
    scan1 = []
    scan2 = []
    
    sm = 0
    if Q == Size/4:
        qlist = [0]
    else:
        qlist = [0,1]
    for q in qlist:
        for chiral in [0,1]:
            for lmbd in [0,1]:
                sm = sm + len(SymBas[lmbd][chiral][q])
                for m in range(len(SymBas[lmbd][chiral][q])):
                    
                    for qp in qlist:
                        for chiralp in [0,1]:
                            for lmbdp in [0,1]:
                                for mp in range(len(SymBas[lmbdp][chiralp][qp])):
                    
                                    
                                    if(q==qp and m==mp and lmbd==lmbdp and chiral == chiralp):
                                        
                                        scan1.append(np.real(np.round(overlap(SymBas[lmbdp][chiralp][qp][mp],SymBas[lmbd][chiral][q][m]),decimals = 5)))
                                    else: 
                                        scan2.append(np.real(np.round(overlap(SymBas[lmbdp][chiralp][qp][mp],SymBas[lmbd][chiral][q][m]),decimals = 5)))

    if([np.sum(np.abs(scan1))-sm,np.sum(np.abs(scan2))]==[0.0,0.0]): 
        tmp = 'Orthonormal!'
    else: 
        tmp = "NOT orthonormal WARNING!"
    print(tmp)

    
#------------------------------------------------------------------------------------------------------------


def sw(n):
    return -n

def ElimSzP(a):   
    if len(a)==1:
        ax = len(np.intersect1d(a[0][0],a[0][1]))==0
    elif len(a)==2:
        ax = (len(np.intersect1d(a[0][0],a[0][1]))==0) and (len(np.intersect1d(a[1][0],a[1][1]))==0)
    return ax


def MemberQ(elm,SelOps):

    ax = 0
    for r in range(len(SelOps)):
        if len(SelOps[r])==2 and len(elm) == 2:
            if elm[0] == SelOps[r][0] or elm[0] == SelOps[r][1] or elm[1] == SelOps[r][0] or elm[1] == SelOps[r][1]:
                ax = ax +1
        elif len(SelOps[r])==1 and len(elm) == 1:
            if elm[0] == SelOps[r][0] or elm[0] == SelOps[r][0]:
                ax = ax +1            
    ax = ax>0 
    return ax

def Symmetrize(elm):

    tst = elm
    tst1a = list(map(sw,tst[0]))
    tst1a.sort()
    tst1b = list(map(sw,tst[1]))
    tst1b.sort()
    tst1 = [tst1a,tst1b]
    if tst ==tst1:
        ax = [tst]
    else:
        ax = [tst,tst1]

    return ax

def OperatorSet(rmax):
        
    rng = [r for r in range(-(rmax+1),(rmax+1))]
    rng.remove(0)
    rng.remove(-1)
    rng.remove(1)
    rng.remove(-rmax-1)

    inds = []
    for r in range(rmax+2):
        tmp = list(map(list,list(itertools.combinations(rng, r))))
        for l in range(len(tmp)):
            inds.append(tmp[l])

    opers = []
    for r in range(len(inds)):
        for rp in range(len(inds)):

            opers.append([inds[r],inds[rp]])

    

    ChOps = [[opers[0]]]

    for cf in opers:

        symcf = Symmetrize(cf)

        if (not MemberQ(symcf, ChOps)) and ElimSzP(symcf):
            ChOps.append(symcf)


    for j in range(len(ChOps)):
        for s in range(len(ChOps[j])):
            ChOps[j][s] = [np.array(ChOps[j][s][0]),np.array(ChOps[j][s][1])]    
    return ChOps
    
    
    
def sp1(i, r, szs, ps, Size):
    "Find the state that is produced when acting\
    with the r-th raising operator on the i-th state"
    
    state = QubitForm(i, Size)
    ts = []
    if state[r] == 0 and state[(r+1)%Size] == 0 and state[(r-1)%Size] == 0:
        state[r] = 1
        if len(szs)!=0 and len(ps)!=0:
            ts = [IndexForm(state),  np.prod(2*state[(szs + r)% Size]-1)*np.prod(1-state[(ps + r)%Size])]
        elif len(szs)!=0 and len(ps)==0:
            ts = [IndexForm(state), np.prod(2*state[(szs + r)%Size]-1)]
        elif len(szs)==0 and len(ps)!=0:
            ts = [IndexForm(state), np.prod(1-state[(ps + r)%Size])]
        else:
            ts = [IndexForm(state), 1.0]
                  
    return ts

def sp0(i, r, szs, ps, Size):
    "Find the state that is produced when acting\
    with the r-th raising operator on the i-th state"
    
    state = QubitForm(i, Size)
    ts = []
                
    if state[r] == 1 and state[(r+1)%Size] == 0 and state[(r-1)%Size] == 0:
        state[r] = 0
        if len(szs)!=0 and len(ps)!=0:
            ts = [IndexForm(state), np.prod(2*state[(szs + r)% Size]-1)*np.prod(1-state[(ps + r)%Size])]
        elif len(szs)!=0 and len(ps)==0:
            ts = [IndexForm(state), np.prod(2*state[(szs + r)%Size]-1)]
        elif len(szs)==0 and len(ps)!=0:
            ts = [IndexForm(state), np.prod(1-state[(ps + r)%Size])]
        else:
            ts = [IndexForm(state), 1.0]        
    return ts


#------------------------------------------------------------------------------------------------------------


def SP(v1s,v2, szs, ps, Size):
    sm = 0
    for R in range(0,Size,2):
        sv = [sp0(j,R, szs, ps, Size) for j in v2[1]]
        output = np.array([[idx, element[0],element[1]] for idx, element in enumerate(sv) if len(element)!=0])
        if len(output) != 0:
            SPv2 = [v2[0][output[:,0].astype(int)]*(output[:,2]),output[:,1]]
            sm += np.array([overlap(v1,SPv2) for v1 in v1s])
            
    for R in range(1,Size,2):
        sv = [sp1(j,R, szs, ps, Size) for j in v2[1]]

        output = np.array([[idx, element[0],element[1]] for idx, element in enumerate(sv) if len(element)!=0])

        if len(output) != 0:
            SPv2 = [v2[0][output[:,0].astype(int)]*(output[:,2]),output[:,1]]
            sm += np.array([overlap(v1,SPv2) for v1 in v1s])
        
    return sm


#------------------------------------------------------------------------------------------------------------



def BuildQMatrices(Q, SymBas, Size, SOps):

    Qab03s = [[] for r in range(len(SOps))]
    Qab12s = [[] for r in range(len(SOps))]


    for j in range(len(SOps)):

        for s in range(len(SOps[j])):

            Qab03 = []
            Qab12 = []

        # ---------------------------------------------------------------------------
            for lmbd in [0,1]:
                if lmbd == 0 and Q<Size/4:
                    qind = 1
                    qindp = 0
                elif lmbd == 1 and Q<Size/4:
                    qind = 0
                    qindp = 1
                elif lmbd == 0 and Q>Size/4:
                    qind = 0
                    qindp = 1                    
                elif lmbd == 1 and Q>Size/4:
                    qind = 1
                    qindp = 0                    
                elif Q == Size/4:
                    qind = 0
                    qindp = 0
                    
                vecL = SymBas[lmbd][0][qind]
                vecR = SymBas[lmbd][1][qind]
                print('     ',(len(vecL),len(vecR)))
                Qab03.append(np.array([SP(vecL, vcR, SOps[j][s][0], SOps[j][s][1], Size) for vcR in vecR]).T)
                print('      ',Qab03[-1].shape)
                print('')
        # ---------------------------------------------------------------------------   
        

                vecR = SymBas[int(np.abs(lmbd-1))][1][qindp]
                Qab12.append(np.array([SP(vecL, vcR, SOps[j][s][0], SOps[j][s][1], Size) for vcR in vecR]).T)

         # ---------------------------------------------------------------------------             
            Qab03s[j].append(Qab03)
            Qab12s[j].append(Qab12)

            
    return Qab03s, Qab12s

#------------------------------------------------------------------------------------------------------------


def SQab03(lmbd, Qab03s, parameters):
    
    for j in range(len(Qab03s)):
        for s in range(len(Qab03s[j])):
            if j==0:
                tmp = parameters[j]*Qab03s[0][0][lmbd]
            else:
                tmp = tmp + parameters[j]*Qab03s[j][s][lmbd]
    
    return tmp

#------------------------------------------------------------------------------------------------------------

def SQab12(lmbd, Qab12s, parameters):
    
    
    for j in range(len(Qab12s)):
        for s in range(len(Qab12s[j])):
            if j==0:
                tmp = parameters[j]*Qab12s[0][0][lmbd]
            else:
                tmp = tmp + parameters[j]*Qab12s[j][s][lmbd]
    
    return tmp

#------------------------------------------------------------------------------------------------------------

def SQss12(chiral, Qab03, Qab12):
    
    if chiral == 0:
        tmp = np.dot(Qab03[1],Qab12[0].conj().T)+np.dot(Qab12[1],Qab03[0].conj().T)
    else:
        tmp = -np.dot(Qab03[1].conj().T,Qab12[1])-np.dot(Qab12[0].conj().T,Qab03[0])
        
    return tmp



#------------------------------------------------------------------------------------------------------------

def BuildQab(Qab03s,Qab12s, parameters):
    
    Qab03 = []
    Qab12 = []
    

    for lmbd in [0,1]:

        Qab03.append(SQab03(lmbd,Qab03s,parameters))
        Qab12.append(SQab12(lmbd,Qab12s,parameters))
        

    return Qab03, Qab12

#------------------------------------------------------------------------------------------------------------

def BuildQss(Qab03,Qab12):  
    
    Qss12 = []
    
    for chiral in [0,1]:
        Qss12.append(SQss12(chiral, Qab03,Qab12))

    return Qss12
        
#------------------------------------------------------------------------------------------------------------
    
    
def MaxSvals(Qab03, Qab12, Qss12, Size):
    
    e1 = np.concatenate([np.abs(np.linalg.eigvalsh(np.dot(Qab03[lmbd],Qab03[lmbd].conj().T))) for lmbd in [0,1]])
    e2 = np.concatenate([np.abs(np.linalg.eigvalsh(np.dot(Qab03[lmbd].conj().T,Qab03[lmbd]))) for lmbd in [0,1]])
    IMaxSx = Size/(2*np.max(np.sqrt(np.concatenate([e1,e2]))))

    e1 = np.concatenate([np.abs(np.linalg.eigvalsh(np.dot(Qab12[lmbd],Qab12[lmbd].conj().T))) for lmbd in [0,1]])
    e2 = np.concatenate([np.abs(np.linalg.eigvalsh(np.dot(Qab12[lmbd].conj().T,Qab12[lmbd]))) for lmbd in [0,1]])
    IMaxSy = Size/(2*np.max(np.sqrt(np.concatenate([e1,e2]))))

    e1 = np.concatenate([np.abs(np.linalg.eigvalsh(np.dot(Qss12[lmbd],Qss12[lmbd].conj().T))) for lmbd in [0,1]])
    e2 = np.concatenate([np.abs(np.linalg.eigvalsh(np.dot(Qss12[lmbd].conj().T,Qss12[lmbd]))) for lmbd in [0,1]])
    IMaxSz = Size/(2*np.max(np.sqrt(np.concatenate([e1,e2]))))
    
    return [IMaxSx, IMaxSy, IMaxSz]
    

#------------------------------------------------------------------------------------------------------------
    
def RZb(lmbd,Qab03,Qab12,Qss12, MaxSy, MaxSz):
    vs = null_space(Qab03[lmbd])
    if len(vs)==0:
        res = []
    else:
        v1 = multi_dot([np.conjugate(vs).T,Qab12[np.abs(lmbd-1)].conj().T,Qab12[np.abs(lmbd-1)],vs])
        
        if lmbd == 1:
            v2 = multi_dot([np.conjugate(vs).T,Qss12[1],Qss12[1].conj().T,vs])
        else: 
            v2 = multi_dot([np.conjugate(vs).T,Qss12[1].conj().T,Qss12[1],vs])
        eigs, res = np.linalg.eigh(MaxSy*MaxSy*v1 + MaxSz*MaxSz*v2)

        idx = np.argsort(eigs)
        eigs = eigs[idx]

        res = (res.T)[idx]
    return np.array([np.dot(res[i],vs.T) for i in range(len(res))])


def RZa(lmbd,Qab03,Qab12,Qss12, MaxSy, MaxSz):
    vs = null_space(Qab03[lmbd].conj().T)


    if len(vs)==0:
        res = []
    else:

        v1 = multi_dot([np.conjugate(vs).T,Qab12[np.abs(lmbd)],Qab12[np.abs(lmbd)].conj().T,vs])
        
        if lmbd == 1:
            v2 = multi_dot([np.conjugate(vs).T,Qss12[0],Qss12[0].conj().T,vs])
        else: 
            v2 = multi_dot([np.conjugate(vs).T,Qss12[0].conj().T,Qss12[0],vs])
        eigs, res = np.linalg.eigh(MaxSy*MaxSy*v1 + MaxSz*MaxSz*v2)


        idx = np.argsort(eigs)
        eigs = eigs[idx]

        res = (res.T)[idx]
    return np.array([np.dot(res[i],vs.T) for i in range(len(res))])

    
    
def BuildEigensystem(Qab03, Qab12, Qss12, Qab03_0, Qab12_0, Qss12_0, Nx, MaxSx, MaxSy, MaxSz):
    
    


    EV = [[],[]]

    for lmbd in [0,1]:

        edat = np.linalg.eigh(MaxSx*MaxSx*np.dot(Qab03[lmbd],Qab03[lmbd].conj().T))
        [EV[lmbd].append([edat[0][r], edat[1][:,r]]) for r in range(len(edat[0])) if edat[0][r]>0.0001]

        vrzb = RZb(lmbd,Qab03,Qab12,Qss12, MaxSy, MaxSz)
        if len(vrzb) != 0:

                [EV[lmbd].append([0,1,vrzb[r]]) for r in range(len(vrzb))]
                
        vrza = RZa(lmbd,Qab03,Qab12,Qss12, MaxSy, MaxSz)
        if len(vrza) != 0:

                [EV[lmbd].append([0,0,vrza[r]]) for r in range(len(vrza))]

    FEV = [[],[]]


    for lmbd in [0,1]:


        auxtemp = []
        eigs = []
        for m in range(len(EV[lmbd])):
            vec = EV[lmbd][m];
            nb = len(Qab03[lmbd].T)
            na = len(Qab03[lmbd])

            if len(vec)==3 and vec[1]==1:
                auxtemp.append([0,[np.zeros(na), np.sign(vec[2][0])*vec[2]]])
                eigs.append(vec[0])
            elif len(vec)==3 and vec[1]==0:
                auxtemp.append([0,[np.sign(vec[2][0])*vec[2], np.zeros(nb)]])
                eigs.append(vec[0])
            elif len(vec)==2:
                aux1 = [(np.sign(vec[1][0])/np.sqrt(2))*vec[1], \
                        (np.sign(vec[1][0])/np.sqrt(2))*\
                        (MaxSx/np.sqrt(vec[0]))*np.dot((Qab03[lmbd].conj().T),vec[1])]
                aux2 = [(np.sign(vec[1][0])/np.sqrt(2))*vec[1],\
                        -(np.sign(vec[1][0])/np.sqrt(2))*\
                        (MaxSx/np.sqrt(vec[0]))*np.dot((Qab03[lmbd].conj().T),vec[1])]
                auxtemp.append([-np.sqrt(vec[0]),aux2])
                eigs.append(-np.sqrt(vec[0]))
                auxtemp.append([np.sqrt(vec[0]),aux1])
                eigs.append(np.sqrt(vec[0]))

        eigs = np.array(eigs)
        auxtemp = [auxtemp[r] for r in eigs.argsort()]
        FEV[lmbd].append(auxtemp)

    return FEV, EV




#------------------------------------------------------------------------------------------------------------

def BuildSxyz(FEV,Qab12, Qss12, MaxSx, MaxSy, MaxSz):
    
    eigs = []
    Sx = []    
    Sy = []
    Sz = []
    Sraise = []

    vecL = FEV[1][0]

    vecR = FEV[0][0]


    UL1 = np.array([vecL[r][1][0] for r in range(len(vecL))]).conj()
    UR1 = np.array([vecR[r][1][1] for r in range(len(vecR))]).T

    UL2 = np.array([vecL[r][1][1] for r in range(len(vecL))]).conj()
    UR2 = np.array([vecR[r][1][0] for r in range(len(vecR))]).T

    Sytemp = -1j*multi_dot([UL1, Qab12[1], UR1])+1j*multi_dot([UL2, Qab12[0].conj().T, UR2])
    Sy.append(MaxSy*np.block([
        [np.zeros((len(Sytemp), len(Sytemp))),               Sytemp],
        [Sytemp.conj().T, np.zeros((len(Sytemp.conj().T), len(Sytemp.conj().T)))               ]]))
    Sy=Sy[0]

    Sztemp = -multi_dot([UL1, Qss12[0], UR2])-multi_dot([UL2, Qss12[1], UR1])
    Sz.append(MaxSz*np.block([
        [np.zeros((len(Sztemp), len(Sztemp))),               Sztemp],
        [Sztemp.conj().T, np.zeros((len(Sztemp.conj().T), len(Sztemp.conj().T)))               ]])) 
    Sz=Sz[0]
    Sraise = Sy-1j*Sz
    
    eigsL = np.array([vecL[r][0] for r in range(len(vecL))])
    eigsR = np.array([vecR[r][0] for r in range(len(vecR))])
    
    eigs.append(np.concatenate([eigsL,eigsR]))
    eigs=eigs[0]

    iem = np.where(eigs<0.)[0]
    em = eigs[np.ix_(iem)]
    iem = iem[np.ix_(np.argsort(em))]
    inds = np.ix_(np.argsort([np.real((np.dot(Sz,Sz)+np.dot(Sy,Sy))[r,r]) for r in np.where(eigs==0.)[0]]))
    ez = np.where(eigs==0.)[0][inds]
    iep = np.where(eigs>0.)[0]
    ep = eigs[np.ix_(iep)]
    iep = iep[np.ix_(np.argsort(ep))]
    inds = np.concatenate([iem,ez,iep])
    
    Ereord = np.ix_(inds)
    Ereord2 = np.ix_(inds,inds)
    
    eigs=eigs[Ereord]
    Sy = Sy[Ereord2]
    Sz = Sz[Ereord2]
    Sraise=Sraise[Ereord2]
    
    aux2 = np.zeros((len(eigs),len(eigs)))
    np.fill_diagonal(aux2,eigs)
    Sx = aux2
    
    
    return Sx, Sy,Sz, Sraise, eigs, Ereord, Ereord2
    
    
    
    
    
    
    
#------------------------------------------------------------------------------------------------------------    
    
    
def BuildTowers(Sraise, eigs, Nx):
    
        
    Towers = []

    
    lst = np.array(range(len(eigs)))
    tmpT = []
    while len(lst) > 0:
        Rs = [np.array(lst[0])]
        Ef = eigs[Rs[-1]]
        E0 = Ef-1
        vc = np.zeros((len(eigs)))
        vc[Rs[-1]] = 1
        bx = np.abs(np.dot(Sraise,vc))
        mx = np.max(bx)
        mx0 = mx

        while (Ef>E0) and (Rs[-1] in lst) and mx > 0.5*mx0:
            vc = np.zeros((len(eigs)))
            vc[Rs[-1]] = 1
            
            bx = np.abs(np.dot(Sraise,vc))

            mx = np.max(bx)


            E0 = Ef
            Rs.append(np.where(bx==mx)[0][0])
            Ef = eigs[Rs[-1]]

        Rs = np.asarray(Rs[0:len(Rs)-1])
        tmpT.append(Rs)
        lst = np.setdiff1d(lst, Rs)
    Towers.append(tmpT)
    
    return Towers[0]

#------------------------------------------------------------------------------------------------------------
        
def PlotTowers(Sy, Towers):
    plt.figure(figsize = (10,10))
    plt.imshow(np.abs((Sy)[np.ix_(np.concatenate(Towers),np.concatenate(Towers))]))

    plt.show()

    
#------------------------------------------------------------------------------------------------------------    
    
def PlotTruncTowers(Sy, Towers):
    Trunc = np.diag([1 for r in range(len(Sy)-1)], -1)+np.diag([1 for r in range(len(Sy)-1)], 1)
    
    plt.figure(figsize = (10,10))
    plt.imshow(Trunc*np.abs((Sy)[np.ix_(np.concatenate(Towers),np.concatenate(Towers))]))

    plt.show()
    
#------------------------------------------------------------------------------------------------------------    

# FEV has TWO momenta in it    
    
def ProdExpansion(qind, lmbd,ST,FEV, vbsQ, Nx):
    chiral = int((np.prod(2*QubitForm(ST, Nx)-1)+1)/2)
    aux = []
    for M in range(len(FEV[lmbd][0])):
        
        # Take M-th eigenstate to calculate overlap with ST
        evecs = FEV[lmbd][0][M]
        
        # Choose set of symmetric basis states that span the M-th eigenstate 'evecs'
        vec = vbsQ[lmbd][chiral][qind]

        sm = 0
        for m in range(len(vec)):
            aux2 = np.where(vec[m][1]==ST)[0]
            if len(aux2)!=0:
                sm += np.conjugate(evecs[1][chiral][m]*vec[m][0][aux2[0]])
        aux.append([evecs[0],sm])
    return np.array(aux)

#------------------------------------------------------------------------------------------------------------

def BuildSyzAverages(Qvec, Sy, Sz, ST, FEV,vbsQ, Nx, Ereord, tArray):
    esl,ovrsl = ProdExpansion(0,1,ST, FEV,vbsQ, Nx).T
    esr,ovrsr = ProdExpansion(1,0,ST, FEV,vbsQ, Nx).T

    ovrs = np.concatenate([ovrsl, ovrsr])[Ereord]
    es = np.concatenate([esl, esr])[Ereord]

    SyAve = []
    SzAve = []
    for t in tArray:
        ovrst = np.exp(-1j*es*t)*ovrs
    
        SyAve.append(np.real(multi_dot([np.conjugate(ovrst),Sy,ovrst])))
        SzAve.append(np.real(multi_dot([np.conjugate(ovrst),Sz,ovrst])))

    return SyAve, SzAve


#------------------------------------------------------------------------------------------------------------

def BuildSyzTruncAverages(Sy, Sz, ST, FEV,vbsQ, Nx, Ereord, tArray, Towers):
    Trunc = np.diag([1 for r in range(len(Sy)-1)], -1)+np.diag([1 for r in range(len(Sy)-1)], 1)
    esl,ovrsl = ProdExpansion(0,1,ST, FEV,vbsQ, Nx).T
#     qr = int(np.abs((Nx/2)-q))
    esr,ovrsr = ProdExpansion(1,0,ST, FEV,vbsQ, Nx).T

    ovrs = np.concatenate([ovrsl, ovrsr])[Ereord]
    es = np.concatenate([esl, esr])[Ereord]

    TruncSyAve = []
    TruncSzAve = []
    reord = np.ix_(np.concatenate(Towers))
    reord2 = np.ix_(np.concatenate(Towers),np.concatenate(Towers))
    for t in tArray:
        ovrst = np.exp(-1j*es*t)*ovrs
        ovrst = ovrst[reord]
        TruncSyAve.append(np.real(multi_dot([np.conjugate(ovrst),Trunc*(Sy[reord2]),ovrst])))
        TruncSzAve.append(np.real(multi_dot([np.conjugate(ovrst),Trunc*(Sz[reord2]),ovrst])))

    return TruncSyAve, TruncSzAve

#------------------------------------------------------------------------------------------------------------

def su2(parameters,q, Qab03s, Qab12s, Qab03s_0, Qab12s_0,Nx):
    
    prms = parameters[0:len(parameters)-1]
    
    Qab03, Qab12 = BuildQab(Qab03s,Qab12s,prms)
    Qss12 = BuildQss(Qab03,Qab12)
    
    if q == 0:
        Qab03_0, Qab12_0, Qss12_0 = Qab03, Qab12, Qss12
    else:
        Qab03_0, Qab12_0 = BuildQab(Qab03s_0,Qab12s_0,prms)
        Qss12_0 = BuildQss(Qab03_0,Qab12_0)
    [MaxSx, MaxSy, MaxSz] = MaxSvals(Qab03_0, Qab12_0, Qss12_0, Nx)
    
    FEV = BuildEigensystem(Qab03, Qab12, Qss12,\
                 Qab03_0, Qab12_0, Qss12_0, \
                          Nx, MaxSx, MaxSy, MaxSz)
    
    Sx, Sy, Sz, Sraise, eigs, Ereord, Ereord2 = \
    BuildSxyz(FEV,Qab12, Qss12, MaxSx, MaxSy, MaxSz)
    
    Towers = BuildTowers(Sraise, eigs, Nx)
    

    ws = []
    for m in range(len(Towers)):    
        aux1 = np.diff(eigs[np.ix_(Towers[m])])
        if len(aux1)>0:
            wave = np.mean(aux1)
            ws.append(np.full((len(Towers[m])), wave))
        else:
            ws.append(np.array([1.0]))
    ws = np.diag(np.concatenate(ws))

    inds = np.ix_(np.concatenate(Towers),np.concatenate(Towers))
    return np.sum(np.abs(np.dot(Sx[inds],Sraise[inds])-np.dot(Sraise[inds],Sx[inds])-parameters[-1]*np.dot(ws,Sraise[inds])))



def BuildBasisAndOperators(Nx, rmax, chk = 'No check'):
    bf = GenerateBasis(Nx)
    ChOps = OperatorSet(rmax)

    dmT = 0
    for qq in [0]:
        print('Evaluating momentum',qq)
        Qvec = qq
        vbsQ, dm = GenerateSymmetricBasis(Qvec, Nx, bf)
        
        if chk == 'Check':        
            CheckInversionSymmetry(Qvec,vbsQ, Nx)

            CheckOrthonormality(Qvec, vbsQ, Nx, bf)


        dmT = dmT + dm

        print('Operators')
        Qab03s_0, Qab12s_0 = BuildQMatrices(Qvec, vbsQ, Nx, ChOps)
        
        print('Saving')
        np.savez_compressed('Icarus_QsOps_Nx_'+str(Nx)+'_momentum_'+str(qq)+'_range_'+str(rmax), 
                                ChOps=ChOps,
                                Qab03s_0=Qab03s_0,
                                Qab12s_0=Qab12s_0)
    print()
    for qq in range(1,int(Nx/4)+1):
        print('Evaluating momentum',qq)
        Qvec = qq
        vbsQ, dm = GenerateSymmetricBasis(Qvec, Nx, bf)
        
        if chk == 'Check':        
            CheckInversionSymmetry(Qvec,vbsQ, Nx)
            CheckOrthonormality(Qvec, vbsQ, Nx, bf)
            
        dmT = dmT + dm
#         print(dm)
        print('Operators')
        Qab03s, Qab12s = BuildQMatrices(Qvec, vbsQ, Nx, ChOps)
        
        print('Saving')
        np.savez_compressed('Icarus_QsOps_Nx_'+str(Nx)+'_momentum_'+str(qq)+'_range_'+str(rmax), 
                            ChOps=ChOps,
                            Qab03s=Qab03s,
                            Qab12s=Qab12s)
        print()
        

    if len(bf)!=dmT:
        print("WRONG BASIS COUNT!!")
    else:
        print("Good setup!")
        
 

        
def QBuildBasisAndOperators(Q, Nx, rmax, bf):
    
    ChOps = OperatorSet(rmax)
    vbsQ, dmT = GenerateSymmetricBasis(Q, Nx, bf)
    Qab03s, Qab12s = BuildQMatrices(Q, vbsQ, Nx, ChOps)
    Qab03s, Qab12s = BuildQMatrices(int(Nx/2-Q), vbsQ, Nx, ChOps)
    
    np.savez_compressed('Icarus_QsOps_Nx_'+str(Nx)+'_momentum_'+str(Q)+'_range_'+str(rmax), 
                        ChOps = ChOps,
                        Qab03s = Qab03s,
                        Qab12s = Qab12s,
                        vbsQ = vbsQ,
                        dmT = dmT)
    
    np.savez_compressed('Icarus_QsOps_Nx_'+str(Nx)+'_momentum_'+str(int(Nx/2-Q))+'_range_'+str(rmax), 
                        ChOps = ChOps,
                        Qab03s = Qab03s,
                        Qab12s = Qab12s,
                        vbsQ = vbsQ,
                        dmT = dmT)
    
    


    
def FullEvolution(Nx, rmax, prms, bf,tArray):


    SyQAve = [[] for r in range(0,int(Nx/4)+1)]
    SzQAve = [[] for r in range(0,int(Nx/4)+1)]
    TruncSyQAve = [[] for r in range(0,int(Nx/4)+1)]
    TruncSzQAve = [[] for r in range(0,int(Nx/4)+1)]


    dmT = 0
    for qq in [0]:
        print('Evaluating momentum',qq)
        Qvec = qq


        loaded = np.load('Icarus_QsOps_Nx_'+str(Nx)+'_momentum_'+str(Qvec)+'_range_'+str(rmax)+'.npz',allow_pickle=True)
        vbsQ = loaded['vbsQ']    
        Qab03s_0 = loaded['Qab03s']
        Qab12s_0 = loaded['Qab12s']
        dmT = dmT + loaded['dmT']
        ChOps = OperatorSet(rmax)
        print(dmT)


        print('Q matrices')

        Qab03_0, Qab12_0 = BuildQab(Qab03s_0,Qab12s_0,prms)
        Qss12_0 = BuildQss(Qab03_0,Qab12_0)
        print()

        print('Eigensystem')
        [MaxSx, MaxSy, MaxSz] = MaxSvals(Qab03_0, Qab12_0, Qss12_0, Nx)
        FEV = BuildEigensystem(Qab03_0, Qab12_0, Qss12_0,\
                         Qab03_0, Qab12_0, Qss12_0, \
                                  Nx, MaxSx, MaxSy, MaxSz)
        print()
        print('Towers and magnetization operators')
        Sx, Sy, Sz, Sraise, eigs, Ereord, Ereord2 = BuildSxyz(FEV,Qab12_0, Qss12_0, MaxSx, MaxSy, MaxSz)
        Towers = BuildTowers(Sraise, eigs, Nx)
        print()
        print('Evolution')
        for State in bf:
            SyAve, SzAve = BuildSyzAverages(Sy, Sz, State, FEV,vbsQ, Nx, Ereord,tArray)
            TruncSyAve, TruncSzAve = BuildSyzTruncAverages(Sy, Sz, State, FEV,vbsQ, Nx, Ereord,tArray,Towers)

            SyQAve[Qvec].append(SyAve)
            SzQAve[Qvec].append(SzAve)
            TruncSyQAve[Qvec].append(TruncSyAve)
            TruncSzQAve[Qvec].append(TruncSzAve)
        print()

    for qq in range(1,int(Nx/4)+1):
        print('Evaluating momentum',qq)
        Qvec = qq


        loaded = np.load('Icarus_QsOps_Nx_'+str(Nx)+'_momentum_'+str(Qvec)+'_range_'+str(rmax)+'.npz',allow_pickle=True)
        vbsQ = loaded['vbsQ']    
        Qab03s = loaded['Qab03s']
        Qab12s = loaded['Qab12s']
        dmT = dmT + loaded['dmT']
        ChOps = OperatorSet(rmax)
        print(loaded['dmT'])


        print('Q matrices')
        prms = np.zeros(len(ChOps)+1)
        prms[0] = 1
        prms[-1] = 1    
        Qab03, Qab12 = BuildQab(Qab03s,Qab12s,prms)
        Qss12 = BuildQss(Qab03,Qab12)
        print()
        print('Eigensystem')
        [MaxSx, MaxSy, MaxSz] = MaxSvals(Qab03_0, Qab12_0, Qss12_0, Nx)
        FEV = BuildEigensystem(Qab03, Qab12, Qss12,\
                         Qab03_0, Qab12_0, Qss12_0, \
                                  Nx, MaxSx, MaxSy, MaxSz)
        print()
        print('Magnetization operators')
        Sx, Sy, Sz, Sraise, eigs, Ereord, Ereord2 = BuildSxyz(FEV,Qab12, Qss12, MaxSx, MaxSy, MaxSz)
        Towers = BuildTowers(Sraise, eigs, Nx)


        print()
        print('Evolution')
        for State in bf:
            SyAve, SzAve = BuildSyzAverages(Sy, Sz, State, FEV,vbsQ, Nx, Ereord,tArray)
            TruncSyAve, TruncSzAve = BuildSyzTruncAverages(Sy, Sz, State, FEV,vbsQ, Nx, Ereord,tArray,Towers)

            SyQAve[Qvec].append(SyAve)
            SzQAve[Qvec].append(SzAve)
            TruncSyQAve[Qvec].append(TruncSyAve)
            TruncSzQAve[Qvec].append(TruncSzAve)
            
            
#         SyAve, SzAve = BuildSyzAverages(Sy, Sz, State, FEV,vbsQ, Nx, Ereord,tArray)
#         TruncSyAve, TruncSzAve = BuildSyzTruncAverages(Sy, Sz, State, FEV,vbsQ, Nx, Ereord,tArray,Towers)


#         SyQAve.append(SyAve)
#         SzQAve.append(SzAve)
#         TruncSyQAve.append(TruncSyAve)
#         TruncSzQAve.append(TruncSzAve)
#         print()
    
    return SyQAve,SzQAve,TruncSyQAve,TruncSzQAve