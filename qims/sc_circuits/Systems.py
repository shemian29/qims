

import numpy as np
import scqubits as scq
import qutip as qt

sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()

def qubit_hamiltonian(prms):
    """
        a
    :param qb:
    :return:
    """
    if prms["h0"] == 'fluxonium':
        fluxonium = scq.Fluxonium(EJ=4,
                                  EC=0.5,
                                  EL=1.3,
                                  flux=0.5,
                                  cutoff=prms["cutoff"])

        # Setup of fluxonium qubit
        evals, evecs = fluxonium.eigensys()
        gst = qt.Qobj(evecs.T[0])
        est = qt.Qobj(evecs.T[1])
        Delta = evals[1] - evals[0]

        ph_ext = prms["flux"]
        B = 2 * 2 * np.pi * (ph_ext - 0.5) * fluxonium.EL * np.abs(
            (est.dag() * qt.Qobj(fluxonium.phi_operator()) * gst).full()[0, 0])
        H0 =( (Delta / 2) * sx + (B / 2) * sz)

    return 2*np.pi*H0





def JJCircuit(EJ,EJb,ECJ,ECJb,N):

    if N == 4:
        JJcirc_yaml = """#JJ circuit
        branches:
        - ["JJ", 0, 1, EJb="""+str(EJb)+""",ECJb = """+str(ECJb)+"""]
        - ["JJ", 1, 2, EJ="""+str(EJ)+""",ECJ = """+str(ECJ)+"""]
        - ["JJ", 2, 3, EJ,ECJ]
        - ["JJ", 3, 4, EJ ,ECJ]
        - ["JJ", 4, 0, EJ ,ECJ]
        """
        trans_mat = np.array([[1, 1, 1, 1],
                              [0, 1, 1, 1],
                              [0, 0, 1, 1],
                              [0, 0, 0, 1]]) * (-1)

    if N == 3:
        JJcirc_yaml = """#JJ circuit
        branches:
        - ["JJ", 0, 1, EJb="""+str(EJb)+""",ECJb = """+str(ECJb)+"""]
        - ["JJ", 1, 2, EJ="""+str(EJ)+""",ECJ = """+str(ECJ)+"""]
        - ["JJ", 2, 3, EJ,ECJ]
        - ["JJ", 3, 0, EJ ,ECJ]
        """
        trans_mat = np.array([[1, 1, 1],
                              [0, 1, 1],
                              [0, 0, 1]]) * (-1)
    elif N == 2:
        JJcirc_yaml = """# JJ circuit
        branches:
        - ["JJ", 0, 1, EJb="""+str(EJb)+""",ECJb = """+str(ECJb)+"""]
        - ["JJ", 1, 2, EJ="""+str(EJ)+""",ECJ = """+str(ECJ)+"""]
        - ["JJ", 2, 0, EJ,ECJ]
        """
        trans_mat = np.array([[1, 1],
                              [0, 1]]) * (-1)
    elif N == 1:
        JJcirc_yaml = """# JJ circuit
        branches:
        - ["JJ", 0, 1, EJb="""+str(EJb)+""",ECJb = """+str(ECJb)+"""]
        - ["JJ", 1, 0, EJ="""+str(EJ)+""",ECJ = """+str(ECJ)+"""]
        """
        trans_mat = np.array([[1]]) * (-1)




    return JJcirc_yaml, trans_mat