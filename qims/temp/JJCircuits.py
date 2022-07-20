


import numpy as np


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