import numpy as np
import qutip as qt

sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()

DD_protocols = {'Free': [ 0 *qt.identity(2)],
                'UDDy9': [sy ,sy ,sy ,sy ,sy ,sy ,sy ,sy ,sy],
                'Chiral': [sy],
                'SH': [sy ,-sy],
                'sCPMG': [sy ,sy ,-sy ,-sy],
                'EDD': [sx ,sy ,sx ,sy ,sy ,sx ,sy ,sx],
                'XY4': [sy ,sx ,sy ,sx],
                'RGA8a': [sx ,-sy ,sx ,-sy ,sy ,-sx ,sy ,-sx],
                'RGA4' :[-sy ,sx ,-sy ,sx],
                'RGA4p' :[-sy ,-sx ,-sy ,-sx],
                'SE': [sx ,sy ,sx ,sy ,sy ,sx ,sy ,sx,\
                        -sx ,-sy ,-sx ,-sy ,-sy ,-sx ,-sy ,-sx]
                  }

def Pulse_Gaussian(t, args):
    """

    :param t:
    :param args:
    :return:
    """

    # return args['amplitude'] * (1 / (np.sqrt(2 * np.pi))) * np.e ** \
    # (-((t - np.floor(t / args['period']) * args['period']) ** 2) / (
    #         2 * args['width'] ** 2))

    return args['amplitude'] * (1 / (np.sqrt(2 * np.pi))) * np.e ** \
    (-((t - np.floor(t / args['period']) * args['period']) ** 2) / (
            2 * args['width'] ** 2))
