


def Kick_X(t, args):
    """

    :param t:
    :param args:
    :return:
    """
    xloc = args['period']*(np.where(args['U_total'] == "X")[0]/len(args['U_total']))
    sm = 0
    for tx in xloc:
        sm = sm + args['amplitude'] * (1 / (np.sqrt(2 * np.pi))) * np.e ** \
    (-((t - tx - np.floor(t / args['period']) * args['period']) ** 2) / (
            2 * args['width'] ** 2))
    return sm


def Kick_Y(t, args):
    """

    :param t:
    :param args:
    :return:
    """
    xloc = args['period']*(np.where(args['U_total'] == "Y")[0]/len(args['U_total']))
    sm = 0
    for tx in xloc:
        sm = sm + args['amplitude'] * (1 / (np.sqrt(2 * np.pi))) * np.e ** \
    (-((t - tx - np.floor(t / args['period']) * args['period']) ** 2) / (
            2 * args['width'] ** 2))
    return sm


def Kick_Z(t, args):
    """

    :param t:
    :param args:
    :return:
    """
    xloc = args['period']*(np.where(args['U_total'] == "Z")[0]/len(args['U_total']))
    sm = 0
    for tx in xloc:
        sm = sm + args['amplitude'] * (1 / (np.sqrt(2 * np.pi))) * np.e ** \
    (-((t - tx - np.floor(t / args['period']) * args['period']) ** 2) / (
            2 * args['width'] ** 2))
    return sm

def Pulse(t, args):
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