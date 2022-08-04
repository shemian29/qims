import numpy as np
import qutip as qt

sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()

DD_protocols = {'Free': [0 * qt.identity(2)],
                'UDDy9': [sy, sy, sy, sy, sy, sy, sy, sy, sy],
                'Chiral': [sy],
                'SH': [sy, -sy],
                'sCPMG': [sy, sy, -sy, -sy],
                'EDD': [sx, sy, sx, sy, sy, sx, sy, sx],
                'XY4': [sy, sx, sy, sx],
                'RGA8a': [sx, -sy, sx, -sy, sy, -sx, sy, -sx],
                'RGA4': [-sy, sx, -sy, sx],
                'RGA4p': [-sy, -sx, -sy, -sx],
                'SE': [sx, sy, sx, sy, sy, sx, sy, sx, \
                       -sx, -sy, -sx, -sy, -sy, -sx, -sy, -sx]
                }



def drive_sx(t, prms):
    sm = 0
    for tx in prms["sx"]:
        drive_prms = prms["drive_prms"]
        t0 = drive_prms["tlocs"][tx] * prms["period"]
        sm = sm + drive_prms['amplitudes'][tx]* np.cos(drive_prms["frequencies"][tx]*t+drive_prms["phases"][tx]) * (1 / (np.sqrt(2 * np.pi))) \
             * np.exp(
            -((t - t0 - np.floor(t / prms['period']) * prms['period']) ** 2) / (2 * drive_prms['widths'][tx] ** 2))

    return sm

def drive_sy(t, prms):
    sm = 0
    for ty in prms["sy"]:
        drive_prms = prms["drive_prms"]
        t0 = drive_prms["tlocs"][ty] * prms["period"]
        sm = sm + drive_prms['amplitudes'][ty]* np.cos(drive_prms["frequencies"][ty]*t+drive_prms["phases"][ty])  * (1 / (np.sqrt(2 * np.pi))) \
             * np.exp(
            -((t - t0 - np.floor(t / prms['period']) * prms['period']) ** 2) / (2 * drive_prms['widths'][ty] ** 2))

    return sm

def drive_sz(t, prms):
    sm = 0
    for tz in prms["sz"]:
        drive_prms = prms["drive_prms"]
        t0 = drive_prms["tlocs"][tz] * prms["period"]
        sm = sm + drive_prms['amplitudes'][tz] * np.cos(drive_prms["frequencies"][tz]*t+drive_prms["phases"][tz]) * (1 / (np.sqrt(2 * np.pi))) \
             * np.exp(
            -((t - t0 - np.floor(t / prms['period']) * prms['period']) ** 2) / (2 * drive_prms['widths'][tz] ** 2))

    return sm

def extract_oploc(op,prms):
    return np.where(np.array([(prms["op_data"]["op_sequence"][r]-op).norm() for r in range(len(prms["op_data"]["op_sequence"]))])==0.0)[0]