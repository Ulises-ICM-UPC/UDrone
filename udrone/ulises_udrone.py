#
# Tue Jul 15 10:13:42 2025, extract from Ulises by Gonzalo Simarro and Daniel Calvete
#
import cv2  # type: ignore
import datetime
import json
import matplotlib.pyplot as plt # type: ignore
import numpy as np  # type: ignore
import os
from pathlib import Path
import random
from scipy import optimize  # type: ignore
from scipy import signal # type: ignore
import scipy as sc # type: ignore
import shutil
import string
import subprocess
import time
import warnings
warnings.filterwarnings("ignore")

#
class MinimizeStopper(object):
    def __init__(self, max_sec=60):
        self.max_sec = max_sec
        self.start = time.time()
    def __call__(self, xk=None):
        elapsed = time.time() - self.start
        if elapsed > self.max_sec:
            assert False
        else:
            pass
def A0A12A_2502(A0, A1):  # 1900-01-01; lm:2025-05-28; lr:2025-06-23
    if not (A0.shape[0] == A1.shape[0] and A0.shape[1] == 8 and A1.shape[1] == 3):
        raise Exception("Invalid input: invalid shapes for 'A0' or 'A1'")
    A = np.concatenate((A0, A1), axis=1)
    if False:  # avoidable check for readability
        assert A.shape[0] == A0.shape[0] == A1.shape[0] and A.shape[1] == 11
        assert np.allclose(A[:, 0], A0[:, 0]) and np.allclose(A[:, 8], A1[:, 0])
    return A
def AllVar2DHor_2410(allVar, nc, nr, zr, rangeC):  # lm:2025-03-21; lr:2025-07-02
    if rangeC == 'close':
        dHor = {'nc': nc, 'nr': nr, 'zr': zr, 'range': rangeC}
        rEarth, oHorizon = REarth_2410(), OHorizon_2410()
        dHor = dHor | {'rEarth': rEarth, 'oHorizon': oHorizon}
        allVarKeys = AllVarKeys(rangeC)
        dAllVar = Array2Dictionary(allVarKeys, allVar)
        ef = UnitVectors_2502(dAllVar)[-1]  # dAngVar < dAllVar
        Pa11 = ArrayPx_2410(dAllVar, dAllVar, rangeC)  # {dExtVar, dCaSVar} < dAllVar
        zr = min(zr, dAllVar['zc'] - 0.1 / rEarth)
        a0, b0 = dAllVar['zc'] - 2 * zr, np.sqrt(2 * (dAllVar['zc'] - zr) * rEarth)
        den = max(np.hypot(ef[0], ef[1]), 1.e-14)  # WATCH OUT: epsilon
        xA = dAllVar['xc'] + b0 * ef[0] / den
        yA = dAllVar['yc'] + b0 * ef[1] / den
        zA = -a0
        dHor = dHor | {'xA': xA, 'yA': yA, 'zA': zA}
        ac, bc = Pa11[0] * xA + Pa11[1] * yA + Pa11[2] * zA + Pa11[3], -Pa11[0] * ef[1] + Pa11[1] * ef[0]
        ar, br = Pa11[4] * xA + Pa11[5] * yA + Pa11[6] * zA + Pa11[7], -Pa11[4] * ef[1] + Pa11[5] * ef[0]
        ad, bd = Pa11[8] * xA + Pa11[9] * yA + Pa11[10] * zA + 1, -Pa11[8] * ef[1] + Pa11[9] * ef[0]
        ccUh1, crUh1, ccUh0 = br * ad - bd * ar, bd * ac - bc * ad, bc * ar - br * ac
        den = max(np.hypot(ccUh1, crUh1), 1.e-14)  # WATCH OUT: epsilon
        ccUh1, crUh1, ccUh0 = [item / den for item in [ccUh1, crUh1, ccUh0]]
        crUh1 = ClipWithSign(crUh1, 1.e-14, np.inf)  # WATCH OUT: epsilon
        dHor = dHor | {'ccUh1': ccUh1, 'crUh1': crUh1, 'ccUh0': ccUh0}
        cUhs = np.linspace(-0.1 * nc, +1.1 * nc, 31)
        rUhs = CUh2RUh_2410(cUhs, dHor)
        cDhs, rDhs, possG = CURU2CDRD_2410(cUhs, rUhs, dAllVar, dAllVar, rangeC, rtrnPossG=True)  # no nc nor nr for possG; just well recovered
        if len(possG) < len(cUhs):  # WATCH OUT
            dHor['ccDh'] = np.zeros(oHorizon + 1)
            dHor['ccDh'][0] = -99  # WATCH OUT: epsilon; constant
        else:
            A = np.ones((len(possG), oHorizon + 1))  # IMP*: initialize with ones
            for n in range(1, oHorizon + 1):  # IMP*: increasing
                A[:, n] = cDhs[possG] ** n
            b = rDhs[possG]
            try:
                AT = np.transpose(A)
                dHor['ccDh'] = np.linalg.solve(np.dot(AT, A), np.dot(AT, b))
                assert np.max(np.abs(b - np.dot(A, dHor['ccDh']))) < 5.e-1  # WATCH OUT: assert meant for try
            except Exception:
                dHor['ccDh'] = np.zeros(oHorizon + 1)
                dHor['ccDh'][0] = -99  # WATCH OUT: epsilon; constant
    elif rangeC == 'long':
        dHor = {}  # WATCH OUT: missing
    else:
        raise Exception("Invalid input: 'rangeC' ('{}') must be 'close' or 'long'".format(rangeC))
    return dHor
def AllVar2DMCS_2410(allVar, nc, nr, rangeC, incHor=True, zr=0.):  # lm:2025-02-17; lr:2025-07-02
    dMCS = {'rangeC': rangeC, 'nc': nc, 'nr': nr}
    allVarKeys = AllVarKeys(rangeC)
    dAllVar = Array2Dictionary(allVarKeys, allVar)
    dMCS = dMCS | {'allVar': allVar, 'allVarKeys': allVarKeys, 'dAllVar': dAllVar}
    dMCS = dMCS | dAllVar  # IMP*: keys
    if rangeC == 'close':
        px = np.asarray([dAllVar['xc'], dAllVar['yc'], dAllVar['zc']])
        dMCS = dMCS | {'px': px, 'pc': px}
    elif rangeC == 'long':
        px = np.asarray([dAllVar['x0'], dAllVar['y0'], dAllVar['z0']])
        dMCS = dMCS | {'px': px, 'p0': px}
    else:
        raise Exception("Invalid input: 'rangeC' ('{}') must be 'close' or 'long'".format(rangeC))
    R = MatrixR_2502(dAllVar)  # dAngVar < dAllVar
    dMCS = dMCS | {'R': R}
    eu, ev, ef = UnitVectors_2502(dAllVar)  # dAngVar < dAllVar
    dMCS = dMCS | {'eu': eu, 'eux': eu[0], 'euy': eu[1], 'euz': eu[2]}
    dMCS = dMCS | {'ev': ev, 'evx': ev[0], 'evy': ev[1], 'evz': ev[2]}
    dMCS = dMCS | {'ef': ef, 'efx': ef[0], 'efy': ef[1], 'efz': ef[2]}
    Px = ArrayPx_2410(dAllVar, dAllVar, rangeC)  # dExtVar, dCaSVar < dAllVar
    if rangeC == 'close':
        dMCS = dMCS | {'Px': Px, 'Pa11': Px}
    elif rangeC == 'long':
        dMCS = dMCS | {'Px': Px, 'Po8': Px}
    else:
        raise Exception("Invalid input: 'rangeC' ('{}') must be 'close' or 'long'".format(rangeC))
    if incHor:
        dHor = AllVar2DHor_2410(allVar, nc, nr, zr, rangeC)
        dMCS = dMCS | {'dHor': dHor}
        dMCS = dMCS | dHor  # IMP*: keys
    return dMCS
def AllVar2SubVar_2410(allVar, subVarKeys, rangeC):  # 2010-01-01; lm:2025-04-28; lr:2025-06-23
    allVarKeys = AllVarKeys(rangeC)
    dAllVar = Array2Dictionary(allVarKeys, allVar)
    subVar = Dictionary2Array(subVarKeys, dAllVar)
    return subVar
def AllVarKeys(rangeC):  # 2010-01-01; lr:2025-04-28; lr:2025-07-11
    if rangeC == 'close':
        allVarKeys = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or']  # WATCH OUT: cannot be changed
        if True:  # avoidable check for readability
            assert len(allVarKeys) == 14
    elif rangeC == 'long':
        allVarKeys = ['x0', 'y0', 'z0', 'ph', 'sg', 'ta', 'sc', 'sr']  # WATCH OUT: cannot be changed
        if True:  # avoidable check for readability
            assert len(allVarKeys) == 8
    else:
        raise Exception("Invalid input: 'rangeC' ('{}') must be 'close' or 'long'".format(rangeC))
    return allVarKeys
def ApplyAffineA01_2504(A01, xs0, ys0):  # 1900-01-01; lm:2025-05-06; lm:2025-06-27
    xs1 = A01[0] * xs0 + A01[1] * ys0 + A01[2]
    ys1 = A01[3] * xs0 + A01[4] * ys0 + A01[5]
    return xs1, ys1
def ApplyHomographyH01_2504(H01, xs0, ys0):  # 2010-01-01; lm:2025-06-06; lm:2025-06-27
    dens = H01[2, 0] * xs0 + H01[2, 1] * ys0 + H01[2, 2]
    dens = ClipWithSign(dens, 1.e-14, np.inf)  # WATCH OUT; epsilon
    xs1 = (H01[0, 0] * xs0 + H01[0, 1] * ys0 + H01[0, 2]) / dens
    ys1 = (H01[1, 0] * xs0 + H01[1, 1] * ys0 + H01[1, 2]) / dens
    return xs1, ys1
def AreVarOK_2410(var, varKeys, areScl=False, dScl=None):  # 2010-01-01; lm:2025-06-23; lr:2025-07-10
    if areScl:
        var = SclVar_2410(var, varKeys, dScl, 'unscale')
    dVar = Array2Dictionary(varKeys, var)
    areVarOK = True
    for key in varKeys:
        if dVar[key] is None:
            return False
        if key in ['zc', 'sca', 'sra', 'sc', 'sr'] and dVar[key] < 1.e-12:  # WATCH OUT: epsilon
            return False
        elif key == 'sg' and np.abs(dVar[key]) > np.pi:  # WATCH OUT: epsilon; np.angle() in (-pi, pi]
            return False
        elif key == 'ta' and not (0 <= dVar[key] <= np.pi):  # WATCH OUT: epsilon
            return False
    return areVarOK
def Array2Dictionary(keys, theArray):  # 1900-01-01; lm:2025-05-01; lr:2025-07-11
    if not (len(set(keys)) == len(keys) == len(theArray)):
        raise Exception("Invalid input: 'keys' and 'theArray' must have the same length")
    if isinstance(theArray, (list, np.ndarray)):
        theDictionary = dict(zip(keys, theArray))
    else:
        raise Exception("Invalid input: 'theArray' must be a list or a NumPy ndarray")
    return theDictionary
def ArrayPx_2410(dExtVar, dCaSVar, rangeC):  # 1900-01-01; lm:2025-04-30; lm:2025-06-21
    Rt = MatrixRt_2410(dExtVar, rangeC)
    K = MatrixK_2410(dCaSVar, rangeC)
    P = np.dot(K, Rt)
    if False:  # avoidable check for readability
        assert P.shape == (3, 4)
    if rangeC == 'close':
        P23 = ClipWithSign(P[2, 3], 1.e-14, np.inf)  # WATCH OUT: epsilon
        Pa = P / P23
        Px = np.hstack([Pa[0, :4], Pa[1, :4], Pa[2, :3]])
    elif rangeC == 'long':
        Po = P[:2, :]
        Px = np.hstack([Po[0, :4], Po[1, :4]])
    else:
        raise Exception("Invalid input: 'rangeC' ('{}') must be 'close' or 'long'".format(rangeC))
    return Px
def CDRD2CURUForParabolicSquaredDistortion_2502(cDs, rDs, oc_, or_, k1asa2, rtrnPossG=False, nc=None, nr=None):  # 2010-01-01; lm:2025-05-28; lr:2025-07-02
    if np.abs(k1asa2) < 1.e-14:  # WATCH OUT: epsilon
        cUs, rUs, possG = 1 * cDs, 1 * rDs, np.arange(len(cDs))
    else:
        chiDas = k1asa2 * ((cDs - oc_) ** 2 + (rDs - or_) ** 2)
        chiUas, possG = XiD2XiUCubicEquation_2410(chiDas, rtrnPossG=rtrnPossG)
        cUs = oc_ + (cDs - oc_) / np.clip(1 + chiUas, 1.e-14, np.inf)  # IMP*: not necessary to clip with sign; 1 + eps
        rUs = or_ + (rDs - or_) / np.clip(1 + chiUas, 1.e-14, np.inf)  # IMP*: not necessary to clip with sign; 1 + eps
    if len(possG) > 0 and rtrnPossG and nc is not None and nr is not None:
        cDsG, rDsG = [item[possG] for item in [cDs, rDs]]
        possGInPossG = CR2PossWithinImage_2502(cDsG, rDsG, nc, nr)  # default margin=0, case=''
        possG = possG[possGInPossG]
    return cUs, rUs, possG
def CDRD2CURU_2410(cDs, rDs, dCaSVar, dDtrVar, rangeC, rtrnPossG=False, nc=None, nr=None, margin=0):  # undistort; potentially expensive; 2010-01-01; lm:2025-05-01; lm:2025-07-01
    uDas, vDas = CR2UaVa_2410(cDs, rDs, dCaSVar, rangeC)
    uUas, vUas, possG = UDaVDa2UUaVUa_2410(uDas, vDas, dDtrVar, rangeC, rtrnPossG=rtrnPossG)  # WATCH OUT: potentially expensive
    cUs, rUs = UaVa2CR_2410(uUas, vUas, dCaSVar, rangeC)
    if len(possG) > 0 and rtrnPossG and nc is not None and nr is not None:
        cDsG, rDsG = [item[possG] for item in [cDs, rDs]]
        possGInPossG = CR2PossWithinImage_2502(cDsG, rDsG, nc, nr, margin=margin, case='')
        possG = possG[possGInPossG]
    return cUs, rUs, possG
def CDRD2XYZ_2410(cDs, rDs, planes, dMCS, rtrnPossG=False, margin=0):  # potentially expensive; 2010-01-01; lm:2025-05-05; lr:2025-07-02
    Px, rangeC, dAllVar, ef, nc, nr = [dMCS[item] for item in ['Px', 'rangeC', 'dAllVar', 'ef', 'nc', 'nr']]
    cUs, rUs, possG = CDRD2CURU_2410(cDs, rDs, dAllVar, dAllVar, rangeC, rtrnPossG=rtrnPossG, nc=nc, nr=nr, margin=margin)
    xs, ys, zs, possGH = CURU2XYZ_2410(cUs, rUs, planes, Px, rangeC, rtrnPossG=rtrnPossG, dCamVar=dAllVar, ef=ef, nc=nc, nr=nr)
    possG = np.intersect1d(possG, possGH, assume_unique=True)
    return xs, ys, zs, possG
def CDRDZ2XY_2410(cDs, rDs, zs, dMCS, rtrnPossG=False, margin=0):  # potentially expensive; 2010-01-01; lm:2025-05-05; lr:2025-07-02
    planes = {'pxs': np.zeros(zs.shape), 'pys': np.zeros(zs.shape), 'pzs': np.ones(zs.shape), 'pts': -zs}
    xs, ys, _, possG = CDRD2XYZ_2410(cDs, rDs, planes, dMCS, rtrnPossG=rtrnPossG, margin=margin)
    return xs, ys, possG
def CDh2RDh_2410(cDhs, dHor, rtrnPossG=False):  # 2000-01-01; lm:2025-05-01; lr:2025-06-23
    rDhs = np.polyval(dHor['ccDh'][::-1], cDhs)
    if rtrnPossG:
        possG = CR2PossWithinImage_2502(cDhs, rDhs, dHor['nc'], dHor['nr'])
    else:
        possG = np.asarray([], dtype=int)
    return rDhs, possG
def CR2CRIntegerAroundAndWeights_2504(cs, rs): # 2000-01-01; lm:2025-04-02; lr:2025-06-22
    csFloor = np.floor(cs).astype(int); csDelta = cs - csFloor 
    rsFloor = np.floor(rs).astype(int); rsDelta = rs - rsFloor
    csIA, rsIA, wsA = np.zeros((len(cs), 4), dtype=int), np.zeros((len(cs), 4), dtype=int), np.zeros((len(cs), 4))
    csIA[:, 0], rsIA[:, 0], wsA[:, 0] = csFloor + 0, rsFloor + 0, (1 - csDelta) * (1 - rsDelta)
    csIA[:, 1], rsIA[:, 1], wsA[:, 1] = csFloor + 1, rsFloor + 0, (0 + csDelta) * (1 - rsDelta)
    csIA[:, 2], rsIA[:, 2], wsA[:, 2] = csFloor + 0, rsFloor + 1, (1 - csDelta) * (0 + rsDelta)
    csIA[:, 3], rsIA[:, 3], wsA[:, 3] = csFloor + 1, rsFloor + 1, (0 + csDelta) * (0 + rsDelta)
    if True:  # avoidable check for readability
        assert np.allclose(wsA.sum(axis=1), 1)
    possCs0, possCs1 = [np.where(np.abs(csDelta - item) < 1.e-9)[0] for item in [0, 1]]  # WATCH OUT: epsilon
    possRs0, possRs1 = [np.where(np.abs(rsDelta - item) < 1.e-9)[0] for item in [0, 1]]  # WATCH OUT: epsilon
    for corner in range(4):  # all corners are given the same value
        if len(possCs0) > 0:
            csIA[possCs0, corner] = csFloor[possCs0]
        if len(possCs1) > 0:
            csIA[possCs1, corner] = csFloor[possCs1] + 1
        if len(possRs0) > 0:
            rsIA[possRs0, corner] = rsFloor[possRs0]
        if len(possRs1) > 0:
            rsIA[possRs1, corner] = rsFloor[possRs1] + 1
    return csIA, rsIA, wsA
def CR2CRIntegerWithinImage_2502(cs, rs, nc, nr, margin=0, case='round'):  # 1900-01-01; lm:2025-05-28; lr:2025-07-07
    csI, rsI = CR2CRInteger_2504(cs, rs, case=case)
    possW = CR2PossWithinImage_2502(csI, rsI, nc, nr, margin=margin, case='')
    csIW, rsIW = [item[possW] for item in [csI, rsI]]
    return csIW, rsIW
def CR2CRInteger_2504(cs, rs, case='round'):  # 1900-01-01; lm:2025-05-28; lr:2025-07-13
    if case == 'round':
        csI = np.round(cs).astype(int)
        rsI = np.round(rs).astype(int)
    elif case == 'floor':
        csI = np.floor(cs).astype(int)
        rsI = np.floor(rs).astype(int)
    else:
        raise Exception("Invalid input: 'case' ('{}') must be 'round' or 'floor'".format(case))
    return csI, rsI
def CR2PossWithinImage_2502(cs, rs, nc, nr, margin=0, case=''):  # 1900-01-01; lm:2025-05-28; lr:2025-07-13
    if len(cs) == 0 or len(rs) == 0:
        return np.asarray([], dtype=int)
    if case in ['round', 'floor']:
        cs, rs = CR2CRInteger_2504(cs, rs, case=case)
    cMin, cMax = -1/2 + margin, nc-1/2 - margin  # IMP*
    rMin, rMax = -1/2 + margin, nr-1/2 - margin  # IMP*
    possW = np.where((cs > cMin) & (cs < cMax) & (rs > rMin) & (rs < rMax))[0]  # WATCH OUT: "<" and ">" for safety
    return possW
def CR2UaVa_2410(cs, rs, dCaSVar, rangeC):  # lm:1900-01-01; lr:2025-05-01; lr:2025-06-30
    if rangeC == 'close':
        sca, sra = [ClipWithSign(dCaSVar[item], 1.e-14, np.inf) for item in ['sca', 'sra']]  # WATCH OUT: epsilon
        uas = (cs - dCaSVar['oc']) * sca
        vas = (rs - dCaSVar['or']) * sra
    elif rangeC == 'long':
        sc, sr = [ClipWithSign(dCaSVar[item], 1.e-14, np.inf) for item in ['sc', 'sr']]  # WATCH OUT: epsilon
        uas = (cs - dCaSVar['oc']) * sc  # uas are actually us in this case
        vas = (rs - dCaSVar['or']) * sr  # vas are actually vs in this case
    else:
        raise Exception("Invalid input: 'rangeC' ('{}') must be 'close' or 'long'".format(rangeC))
    return uas, vas
def CRWithinImage2NormalizedLengthsAndAreas_2504(nc, nr, cs, rs, margin=0):  # lm:2022-11-10; lr:2025-06-23
    if not len(CR2PossWithinImage_2502(cs, rs, nc, nr, margin=margin)) == len(cs):
        raise Exception("Invalid input: 'cs' and 'rs' must be within the image boundaries")
    cMin, cMax = margin, nc - 1 - margin
    rMin, rMax = margin, nr - 1 - margin
    lns = np.zeros((len(cs), 4))
    lns[:, 0] = cs - cMin
    lns[:, 1] = cMax - cs
    lns[:, 2] = rs - rMin 
    lns[:, 3] = rMax - rs
    normalizedLns = np.min(lns, axis=1) / max(cMax - cMin, rMax - rMin)
    if False:  # avoidable check for readability
        assert np.min(normalizedLns) >= 0 and len(normalizedLns) == len(cs)
    ars = np.zeros((len(cs), 4))
    ars[:, 0] = (cs - cMin) * (rs - rMin)
    ars[:, 1] = (cs - cMin) * (rMax - rs)
    ars[:, 2] = (cMax - cs) * (rs - rMin)
    ars[:, 3] = (cMax - cs) * (rMax - rs)
    normalizedArs = np.min(ars, axis=1) / ((cMax - cMin) * (rMax - rMin))
    if False:  # avoidable check for readability
        assert np.min(normalizedArs) >= 0 and len(normalizedArs) == len(cs)
    return normalizedLns, normalizedArs
def CURU2B(cUs, rUs):  # 1900-01-01; lm:2025-05-28; lr:2025-07-02
    poss0, poss1 = Poss0AndPoss1InFind2DTransform_2504(len(cUs))
    b = np.zeros(2 * len(cUs))
    b[poss0] = cUs
    b[poss1] = rUs
    return b
def CURU2CDRD_2410(cUs, rUs, dCaSVar, dDtrVar, rangeC, rtrnPossG=False, nc=None, nr=None, margin=0):  # distort; 2010-01-01; lm:2025-05-01; lm:2025-07-01
    uUas, vUas = CR2UaVa_2410(cUs, rUs, dCaSVar, rangeC)
    uDas, vDas, possG = UUaVUa2UDaVDa_2410(uUas, vUas, dDtrVar, rangeC, rtrnPossG=rtrnPossG)
    cDs, rDs = UaVa2CR_2410(uDas, vDas, dCaSVar, rangeC)
    if len(possG) > 0 and rtrnPossG and nc is not None and nr is not None:
        cDsG, rDsG = [item[possG] for item in [cDs, rDs]]
        possGInPossG = CR2PossWithinImage_2502(cDsG, rDsG, nc, nr, margin=margin, case='')
        possG = possG[possGInPossG]
    return cDs, rDs, possG
def CURU2XYZ_2410(cUs, rUs, planes, Px, rangeC, rtrnPossG=False, dCamVar=None, ef=None, nc=None, nr=None):  # 2000-01-01; lm:2025-05-01; lr:2025-07-02
    if rangeC == 'close':
        A11s, A12s, A13s, bb1s = Px[0] - cUs * Px[8], Px[1] - cUs * Px[9], Px[2] - cUs * Px[10], cUs - Px[3]
        A21s, A22s, A23s, bb2s = Px[4] - rUs * Px[8], Px[5] - rUs * Px[9], Px[6] - rUs * Px[10], rUs - Px[7]
        A31s, A32s, A33s, bb3s = planes['pxs'], planes['pys'], planes['pzs'], -planes['pts']
    elif rangeC == 'long':
        A11s, A12s, A13s, bb1s = Px[0], Px[1], Px[2], cUs - Px[3]
        A21s, A22s, A23s, bb2s = Px[4], Px[5], Px[6], rUs - Px[7]
        A31s, A32s, A33s, bb3s = planes['pxs'], planes['pys'], planes['pzs'], -planes['pts']
    else:
        raise Exception("Invalid input: 'rangeC' ('{}') must be 'close' or 'long'".format(rangeC))
    dens = A11s * (A22s * A33s - A23s * A32s) + A12s * (A23s * A31s - A21s * A33s) + A13s * (A21s * A32s - A22s * A31s)
    dens = ClipWithSign(dens, 1.e-14, np.inf)  # WATCH OUT: epsilon
    xs = (bb1s * (A22s * A33s - A23s * A32s) + A12s * (A23s * bb3s - bb2s * A33s) + A13s * (bb2s * A32s - A22s * bb3s)) / dens
    ys = (A11s * (bb2s * A33s - A23s * bb3s) + bb1s * (A23s * A31s - A21s * A33s) + A13s * (A21s * bb3s - bb2s * A31s)) / dens
    zs = (A11s * (A22s * bb3s - bb2s * A32s) + A12s * (bb2s * A31s - A21s * bb3s) + bb1s * (A21s * A32s - A22s * A31s)) / dens
    if rtrnPossG:
        cUsR, rUsR, possG = XYZ2CURU_2410(xs, ys, zs, Px, rangeC, rtrnPossG=rtrnPossG, dCamVar=dCamVar, ef=ef, nc=nc, nr=nr)
        cUsG, rUsG, cUsRG, rUsRG = [item[possG] for item in [cUs, rUs, cUsR, rUsR]]
        possGInPossG = np.where(np.hypot(cUsRG - cUsG, rUsRG - rUsG) < 1.e-6)[0]  # WATCH OUT: epsilon
        possG = possG[possGInPossG]
        xsG, ysG, zsG = [item[possG] for item in [xs, ys, zs]]
        pxsG, pysG, pzsG, ptsG = [planes[item][possG] for item in ['pxs', 'pys', 'pzs', 'pts']]
        possGInPossG = np.where(np.abs(pxsG * xsG + pysG * ysG + pzsG * zsG + ptsG) < 1.e-6)[0]  # WATCH OUT: epsilon
        possG = possG[possGInPossG]
    else:
        possG = np.asarray([], dtype=int)
    return xs, ys, zs, possG
def CURUXYZ2A1(cUs, rUs, xs, ys, zs):  # for close range only; 1900-01-01; lm:2025-05-28; lr:2025-06-23
    poss0, poss1 = Poss0AndPoss1InFind2DTransform_2504(len(cUs))
    A1 = np.zeros((2 * len(xs), 3))
    A1[poss0, 0], A1[poss0, 1], A1[poss0, 2] = -cUs * xs, -cUs * ys, -cUs * zs
    A1[poss1, 0], A1[poss1, 1], A1[poss1, 2] = -rUs * xs, -rUs * ys, -rUs * zs
    return A1
def CURUXYZ2A_2502(cUs, rUs, xs, ys, zs, rangeC, A0=None, A1=None):  # 1900-01-01; lm:2025-05-28; lr:2025-06-23
    if A0 is None:
        A0 = XYZ2A0(xs, ys, zs)
    if rangeC == 'close':
        if A1 is None:
            A1 = CURUXYZ2A1(cUs, rUs, xs, ys, zs)
        A = A0A12A_2502(A0, A1)
    elif rangeC == 'long':
        A = A0
    else:
        raise Exception("Invalid input: 'rangeC' ('{}') must be 'close' or 'long'".format(rangeC))
    return A
def CURUXYZ2Px_2502(cUs, rUs, xs, ys, zs, rangeC, A=None, A0=None, A1=None):  # 1900-01-01; lm:2025-05-28; lr:2025-07-02
    if A is None:
        A = CURUXYZ2A_2502(cUs, rUs, xs, ys, zs, rangeC, A0=A0, A1=A1)
    b = CURU2B(cUs, rUs)
    try:
        AT = np.transpose(A)
        Px = np.linalg.solve(np.dot(AT, A), np.dot(AT, b))
    except Exception:
        Px = None
    return Px
def CUh2RUh_2410(cUhs, dHor, eps=1.e-12):  # 2000-01-01; lm:2025-04-25; lr:2025-07-02
    crUh1 = ClipWithSign(dHor['crUh1'], eps, np.inf)
    rUhs = - (dHor['ccUh1'] * cUhs + dHor['ccUh0']) / crUh1
    return rUhs
def CleanAFld_2504(pathFld):  # 2000-01-01; lm:2025-04-19; lr:2025-06-27
    if not os.path.exists(pathFld):
        return None
    for item in os.listdir(pathFld):
        pathItem = os.path.join(pathFld, item)
        if os.path.isfile(pathItem) or os.path.islink(pathItem):
            os.remove(pathItem)
        elif os.path.isdir(pathItem):
            shutil.rmtree(pathItem)
    return None
def ClipWithSign(xs, x0, x1):  # 1900-01-01; lm:2025-05-28; lr:2025-07-14
    if not (0 < x0 < x1):
        raise Exception("Invalid input: invalid 'x0' and/or 'x1'; must satisfy 0 < x0 < x1")
    signs = np.where(np.sign(xs) == 0, 1, np.sign(xs))
    xs = signs * np.clip(np.abs(xs), x0, x1)
    return xs
def Cloud2XULAnd_2504(xs, ys):  # 2010-01-01; lm:2025-03-31; lr:2025-07-07
    (xUL, xUR, xDR, xDL), (yUL, yUR, yDR, yDL) = CloudOfPoints2Rectangle_2504(xs, ys)[:2]
    angle = np.angle((xUR - xUL) + 1j * (yUR - yUL))
    xyLengthInC = np.hypot(xUR - xUL, yUR - yUL)
    xyLengthInR = np.hypot(xUR - xDR, yUR - yDR)
    if True:  # avoidable check for readability
        assert np.isclose(xyLengthInC, np.hypot(xDR - xDL, yDR - yDL))
        assert np.isclose(xyLengthInR, np.hypot(xUL - xDL, yUL - yDL))
    return xUL, yUL, angle, xyLengthInC, xyLengthInR
def CloudAndPpm2DPdf_2504(xs, ys, ppm):  # lm:2025-03-31; lr:2025-07-10
    xUL, yUL, angle, xyLengthInC, xyLengthInR = Cloud2XULAnd_2504(xs, ys)
    dPdf = LoadDPdfTxt_2504(xUL=xUL, yUL=yUL, angle=angle, xyLengthInC=xyLengthInC, xyLengthInR=xyLengthInR, ppm=ppm)
    return dPdf
def CloudOfPoints2RectangleAux_2504(angle, xs, ys, margin=0.):  # 1900-01-01; lm:2025-05-06; lr:2025-06-30
    lDs = - np.sin(angle) * xs + np.cos(angle) * ys  # signed-distances to D-line dir = (+cos, +sin) through origin (0, 0); positive above
    lD0 = {'lx': -np.sin(angle), 'ly': +np.cos(angle), 'lt': -(np.min(lDs)-margin)}
    lD1 = {'lx': -np.sin(angle), 'ly': +np.cos(angle), 'lt': -(np.max(lDs)+margin)}
    lPs = + np.cos(angle) * xs + np.sin(angle) * ys  # signed-distances to P-line dir = (+sin, -cos) through origin (0, 0); positive right
    lP0 = {'lx': +np.cos(angle), 'ly': +np.sin(angle), 'lt': -(np.min(lPs)-margin)}
    lP1 = {'lx': +np.cos(angle), 'ly': +np.sin(angle), 'lt': -(np.max(lPs)+margin)}
    xcs, ycs = [np.zeros(4) for _ in range(2)]
    xcs[0], ycs[0] = IntersectionOfTwoLines_2506(lD0, lP0)[:2]
    xcs[1], ycs[1] = IntersectionOfTwoLines_2506(lP0, lD1)[:2]
    xcs[2], ycs[2] = IntersectionOfTwoLines_2506(lD1, lP1)[:2]
    xcs[3], ycs[3] = IntersectionOfTwoLines_2506(lP1, lD0)[:2]
    area = (np.max(lDs) - np.min(lDs) + 2 * margin) * (np.max(lPs) - np.min(lPs) + 2 * margin)  # IMP*
    return xcs, ycs, area
def CloudOfPoints2Rectangle_2504(xs, ys, margin=0.):  # 1900-01-01; lm:2025-07-07; lr:2025-07-10
    xcs, ycs, area = None, None, np.inf
    for angleH in np.linspace(0, np.pi / 2, 1000):  # WATCH OUT; a capon
        xcsH, ycsH, areaH = CloudOfPoints2RectangleAux_2504(angleH, xs, ys, margin=margin)  # already oriented clockwise
        if areaH < area:
            xcs, ycs, area = xcsH, ycsH, areaH
    pos0 = np.argmin(np.hypot(xcs - xs[0], ycs - ys[0]))
    xcs, ycs = np.roll(xcs, -pos0), np.roll(ycs, -pos0)
    if True:  # avoidable check for readability
        assert len(xcs) == len(ycs) == 4
    return xcs, ycs, area
def CreatePlw_2504(dPlwPC, dImgs):  # lm:2021-09-15; lr:2025-06-24
    cameras = [item for item in dImgs.keys() if item in dPlwPC['camerasOfUse']]
    if not all(img.shape[2] == 3 for img in [dImgs[camera] for camera in cameras]):
        raise Exception("Invalid input: images must have 3 channels")
    if len(cameras) == 0:
        return np.zeros((dPlwPC['nr'], dPlwPC['nc'], 3), dtype=np.uint8)
    wsPlw = 1.e-14 * np.ones(dPlwPC['nc'] * dPlwPC['nr'])
    for camera in cameras:
        plwPoss = dPlwPC[camera]['plwPoss']  # planview positions captured by the camera
        wsPlw[plwPoss] += dPlwPC[camera]['ws']  # add weight contribution camera given pixel position within the camera
    imgPlw = np.zeros((dPlwPC['nr'], dPlwPC['nc'], 3))
    for camera in cameras:
        plwPoss = dPlwPC[camera]['plwPoss']  # planview positions captured by the camera
        csPlw, rsPlw = [dPlwPC[item][plwPoss] for item in ['cs', 'rs']]  # planview pixels captured by the camera
        for corner in range(4):
            csIAC, rsIAC, wsAC = [dPlwPC[camera][item][:, corner] for item in ['csIA', 'rsIA', 'wsA']]
            imgPlw[rsPlw, csPlw, :] += dImgs[camera][rsIAC, csIAC, :] * np.outer(wsAC * dPlwPC[camera]['ws'] / wsPlw[plwPoss], np.ones(3))
    imgPlw = imgPlw.astype(np.uint8)
    return imgPlw
def DPlwPC_2504(dsDMCSs, dPdf, z):  # lm:2024-06-24; lr:2025-06-24
    ncP, nrP, csP, rsP, xsP, ysP = [dPdf[item] for item in ['nc', 'nr', 'cs', 'rs', 'xs', 'ys']]    
    dPlwPC = {'nc': ncP, 'nr': nrP, 'cs': csP, 'rs': rsP, 'xs': xsP, 'ys': ysP}
    zsP, cameras = z * np.ones(xsP.shape), sorted(dsDMCSs)
    dPlwPC |= {'zs': zsP, 'cameras': cameras}
    camerasOfUse = []
    for camera in cameras:
        dMCS = dsDMCSs[camera]
        csC, rsC, plwPossC = XYZ2CDRD_2410(xsP, ysP, zsP, dMCS, rtrnPossG=True, margin=1)  # IMP*; margin
        if len(plwPossC) == 0:  # camera does not look at planview domain
            continue
        csC, rsC = csC[plwPossC], rsC[plwPossC]  # IMP*
        wsC = CRWithinImage2NormalizedLengthsAndAreas_2504(dMCS['nc'], dMCS['nr'], csC, rsC)[0]
        csCIA, rsCIA, wsCA = CR2CRIntegerAroundAndWeights_2504(csC, rsC)
        dPlwPC |= {camera: {'plwPoss': plwPossC, 'ws': wsC, 'csIA': csCIA, 'rsIA': rsCIA, 'wsA': wsCA}}
        camerasOfUse.append(camera)
    dPlwPC |= {'camerasOfUse': camerasOfUse}
    return dPlwPC
def DetectHorizon(img, dHorApp, aWindow=0.10, aError=0.002, aFilt=0.01, oHorizon=2, quality_min=0.25, pDesired=1-1.e-9, margin=0.):  # 2000-01-01; lm:2025-03-21; lr:2025-06-23
    img = PathImgOrImg2Img(img)
    nr, nc = img.shape[:2]
    cDhs = np.arange(nc)
    rDhsApp = np.round(CDh2RDh_2410(cDhs, dHorApp)[0]).astype(int)
    nrAux = int(aWindow * nr) + 1
    imgAux = np.zeros((nrAux, nc, 3), dtype=np.uint8)
    for c in range(nc):
        r0 = rDhsApp[c] - nrAux // 2
        r1 = r0 + nrAux
        rsInImg = np.clip(np.arange(r0, r1), 0, nr - 1)
        imgAux[:, c, :] = img[rsInImg, c, :]
    imgAux = cv2.GaussianBlur(imgAux, (1, 2 * int(aFilt * nr / 2) + 1), sigmaX=0, sigmaY=0)
    imgAux = cv2.cvtColor(imgAux, cv2.COLOR_BGR2GRAY)
    rDhsAux = np.argmin(cv2.Sobel(imgAux, cv2.CV_64F, dx=0, dy=1, ksize=3), axis=0)  # IMP*: np.argmin
    rDhs = rDhsApp + rDhsAux - nrAux // 2  # IMP*: as for r0
    A, b = np.vander(cDhs, N=oHorizon+1, increasing=True), rDhs
    iForRANSAC, nForRANSAC, possG = 0, np.inf, []
    while iForRANSAC < nForRANSAC:
        possH = np.random.choice(len(cDhs), size=oHorizon+1, replace=False)
        AH, bH = A[possH, :], b[possH]
        try:
            AHT = np.transpose(AH)
            sol = np.linalg.solve(np.dot(AHT, AH), np.dot(AHT, bH))
            rDhsH = np.dot(A, sol)
            errors = np.abs(rDhs - rDhsH)
        except Exception:
            continue
        possGH = np.where(errors <= min(aError * nr, 0.5))[0]
        if len(possGH) > len(possG):
            possG = possGH
            pOutlier = 1 - len(possG) / len(cDhs) + margin * len(possG) / len(cDhs)
            nForRANSAC = NForRANSAC(pOutlier, pDesired, oHorizon+1)
            if len(possG) == len(cDhs):
                break
        iForRANSAC += 1
    quality = len(possG) / len(cDhs)
    if quality < quality_min:
        return np.asarray([], dtype=int), np.asarray([], dtype=int), 0
    cDhs, rDhs = [item[possG] for item in [cDhs, rDhs]]
    return cDhs, rDhs, quality
def Dictionary2Array(keys, theDictionary):  # 1900-01-01; lm:2025-05-01; lr:2025-07-11
    if not set(keys).issubset(theDictionary):
        raise Exception("Invalid input: 'keys' must be contained in 'theDictionary'")
    theArray = np.asarray([theDictionary[key] for key in keys])
    return theArray
def DisplayCRInImage_2504(img, cs, rs, margin=0, factor=1., colors=None, pathOut=None):  # 1900-01-01; lm:2025-05-28; lr:2025-06-27
    img = PathImgOrImg2Img(img)
    nr, nc = img.shape[:2]
    imgOut = img.copy()  # IMP*: otherwise, if imgNew = DisplayCRInImage_2504(img, ...) then img is also modified
    csIW, rsIW = CR2CRIntegerWithinImage_2502(cs, rs, nc, nr, margin=margin)
    if len(csIW) == 0:
        return imgOut
    if colors is None:
        colors = [[0, 0, 0]]
    colors = (colors * ((len(csIW) + len(colors) - 1) // len(colors)))[:len(csIW)]
    radius = int(factor * np.sqrt(nc * nr) / 2.e+2 + 1)
    for pos in range(len(csIW)):
        cv2.circle(imgOut, (csIW[pos], rsIW[pos]), radius, colors[pos], -1)
    if pathOut is not None:
        os.makedirs(os.path.dirname(pathOut), exist_ok=True)
        cv2.imwrite(pathOut, imgOut)
    return imgOut
def DistancePointToPoint3D(x0, y0, z0, x1, y1, z1):  # 2000-01-01; lm:2025-06-04; lr:2025-07-14
    distance = np.hypot(np.hypot(x1 - x0, y1 - y0), z1 - z0)
    return distance
def ErrorG_2410(xs, ys, zs, cDs, rDs, dMCS):  # explicit; 2010-01-01; lm:2025-05-29; lr:2025-06-23
    if len(xs) == 0:
        errorG = 0
    else:
        errorsG = ErrorsG_2504(xs, ys, zs, cDs, rDs, dMCS)  # WATCH OUT: all positions
        errorG = np.sqrt(np.mean(errorsG ** 2))
    return errorG
def ErrorH_2410(cDhs, rDhs, dHor):  # 2010-01-01; lm:2025-05-29; lr:2025-06-23
    if len(cDhs) == 0:
        errorH = 0
    else:
        rDhsR = CDh2RDh_2410(cDhs, dHor)[0]  # IMP*: all positions
        errorsH = np.abs(rDhsR - rDhs)
        errorH = np.sqrt(np.mean(errorsH ** 2))
    return errorH
def ErrorT_2410(dGH1, dMCS, dHor):  # 2010-01-01; lm:2025-05-29; lr:2025-06-23
    xs, ys, zs, cDs, rDs = [dGH1[item] for item in ['xs', 'ys', 'zs', 'cDs', 'rDs']]
    errorG = ErrorG_2410(xs, ys, zs, cDs, rDs, dMCS)
    if any(item not in dGH1 or dGH1[item] is None or len(dGH1[item]) == 0 for item in ['cDhs', 'rDhs']):
        errorH = 0
    else:
        cDhs, rDhs = [dGH1[item] for item in ['cDhs', 'rDhs']]
        errorH = ErrorH_2410(cDhs, rDhs, dHor)
    errorT = errorG + errorH  # WATCH OUT: no weights
    return errorT
def ErrorsG_2504(xs, ys, zs, cDs, rDs, dMCS):  # explicit; 2010-01-01; lm:2025-05-29; lr:2025-06-23
    if len(xs) == 0:
        errorsG = np.asarray([])
    else:
        cDsR, rDsR = XYZ2CDRD_2410(xs, ys, zs, dMCS)[:2]  # IMP*: all positions
        errorsG = np.hypot(cDsR - cDs, rDsR - rDs)
    return errorsG
def FilterEquispacedXYData_2502(ys, dx, length, method='ButterWorth'): # lm:2025-07-02; lr:2025-07-10
    length = max(dx, length)
    if method == 'ButterWorth':
        b, a = signal.butter(2, dx / length, output='ba')  # WATCH OUT: 2 is the order
        ysF = signal.filtfilt(b, a, ys)
    elif method == 'gaussian':
        ysF = sc.ndimage.gaussian_filter1d(ys, length / dx)
    else:
        raise Exception("Invalid input: 'method' ('{}') must be 'ButterWorth' or 'gaussian'".format(method))
    return ysF
def FilterNotEquispacedXYData_2502(xs, ys, length, method='ButterWorth'):  # 2000-01-01; lm:2025-06-20; lr:2025-07-10
    if not np.min(np.diff(xs)) > 0:
        raise Exception("Invalid input: 'xs' must be strictly increasing")
    nOfPoints = int((np.max(xs) - np.min(xs)) / np.percentile(np.diff(xs), 25) + 1)
    xsE = np.linspace(xs[0], xs[-1], nOfPoints)
    ysE = np.interp(xsE, xs, ys, left=ys[0], right=ys[-1])
    dx = np.mean(np.diff(xsE))
    ysEF = FilterEquispacedXYData_2502(ysE, dx, length, method=method)
    ysF = np.interp(xs, xsE, ysEF)
    return ysF
def FindAFirstSeed_2502(dBasSD, dGH1, dGvnVar, dtMaxInSec=60):  # lm:2025-06-25; lr:2025-07-13
    if dBasSD['rangeC'] == 'close':
        xcApp, ycApp, zcApp, k1asa2App, seemsOK = GCPsD2Pc(dGH1['cDs'], dGH1['rDs'], dGH1['xs'], dGH1['ys'], dGH1['zs'], 0.0015*np.hypot(dBasSD['nc'], dBasSD['nr']), nc=dBasSD['nc'], nr=dBasSD['nr'], margin=0.5)  # WATCH OUT: epsilon, 0.001 for error and 0.5 for RANSAC
    freVarKeys = [item for item in dBasSD['selVarKeys'] if item not in dGvnVar]
    theArgs = {'dBasSD': dBasSD, 'dGH1': dGH1, 'subVarKeys': freVarKeys, 'dRemVar': dGvnVar}
    dMCSSeed, errorTSeed, time0 = None, np.inf, datetime.datetime.now()
    while datetime.datetime.now() - time0 < datetime.timedelta(seconds=dtMaxInSec):  # IMP*
        if np.random.random() < 0.25 and dBasSD['rangeC'] == 'close' and seemsOK:  # IMP*
            sclFreVar = RndSclVar_2506(freVarKeys, dBasSD, dGvnVar, xs=dGH1['xs'], ys=dGH1['ys'], zs=dGH1['zs'], xc=xcApp, yc=ycApp, zc=zcApp, k1asa2=k1asa2App)
        else:
            sclFreVar = RndSclVar_2506(freVarKeys, dBasSD, dGvnVar, xs=dGH1['xs'], ys=dGH1['ys'], zs=dGH1['zs'])
        if not AreVarOK_2410(sclFreVar, freVarKeys, areScl=True, dScl=dBasSD['dScl']):
            continue
        errorT = SclSubVar2FTM_2502(sclFreVar, theArgs)
        if errorT < 0.4 * np.hypot(dBasSD['nc'], dBasSD['nr']):  # IMP*: 0.4; 0.6
            try:
                sclFreVar = optimize.minimize(SclSubVar2FTM_2502, sclFreVar, args=theArgs, callback=MinimizeStopper(5.)).x  # WATCH OUT: time = 1, 5, 10?
            except Exception:
                continue
            if not AreVarOK_2410(sclFreVar, freVarKeys, areScl=True, dScl=dBasSD['dScl']):
                continue
            errorT = SclSubVar2FTM_2502(sclFreVar, theArgs)
            if errorT < errorTSeed:
                dMCS = SclSubVar2DMCS_2502(dBasSD, sclFreVar, freVarKeys, dGvnVar, incHor=True)
                if True:  # avoidable check for readability
                    assert np.isclose(errorT, ErrorT_2410(dGH1, dMCS, dMCS['dHor']))
                dMCSSeed, errorTSeed = dMCS, errorT
            if errorT < 0.05 * np.hypot(dBasSD['nc'], dBasSD['nr']):  # IMP*: epsilon 0.05; 0.1
                break
    return dMCSSeed, errorTSeed
def FindAffineA01_2504(xs0, ys0, xs1, ys1):  # 1900-01-01; lm:2025-05-06; lr:2025-06-26
    minNOfPoints = 3
    if not (len(xs0) == len(ys0) == len(xs1) == len(ys1) >= minNOfPoints):
        return None
    A, b = np.zeros((2 * len(xs0), 6)), np.zeros(2 * len(xs0))  # IMP*; initialize with zeroes
    poss0, poss1 = Poss0AndPoss1InFind2DTransform_2504(len(xs0))
    A[poss0, 0], A[poss0, 1], A[poss0, 2], b[poss0] = xs0, ys0, np.ones(xs0.shape), xs1
    A[poss1, 3], A[poss1, 4], A[poss1, 5], b[poss1] = xs0, ys0, np.ones(xs0.shape), ys1
    try:
        AT = np.transpose(A)
        A01 = np.linalg.solve(np.dot(AT, A), np.dot(AT, b))
    except Exception:  # aligned points
        return None
    return A01
def FindHomographyH01ViaRANSAC_2504(xs0, ys0, xs1, ys1, errorC, pDesired=1-1.e-9, margin=0.5, max_iterations=1000):  # 2010-01-01; lm:2025-06-06; lm:2025-07-01
    minNOfPoints = 4  # 8 unknowns
    possG = FindPossGoodForHomographyH01ViaRANSAC_2504(xs0, ys0, xs1, ys1, errorC, pDesired=pDesired, margin=margin, max_iterations=max_iterations)
    if len(possG) < minNOfPoints:
        H01 = None
    else:
        H01 = FindHomographyH01_2504(xs0[possG], ys0[possG], xs1[possG], ys1[possG])
    return H01, possG
def FindHomographyH01_2504(xs0, ys0, xs1, ys1):  # 2010-01-01; lm:2025-06-06; lm:2025-06-27
    minNOfPoints = 4
    if not (len(xs0) == len(ys0) == len(xs1) == len(ys1) >= minNOfPoints):
        return None
    A, b = np.zeros((2 * len(xs0), 8)), np.zeros(2 * len(xs0))  # IMP*; initialize with zeroes
    poss0, poss1 = Poss0AndPoss1InFind2DTransform_2504(len(xs0))
    A[poss0, 0], A[poss0, 1], A[poss0, 2], A[poss0, 6], A[poss0, 7], b[poss0] = xs0, ys0, np.ones(xs0.shape), -xs0 * xs1, -ys0 * xs1, xs1
    A[poss1, 3], A[poss1, 4], A[poss1, 5], A[poss1, 6], A[poss1, 7], b[poss1] = xs0, ys0, np.ones(xs0.shape), -xs0 * ys1, -ys0 * ys1, ys1
    try:
        AT = np.transpose(A)
        sol = np.linalg.solve(np.dot(AT, A), np.dot(AT, b))
        H01 = np.ones((3, 3))  # IMP*; initialize with ones, for H[2, 2]
        H01[0, :], H01[1, :], H01[2, :2] = sol[:3], sol[3:6], sol[6:]
    except Exception:  # aligned points
        H01 = None
    return H01
def FindPossGoodForHomographyH01ViaRANSAC_2504(xs0, ys0, xs1, ys1, errorC, pDesired=1-1.e-9, margin=0.5, max_iterations=1000):  # 2010-01-01; lm:2025-06-06; lm:2025-07-01
    minNOfPoints = 4  # 8 unknowns
    if not (len(xs0) == len(ys0) == len(xs1) == len(ys1) >= minNOfPoints):
        return np.asarray([], dtype=int)
    possG, iForRANSAC, nForRANSAC = np.asarray([]), 0, np.inf
    while iForRANSAC < min(nForRANSAC, max_iterations):
        possH = np.random.choice(len(xs0), size=minNOfPoints, replace=False)
        H01H = FindHomographyH01_2504(xs0[possH], ys0[possH], xs1[possH], ys1[possH])
        if H01H is None:
            continue
        xs1H, ys1H = ApplyHomographyH01_2504(H01H, xs0, ys0)
        possGH = np.where(np.hypot(xs1H - xs1, ys1H - ys1) < errorC)[0]
        if len(possGH) > len(possG):
            possG = possGH
            pOutlier = 1 - len(possG) / len(xs0) + margin * len(possG) / len(xs0)  # margin in [0, 1); the higher the safer
            nForRANSAC = NForRANSAC(pOutlier, pDesired, minNOfPoints)
        iForRANSAC += 1
        if len(possG) == len(xs0):
            break
    return possG
def GCPs2K1asa2_2410(cDs, rDs, xs, ys, zs, oc_, or_, nc, nr, A0=None):  # for close range only; 2010-01-01; lm:2025-06-22; lr:2025-07-11
    if A0 is None:
        A0 = XYZ2A0(xs, ys, zs)
    if not A0.shape == (2*len(xs), 8):
        raise Exception("Invalid input: failed to compute 'A0'; please check the input GCPs")
    fracl = (np.sqrt(5) - 1) / 2  # fraction long
    fracs = (3 - np.sqrt(5)) / 2  # fraction short
    if True:  # avoidable check for readability
        assert np.isclose(1 / fracl, fracl / fracs, atol=1.e-6) and np.isclose(1, fracl + fracs, atol=1.e-6)
    norm2 = nc ** 2 + nr ** 2
    k1asa2Min, k1asa2Max = -16 / (27 * norm2), 4 * 16 / (27 * norm2)  # upper limit is larger in the notes, but seems not convenient
    x0, x1 = k1asa2Min, k1asa2Max
    xb, xc = x0 + fracs * (x1 - x0), x0 + fracl * (x1 - x0)
    f0, fb, fc, f1 = [K1asa22Error_2410(cDs, rDs, xs, ys, zs, oc_, or_, item, A0=A0) for item in [x0, xb, xc, x1]]
    if np.argmin([f0, fb, fc, f1]) in [0, 3]:
        nTMP, nTMP_max, x1_max = 0, 100, 16 / norm2
        while nTMP < nTMP_max and x1 < x1_max:  # x0 not changed
            nTMP, x1 = nTMP + 20, x1 + 0.1 * (x1 - x0)  # WATCH OUT: epsilon; WATCH OUT: it was + 100 and 0.5 *
            xsTMP = np.linspace(x0, x1, nTMP)
            fsTMP = [K1asa22Error_2410(cDs, rDs, xs, ys, zs, oc_, or_, item, A0=A0) for item in xsTMP]  # WATCH OUT: expensive
            posMin = np.argmin(fsTMP)
            if 0 < posMin < nTMP - 1:
                if posMin >= 2:
                    x0, xb, xc, x1 = xsTMP[posMin-2], xsTMP[posMin-1], xsTMP[posMin-0], xsTMP[posMin+1]
                else:
                    x0, xb, xc, x1 = xsTMP[posMin-1], xsTMP[posMin+0], xsTMP[posMin+1], xsTMP[posMin+2]
                break
        if nTMP >= nTMP_max or x1 >= x1_max:
            return 0.  # IMP*
    while abs(x1 - x0) > 1.e-5 / norm2:  # WATCH OUT: epsilon
        if fb < fc:
            x1, f1 = xc, fc
            xc, fc = xb, fb
            xb = x0 + fracs * (x1 - x0)  # x1 has been updated
            fb = K1asa22Error_2410(cDs, rDs, xs, ys, zs, oc_, or_, xb, A0=A0)
        else:
            x0, f0 = xb, fb
            xb, fb = xc, fc
            xc = x0 + fracl * (x1 - x0)  # x0 has been updated
            fc = K1asa22Error_2410(cDs, rDs, xs, ys, zs, oc_, or_, xc, A0=A0)
    k1asa2 = (x0 + x1) / 2
    return k1asa2
def GCPsD2Pc(cDs, rDs, xs, ys, zs, errorC, A0=None, nc=None, nr=None, oc_=None, or_=None, pDesired=1-1.e-9, margin=0.5):  # for close range only; 2010-01-01; lm:2025-06-22; lr:2025-07-09
    rangeC = 'close'
    possG, k1asa2 = GoodGCPs_2504(cDs, rDs, xs, ys, zs, errorC, rangeC, nc=nc, nr=nr, oc_=oc_, or_=or_, pDesired=pDesired, margin=margin)
    if k1asa2 is None:
        return None, None, None, None, False
    cDs, rDs, xs, ys, zs = [item[possG] for item in [cDs, rDs, xs, ys, zs]]
    oc_, or_ = UpdateOCOR(oc_=oc_, or_=or_, nc=nc, nr=nr)
    cUs, rUs = CDRD2CURUForParabolicSquaredDistortion_2502(cDs, rDs, oc_, or_, k1asa2)[:2]  # WATCH OUT: all positions
    xc, yc, zc, seemsOK = GCPsU2Pc(cUs, rUs, xs, ys, zs, A0=A0)
    if not seemsOK:
        return None, None, None, None, False
    return xc, yc, zc, k1asa2, seemsOK
def GCPsU2Pc(cUs, rUs, xs, ys, zs, A=None, A0=None, A1=None):  # for close range only; 2010-01-01; lm:2025-06-21; lr:2025-07-09
    rangeC = 'close'
    Pa11 = CURUXYZ2Px_2502(cUs, rUs, xs, ys, zs, rangeC, A=A, A0=A0, A1=A1)
    if Pa11 is None:
        return None, None, None, False
    P = np.array([Pa11[:4], Pa11[4:8], np.append(Pa11[8:], 1.)])
    try:
        kernelP = np.linalg.svd(P)[-1][-1]
        assert len(kernelP) == 4
        xc, yc, zc = kernelP[:3] / kernelP[-1]
    except Exception:
        return None, None, None, False
    ds = DistancePointToPoint3D(xs, ys, zs, xc, yc, zc)
    seemsOK = np.median(ds) < 1.e+3 and zc > 0  # IMP*: epsilon
    return xc, yc, zc, seemsOK
def GHDroneExtractVideoToPathFldFrames(pathFldMain, active=True, extsVids=['mp4', 'avi', 'mov'], fps=0., round=True, stamp='millisecond', extImg='png', overwrite=False):  # lm:2025-06-30; lr:2025-07-10
    pathFldVid = os.path.join(pathFldMain, 'data')
    pathVid = GHLookFor0Or1PathVideoOrFail(pathFldVid, extsVids=extsVids)
    if pathVid == '':  # no video, there must be frames in pathFldDFrames
        pathFldFrames = os.path.join(pathFldVid, 'frames')  # IMP*: nomenclature
        if not os.path.exists(pathFldFrames) or len(os.listdir(pathFldFrames)) == 0:
            raise Exception("Unexpected condition: no video or frames found at '{}'".format(os.sep.join(pathFldVid.split(os.sep)[-2:])))  # WATCH OUT: [-2:]
    else:
        pathFldFrames = os.path.join(pathFldMain, 'scratch', 'frames')  # IMP*
        if active and (not os.path.exists(pathFldFrames) or len(os.listdir(pathFldFrames)) == 0 or overwrite):
            GHExtractVideo(pathVid, pathFldFrames, fps=fps, round=round, stamp=stamp, extImg=extImg, overwrite=overwrite)
    return pathFldFrames
def GHDronePlotExtrinsic_2504(pathFld, pathImg, fw, fh, fontsize, dpi, pathFldB=None, length_stamp=12, ending='cal.txt'):  # lm:2025-06-30; lr:2025-07-13
    rangeC = 'close'
    ts, *vars = PathFldVideoCal2TsAnd_2504(pathFld, rangeC, length_stamp=length_stamp, ending=ending)[:-1]
    if pathFldB is not None:
        tsB, *varsB = PathFldVideoCal2TsAnd_2504(pathFldB, rangeC, length_stamp=length_stamp, ending=ending)[:-1]
    else:
        tsB, varsB = ts, vars  # just useful
    ylabels = [r'$x$ [m]', r'$y$ [m]', r'$z$ [m]', r'$\phi$ [rad]', r'$\sigma$ [rad]', r'$\tau$ [rad]']
    vars, varsB, ylabels = [[item[0], item[3], item[1], item[4], item[2], item[5]] for item in [vars, varsB, ylabels]]
    plt.figure(figsize=(2*fw, 3*fh))
    plt.rcParams.update(LoadParamsC(fontsize=fontsize))
    for pos in range(6):
        plt.subplot(3, 2, pos+1)
        plt.plot(ts, vars[pos], 'ko')
        if pathFldB is not None:
            plt.plot(tsB, varsB[pos], 'b-')
        plt.xlabel(r'time [s]'); plt.ylabel(ylabels[pos])
    plt.tight_layout()
    os.makedirs(os.path.dirname(pathImg), exist_ok=True)
    plt.savefig(pathImg, dpi=dpi)
    plt.close()
    return None
def GHDroneReadTimestacksTxt(pathFldMain, ppm):  # lm:2025-07-08; lm:2025-07-14
    codesT = np.loadtxt(os.path.join(pathFldMain, 'data', 'timestacks_xyz.txt'), usecols=0, dtype=str)
    data = np.loadtxt(os.path.join(pathFldMain, 'data', 'timestacks_xyz.txt'), usecols=range(1, 4), dtype=float, ndmin=2)
    dXsT, dYsT, dZsT = {}, {}, {}
    for codeT in set(codesT):
        possC = np.where(codesT == codeT)[0]
        if len(possC) < 2:
            raise Exception("Invalid input: timestack '{}' must contain at least 2 points".format(codeT))
        dXsT[codeT], dYsT[codeT], dZsT[codeT] = [RefinePolyline3D({'xs': data[possC, 0], 'ys': data[possC, 1], 'zs': data[possC, 2]}, 1/ppm)[item] for item in ['xs', 'ys', 'zs']]
        pathOutTxt = os.path.join(pathFldMain, 'output', 'numerics', 'timestacks', 'timestack_{}_cxyz.txt'.format(codeT))
        os.makedirs(os.path.dirname(pathOutTxt), exist_ok=True)
        with open(pathOutTxt, 'w') as fileout:
            for pos in range(len(dXsT[codeT])):
                fileout.write('{:12.0f} {:15.3f} {:15.3f} {:15.3f}\t c, x, y and z\n'.format(pos, dXsT[codeT][pos], dYsT[codeT][pos], dZsT[codeT][pos]))  # WATCH OUT: formatting
    return dXsT, dYsT, dZsT
def GHExtractVideo(pathVid, pathFldFrames, fps=0., round=True, stamp='millisecond', extImg='png', overwrite=False):  # 2010-01-01; lm:2025-06-26; lr:2025-07-11
    if os.path.exists(pathFldFrames):
        if overwrite:
            CleanAFld_2504(pathFldFrames)
        else:
            return None
    pathFldVid, fnVid = os.path.split(pathVid)
    fpsVid, nOfFramesVid = PathVid2FPS(pathVid), PathVid2NOfFrames(pathVid)
    lenVid = (nOfFramesVid - 1) / fpsVid
    fps = RecomputeFPS_2502(fps, fpsVid, round=round)
    isFpsTheSame = np.isclose(fps, fpsVid, rtol=1.e-6)
    if not isFpsTheSame:
        nOfFramesApp = lenVid * fps + 1  # ~ approximation of the number of frames of the new video
        if nOfFramesApp <= 3:  # WATCH OUT: epsilon
            raise Exception("Invalid input: requested fps is too low for video '{}'; please specify fps > {:.4f}".format(fnVid, 4 / lenVid))  # WATCH OUT: formatting, epsilon
    fnVidTMP = '000_{}{}'.format(''.join(random.choices(string.ascii_letters, k=20)), os.path.splitext(fnVid)[1])
    pathVidTMP = os.path.join(pathFldVid, fnVidTMP)
    if os.path.exists(pathVidTMP):  # miracle
        os.remove(pathVidTMP)
    if not isFpsTheSame:
        shutil.move(pathVid, pathVidTMP)  # pathVidTMP is a backup of the original
        Vid2VidModified_2504(pathVidTMP, pathVid, fps=fps, round=round)  # IMP*: this pathVid has the desired fps
    PathVid2AllFrames_2506(pathVid, pathFldFrames, stamp=stamp, extImg=extImg)
    if not isFpsTheSame:
        shutil.move(pathVidTMP, pathVid)  # pathVid is the original
    return None
def GHInform_2506(UCode, pathFldMain, par, pos, margin=0, sB='*', nFill=10):  # lm:2025-06-19; lr:2025-07-14
    text0 = 'session started for'
    text1 = 'session finished successfully for' 
    textC = os.sep.join(pathFldMain.split(os.sep)[-2:])  # WATCH OUT: [-2:] formatting
    if len(text1) > len(text0):
        l0, l1 = len(text1) - len(text0), 0
    else:
        l0, l1 = 0, len(text0) - len(text1)
    if pos == 0:
        msg = '{} {} /{}'.format(UCode, text0, textC)
        print("\n{:s}\n".format(msg.center(len(msg)+nFill+l0, '_')))
        print("Parameters:")
        PrintDictionary_2506(par, margin=margin, sB=sB)
    else:
        msg = '{} {} /{}'.format(UCode, text1, textC)
        print("\n{:s}\n".format(msg.center(len(msg)+nFill+l1, '_')))
    return None
def GHLens2SelVarKeysOrFail(lens, lens2SelVarKeys):  # 2010-01-01; lm:2025-07-10; lr:2025-07-13
    try:
        selVarKeys = lens2SelVarKeys[lens]
    except Exception as eTMP:
        raise Exception("Invalid input: failed to obtain 'selVarKeys': {}".format(eTMP))
    return selVarKeys
def GHLoadDGit():  # lm:2025-06-30; lr:2025-07-13
    dGit = {}
    dGit |= {'ind': 2}
    dGit |= {'sOK': '\033[92m\u25CF\033[0m'}  # '\033[92m\u25CF\033[0m' '\U0001F7E2' '\033[92m✔\033[0m'
    dGit |= {'sWO': '\033[93m\u25CF\033[0m'}  # '\033[93m\u25CF\033[0m' '\U0001F7E0'
    dGit |= {'sKO': '\033[91m\u25CF\033[0m'}  # '\033[91m\u25CF\033[0m' '\U0001F534' '\033[31m✘\033[0m'
    dGit |= {'sB1': '\u2022'}  # bullet
    dGit |= {'sB2': '\u2023'}  # triangular bullet
    dGit |= {'sB3': '\u25E6'}  # white bullet
    dGit |= {'sB4': '\u2043'}  # hyphen bullet
    dGit |= {'sB5': '\u2219'}  # bullet operator
    dGit |= {'sB6': '\u25AA'}  # small black square
    dGit |= {'sB7': '\u25AB'}  # small white square
    dGit |= {'sB8': '\u25CF'}  # black circle
    dGit |= {'sB9': '\u25CB'}  # white circle
    dGit |= {'fontsize': 20}
    dGit |= {'fs': 8}  # figure size for scatter plot
    dGit |= {'fw': 10}  # figure width
    dGit |= {'fh': 4}  # figure height
    dGit |= {'dpiLQ': 100}
    dGit |= {'dpiHQ': 200}
    return dGit
def GHLoadPar(pathFldMain):  # lm:2025-07-08; lr:2025-07-14
    pathJson = os.path.join(pathFldMain, 'data', 'parameters.json')  # IMP*: nomenclature
    try:
        with open(pathJson, 'r') as f:
            par = json.load(f)
    except Exception as eTMP:
        raise Exception("Invalid input: failed to read '{}': {}".format(pathJson, eTMP))
    return par
def GHLookFor0Or1PathVideoOrFail(pathFld, extsVids=['mp4', 'avi', 'mov']):  # 2010-01-01; lm:2025-06-15; lr:2025-07-14
    pathsVids = [item.path for item in os.scandir(pathFld) if item.is_file() and os.path.splitext(item.name)[1][1:].lower() in extsVids]
    if len(pathsVids) == 0:
        pathVid = ''
    elif len(pathsVids) == 1:
        pathVid = pathsVids[0]
    else:
        raise Exception("Unexpected condition: more than one video found at '{}'".format(pathFld))
    return pathVid
def GoodGCPsGivenK1asa2_2504(cDs, rDs, xs, ys, zs, errorC, rangeC, A0=None, oc_=None, or_=None, k1asa2=None, pDesired=1-1.e-9, margin=0.5):  # lm:2025-07-14; lr:2025-07-14
    if rangeC == 'close':
        if any(item is None for item in [oc_, or_, k1asa2]):
            raise Exception("Invalid input: 'oc_', 'or_' and 'k1asa2' must be floats")
        cUs, rUs = CDRD2CURUForParabolicSquaredDistortion_2502(cDs, rDs, oc_, or_, k1asa2)[:2]  # WATCH OUT: all positions
        nOfPoints = 6
    elif rangeC == 'long':
        cUs, rUs = cDs, rDs
        nOfPoints = 4
    A = CURUXYZ2A_2502(cUs, rUs, xs, ys, zs, rangeC, A0=A0)
    b = CURU2B(cUs, rUs)
    iForRANSAC, nForRANSAC, possG = 0, np.inf, []
    while iForRANSAC < nForRANSAC:
        possH = np.random.choice(range(len(cUs)), size=nOfPoints, replace=False)
        poss01 = [2*item+0 for item in possH] + [2*item+1 for item in possH]  # WATCH OUT: works
        AH, bH = A[poss01, :], b[poss01]
        try:
            AHT = np.transpose(AH)
            PxH = np.linalg.solve(np.dot(AHT, AH), np.dot(AHT, bH))
            cUsH, rUsH = XYZ2CURU_2410(xs, ys, zs, PxH, rangeC)[:2]  # WATCH OUT: all positions
            errorsH = np.hypot(cUsH - cUs, rUsH - rUs)
        except Exception:
            continue
        possGH = np.where(errorsH <= errorC)[0]
        if len(possGH) > len(possG):
            possG = possGH
            pOutlier = 1 - len(possG) / len(cDs) + margin * len(possG) / len(cDs)  # margin in [0, 1): the higher, the more demanding
            nForRANSAC = NForRANSAC(pOutlier, pDesired, nOfPoints)
            if len(possG) == len(cDs):
                break
        iForRANSAC += 1
    return possG
def GoodGCPs_2504(cDs, rDs, xs, ys, zs, errorC, rangeC, nc=None, nr=None, oc_=None, or_=None, pDesired=1-1.e-9, margin=0.5):  # lm:2025-06-20; lr:2025-07-07
    if len(cDs) < {'close': 6, 'long': 4}[rangeC]:
        return np.asarray([], dtype=int), None
    A0 = XYZ2A0(xs, ys, zs)
    if rangeC == 'close':
        oc_, or_ = UpdateOCOR(oc_=oc_, or_=or_, nc=nc, nr=nr)
        k1asa2 = 0
        possG = GoodGCPsGivenK1asa2_2504(cDs, rDs, xs, ys, zs, 3*errorC, rangeC, A0=A0, oc_=oc_, or_=or_, k1asa2=k1asa2, pDesired=pDesired, margin=margin)
        if len(possG) == 0:
            return np.asarray([], dtype=int), None
        if True:  # avoidable check for readability
            assert np.allclose(possG, sorted(possG))
        cDsG, rDsG, xsG, ysG, zsG = [item[possG] for item in [cDs, rDs, xs, ys, zs]]
        A0G = A0[sorted([2*item for item in possG] + [2*item+1 for item in possG]), :]  # IMP*: sorted; checked 2025-07-07
        k1asa2 = GCPs2K1asa2_2410(cDsG, rDsG, xsG, ysG, zsG, oc_, or_, nc, nr, A0=A0G)
        possG = GoodGCPsGivenK1asa2_2504(cDs, rDs, xs, ys, zs, 1*errorC, rangeC, A0=A0, oc_=oc_, or_=or_, k1asa2=k1asa2, pDesired=pDesired, margin=margin)
        if len(possG) == 0:
            return np.asarray([], dtype=int), None
        if True:  # avoidable check for readability
            assert np.allclose(possG, sorted(possG))
        cDsG, rDsG, xsG, ysG, zsG = [item[possG] for item in [cDs, rDs, xs, ys, zs]]
        A0G = A0[sorted([2*item for item in possG] + [2*item+1 for item in possG]), :]  # IMP*: sorted
        k1asa2 = GCPs2K1asa2_2410(cDsG, rDsG, xsG, ysG, zsG, oc_, or_, nc, nr, A0=A0G)
    elif rangeC == 'long':
        k1asa2 = None
        possG = GoodGCPsGivenK1asa2_2504(cDs, rDs, xs, ys, zs, 1*errorC, rangeC, A0=A0, pDesired=pDesired, margin=margin)
    else:
        raise Exception("Invalid input: 'rangeC' ('{}') must be 'close' or 'long'".format(rangeC))
    return possG, k1asa2
def InformErrorsG_2506(xs, ys, zs, cDs, rDs, dMCS, errorC, codes=None, margin=0, sB='*', sWO='!'):  # 2010-01-01; lm:2025-07-01; lr:2025-07-14
    errorsG = ErrorsG_2504(xs, ys, zs, cDs, rDs, dMCS)
    for posErrorG, errorG in enumerate(errorsG):
        if errorG > errorC:
            if codes is not None:
                strTMP = 'code = {}'.format(codes[posErrorG])  # WATCH OUT: formatting
            else:
                strTMP = 'c = {:.3f} and r = {:.3f}'.format(cDs[posErrorG], rDs[posErrorG])  # WATCH OUT: formatting
            print('{}{} Error for GCP with {} is {:.2f} pixels {}'.format(' '*margin, sB, strTMP, errorG, sWO))  # WATCH OUT: formatting
    return None
def IntersectionOfTwoLines_2506(line0, line1, eps=1.e-14):  # 2000-01-01; lm:2025-06-11; lr:2025-06-30
    den = ClipWithSign(line0['lx'] * line1['ly'] - line1['lx'] * line0['ly'], eps, np.inf)  # WATCH OUT; epsilon
    xI = (line1['lt'] * line0['ly'] - line0['lt'] * line1['ly']) / den
    yI = (line0['lt'] * line1['lx'] - line1['lt'] * line0['lx']) / den
    return xI, yI
def IsFlModified_2506(pathFl, t0=None, margin=datetime.timedelta(seconds=2)):  # 2000-01-01; lm:2025-06-26; lr:2025-07-13
    if t0 is None:
        t0 = datetime.datetime.now()    
    tM = datetime.datetime.fromtimestamp(os.path.getmtime(pathFl))
    tC = datetime.datetime.fromtimestamp(os.path.getctime(pathFl))
    isFlModified = abs(tM - t0) <= margin or abs(tC - t0) <= margin
    return isFlModified
def IsFldModified_2506(pathFld, t0=None, margin=datetime.timedelta(seconds=2), recursive=False):  # 2000-01-01; lm:2025-06-26; lr:2025-07-13
    if t0 is None:
        t0 = datetime.datetime.now()    
    isFldModified = False
    if recursive:
        for root, _, fns in os.walk(pathFld):
            for fn in fns:
                pathFl = os.path.join(root, fn)
                if os.path.isfile(pathFl):
                    if IsFlModified_2506(pathFl, t0=t0, margin=margin):
                        return True
    else:
        for fn in os.listdir(pathFld):
            pathFl = os.path.join(pathFld, fn)
            if os.path.isfile(pathFl):
                if IsFlModified_2506(pathFl, t0=t0, margin=margin):
                    return True
    return isFldModified
def IsImg_2504(img):  # 2000-01-01; lm:2025-05-27; lr:2025-07-01
    isImg = True
    if not isinstance(img, np.ndarray):
        return False
    if not img.ndim >= 2:
        return False
    nr, nc = img.shape[:2]
    if not (nr > 0 and nc > 0):
        return False
    if img.dtype.kind not in ('u', 'i') or img.min() < 0 or img.max() > 255:
        return False
    return isImg
def K1asa22Error_2410(cDs, rDs, xs, ys, zs, oc_, or_, k1asa2, A0=None):  # for close range only; 2010-01-01; lm:2025-05-28; lm:2025-06-22
    rangeC = 'close'
    cUs, rUs = CDRD2CURUForParabolicSquaredDistortion_2502(cDs, rDs, oc_, or_, k1asa2)[:2]  # WATCH OUT: all positions
    Px = CURUXYZ2Px_2502(cUs, rUs, xs, ys, zs, rangeC, A0=A0)
    if Px is None:
        return np.inf
    cUsR, rUsR = XYZ2CURU_2410(xs, ys, zs, Px, rangeC)[:2]  # WATCH OUT: all positions
    error = RMSE2D_2506(cUs, rUs, cUsR, rUsR)
    return error
def Keypoints_2506(img, method='sift', mask=None, nOfFeatures=5000):  # 1900-01-01; lm:2025-06-23; lr:2025-07-11
    img = PathImgOrImg2Img(img)
    nr, nc = img.shape[:2]
    try:
        if method == 'sift':
            det = cv2.SIFT_create(nfeatures=nOfFeatures)
        elif method == 'orb':
            det = cv2.ORB_create(nfeatures=nOfFeatures, scoreType=cv2.ORB_FAST_SCORE)
        else:
            raise Exception("Invalid input: method ('{}') must be 'sift' or 'orb'".format(method))
        kps, des = det.detectAndCompute(img, mask)
        ctrl = kps is not None and des is not None and len(kps) == len(des) > 0
        if not ctrl:
            return None, None, None, None, False
    except Exception:
        return None, None, None, None, False
    return nc, nr, kps, des, ctrl
def LoadDBasSD_2506(selVarKeys, nc, nr, rangeC, zr=0., z0=0., dBasSI=None, xc=None, yc=None, zc=None, x0=None, y0=None):  # SD = Size Dependent; 2010-01-01; lr:2025-06-23; lr:2025-07-11
    if dBasSI is None:
        dBasSI = LoadDBasSI_2410(selVarKeys, rangeC, zr=zr, z0=z0) 
    dBasSD = {'nc': nc, 'nr': nr, 'dBasSI': dBasSI}
    dBasSD.update(dBasSI)  # IMP*: all the keys
    dRefVal, dRefRng, dScl = {}, {}, {}
    if rangeC == 'close':
        k1aC = 1.0  # experience; maximum expected absolute value
        scaC = 1.0 / np.hypot(nc, nr)  # experience
        ocC, orC = N2K_2410(nc), N2K_2410(nr)
        dRefVal['xc'], dRefRng['xc'], dScl['xc'] = xc, 2.0e+1, 1.0e+1
        dRefVal['yc'], dRefRng['yc'], dScl['yc'] = yc, 2.0e+1, 1.0e+1
        dRefVal['zc'], dRefRng['zc'], dScl['zc'] = zc, 2.0e+1, 1.0e+1
        dRefVal['ph'], dRefRng['ph'], dScl['ph'] = 0.*np.pi/2., np.pi/1., 1.0e+0
        dRefVal['sg'], dRefRng['sg'], dScl['sg'] = 0.*np.pi/2., np.pi/4., 1.0e+0
        dRefVal['ta'], dRefRng['ta'], dScl['ta'] = 1.*np.pi/2., np.pi/2., 1.0e+0
        dRefVal['k1a'], dRefRng['k1a'], dScl['k1a'] = 0.0e+0, k1aC, k1aC
        dRefVal['k2a'], dRefRng['k2a'], dScl['k2a'] = 0.0e+0, 1.0e-1, 1.0e-1
        dRefVal['p1a'], dRefRng['p1a'], dScl['p1a'] = 0.0e+0, 1.0e-2, 1.0e-2
        dRefVal['p2a'], dRefRng['p2a'], dScl['p2a'] = 0.0e+0, 1.0e-2, 1.0e-2
        dRefVal['sca'], dRefRng['sca'], dScl['sca'] = scaC, scaC / 2, scaC / 10
        dRefVal['sra'], dRefRng['sra'], dScl['sra'] = scaC, scaC / 2, scaC / 10
        dRefVal['oc'], dRefRng['oc'], dScl['oc'] = ocC, ocC / 2, ocC / 10
        dRefVal['or'], dRefRng['or'], dScl['or'] = orC, orC / 2, orC / 10
    elif rangeC == 'long':
        dRefVal['x0'], dRefRng['x0'], dScl['x0'] = x0, 1.0e+2, 1.0e+2
        dRefVal['y0'], dRefRng['y0'], dScl['y0'] = y0, 1.0e+2, 1.0e+2
        dRefVal['ph'], dRefRng['ph'], dScl['ph'] = 0.*np.pi/2., np.pi/1., 1.0e+0
        dRefVal['sg'], dRefRng['sg'], dScl['sg'] = 0.*np.pi/2., np.pi/4., 1.0e+0
        dRefVal['ta'], dRefRng['ta'], dScl['ta'] = 0.*np.pi/2., np.pi/2., 1.0e+0
        dRefVal['sc'], dRefRng['sc'], dScl['sc'] = 5.0e+0, 4.0e+0, 1.0e+0  # WATCH OUT
        dRefVal['sr'], dRefRng['sr'], dScl['sr'] = 5.0e+0, 4.0e+0, 1.0e+0  # WATCH OUT
    else:
        raise Exception("Invalid input: 'rangeC' ('{}') must be 'close' or 'long'".format(rangeC))
    dBasSD = dBasSD | {'dRefVal': dRefVal, 'dRefRng': dRefRng, 'dScl': dScl}
    return dBasSD
def LoadDBasSI_2410(selVarKeys, rangeC, zr=0., z0=0.):  # SI = Size Independent; 2010-01-01; lr:2025-04-28; lr:2025-07-11
    minVarKeys, allVarKeys = MinVarKeys(rangeC), AllVarKeys(rangeC)
    assert set(minVarKeys) <= set(selVarKeys) <= set(allVarKeys)
    dBasSI = {'selVarKeys': selVarKeys, 'rangeC': rangeC, 'zr': zr, 'z0': z0} 
    dBasSI = dBasSI | {'minVarKeys': minVarKeys, 'allVarKeys': allVarKeys}
    dBasSI = dBasSI | {'rEarth': REarth_2410(), 'oHorizon': OHorizon_2410()}
    return dBasSI
def LoadDMCSFromCalTxt_2502(pathCalTxt, rangeC, incHor=True, zr=0.):  # 2010-01-01; lm:2025-05-01; lr:2025-07-11
    allVar, nc, nr = ReadCalTxt_2410(pathCalTxt, rangeC)[:3]
    dMCS = AllVar2DMCS_2410(allVar, nc, nr, rangeC, incHor=incHor, zr=zr)
    return dMCS
def LoadDPdfTxt_2504(pathFile=None, xUL=None, yUL=None, angle=None, xyLengthInC=None, xyLengthInR=None, ppm=None):  # lm:2025-02-07; lr:2025-06-23
    if any(item is None for item in [xUL, yUL, angle, xyLengthInC, xyLengthInR, ppm]):
        xUL, yUL, angle, xyLengthInC, xyLengthInR, ppm = ReadPdfTxt_2410(pathFile)
    dPdf = {'pathFile': pathFile, 'xUL': xUL, 'yUL': yUL, 'angle': angle, 'xyLengthInC': xyLengthInC, 'xyLengthInR': xyLengthInR, 'ppm': ppm}
    nc, nr = [int(dPdf[item] * dPdf['ppm']) + 1 for item in ['xyLengthInC', 'xyLengthInR']]  # IMP*
    dPdf = dPdf | {'nc': nc, 'nr': nr, 'nOfPixels': nc * nr}
    xyLengthInC, xyLengthInR = [(item - 1) / dPdf['ppm'] for item in [nc, nr]]  # IMP*
    dPdf = dPdf | {'xyLengthInC': xyLengthInC, 'xyLengthInR': xyLengthInR}  # overwrites
    csBasic, rsBasic = [np.arange(item) for item in [nc, nr]]
    dPdf = dPdf | {'csBasic': csBasic, 'rsBasic': rsBasic}
    mcs, mrs = np.meshgrid(csBasic, rsBasic)
    csAll, rsAll = np.reshape(mcs, -1), np.reshape(mrs, -1)
    dPdf = dPdf | {'cs': csAll, 'rs': rsAll}
    csC, rsC = np.asarray([0, nc-1, nc-1, 0]), np.asarray([0, 0, nr-1, nr-1])
    xsC, ysC = PlanCR2XY_2504(csC, rsC, xUL=xUL, yUL=yUL, angle=angle, ppm=ppm)[:2]  # WATCH OUT; all positions
    dPdf = dPdf | {'csC': csC, 'rsC': rsC, 'xsC': xsC, 'ysC': ysC}
    ACR2XY = FindAffineA01_2504(csC, rsC, xsC, ysC)
    AXY2CR = FindAffineA01_2504(xsC, ysC, csC, rsC)
    dPdf = dPdf | {'ACR2XY': ACR2XY, 'AXY2CR': AXY2CR}
    xsAll, ysAll = PlanCR2XY_2504(csAll, rsAll, ACR2XY=ACR2XY)[:2]
    dPdf = dPdf | {'xs': xsAll, 'ys': ysAll}
    xsU, ysU = PlanCR2XY_2504(csBasic, rsBasic[+0] * np.ones(nc), ACR2XY=ACR2XY)[:2] # up
    xsD, ysD = PlanCR2XY_2504(csBasic, rsBasic[-1] * np.ones(nc), ACR2XY=ACR2XY)[:2] # down
    xs0, ys0 = PlanCR2XY_2504(csBasic[+0] * np.ones(nr), rsBasic, ACR2XY=ACR2XY)[:2] # left
    xs1, ys1 = PlanCR2XY_2504(csBasic[-1] * np.ones(nr), rsBasic, ACR2XY=ACR2XY)[:2] # right
    dPdf = dPdf | {'polylineU': {'xs': xsU, 'ys': ysU}, 'polylineD': {'xs': xsD, 'ys': ysD}}
    dPdf = dPdf | {'polyline0': {'xs': xs0, 'ys': ys0}, 'polyline1': {'xs': xs1, 'ys': ys1}}
    return dPdf
def LoadParamsC(fontsize=24):  # lm:20205-06-24; lr:2025-06-24
    paramsC = {'font.size': fontsize, 'axes.titlesize': fontsize, 'axes.labelsize': fontsize, 'xtick.labelsize': fontsize, 'ytick.labelsize': fontsize, 'legend.fontsize': fontsize, 'figure.titlesize': fontsize}
    return paramsC
def ManualCalibrationOfSeveralImages_2502(dBasSD, dsGH1s, dGvnVar, freDVarKeys, freUVarKeys, dsMCSs, nOfSeeds=20):  # lm:2025-05-19; lr:2025-07-14
    if not len(dsGH1s) == len(dsMCSs):
        raise Exception("Invalid input: 'dsGH1s' and 'dsMCSs' must have the same length")
    nOfImgs = len(dsGH1s)
    errorTs = np.asarray([ErrorT_2410(dsGH1s[pos], dsMCSs[pos], dsMCSs[pos]['dHor']) for pos in range(nOfImgs)])
    errorT = np.sqrt(np.mean(errorTs ** 2))
    freDVars, freUVars = [], []
    for pos in range(nOfImgs):
        assert dsMCSs[pos]['nc'] == dBasSD['nc'] and dsMCSs[pos]['nr'] == dBasSD['nr']
        freDVars.append(AllVar2SubVar_2410(dsMCSs[pos]['allVar'], freDVarKeys, dBasSD['rangeC']))
        freUVars.append(AllVar2SubVar_2410(dsMCSs[pos]['allVar'], freUVarKeys, dBasSD['rangeC']))
    freUVar = np.average(np.asarray(freUVars), axis=0)  # IMP*
    if True:  # avoidable check for readability
        assert freUVar.shape == freUVars[0].shape
    dFreUVar = Array2Dictionary(freUVarKeys, freUVar)
    for _ in range(4):
        for pos in range(nOfImgs):
            dsMCSs[pos], errorTs[pos] = ManualCalibration_2502(dBasSD, dsGH1s[pos], dGvnVar | dFreUVar, dsMCSs)
            freDVars[pos] = AllVar2SubVar_2410(dsMCSs[pos]['allVar'], freDVarKeys, dBasSD['rangeC'])
        errorTN = np.sqrt(np.mean(errorTs ** 2))
        if errorTN < errorT * (1 - 1.e-3):  # WATCH OUT: epsilon 2025-07-14
            errorT = errorTN
        else:
            break
        freUVar = UpdateFreUVar(dBasSD, dsGH1s, dGvnVar, freDVars, freDVarKeys, freUVar, freUVarKeys, nOfSeeds=nOfSeeds)[0]
        dFreUVar = Array2Dictionary(freUVarKeys, freUVar)
    return dsMCSs, errorTs
def ManualCalibration_2502(dBasSD, dGH1, dGvnVar, dsMCSs, dtMaxInSec=60):  # lm:2025-06-25; lr:2025-07-14
    dtMax = datetime.timedelta(seconds=dtMaxInSec)
    freVarKeys = [item for item in dBasSD['selVarKeys'] if item not in dGvnVar]
    if len(dGH1['xs']) < int((len(freVarKeys) + 1) / 2):
        return None, np.inf
    if len(dsMCSs) > 0:
        dMCSSeed, errorTSeed = ReadAFirstSeed_2410(dBasSD, dGH1, dGvnVar, dsMCSs)  # can give None, np.inf
    else:
        dMCSSeed, errorTSeed = None, np.inf  # IMP*
    if dMCSSeed is None or errorTSeed > 0.2 * np.hypot(dBasSD['nc'], dBasSD['nr']):  # WATCH OUT: epsilon
        dMCSSeed, errorTSeed = FindAFirstSeed_2502(dBasSD, dGH1, dGvnVar, dtMaxInSec=dtMaxInSec)
    if dMCSSeed is None:
        return None, np.inf  # IMP*
    freVarSeed = AllVar2SubVar_2410(dMCSSeed['allVar'], freVarKeys, dBasSD['rangeC'])
    sclFreVarSeed = SclVar_2410(freVarSeed, freVarKeys, dBasSD['dScl'], 'scale')
    theArgs = {'dBasSD': dBasSD, 'dGH1': dGH1, 'subVarKeys': freVarKeys, 'dRemVar': dGvnVar}
    sclFreVar, dMCS, errorT = sclFreVarSeed, dMCSSeed, errorTSeed
    counter, counterMax, datetime0 = 0, np.inf, datetime.datetime.now()
    while counter < counterMax and datetime.datetime.now() - datetime0 < dtMax:
        prtFactor, counterMax = PrtFactorAndNOfSeeds_2502(errorT)
        if counter == 0:  # IMP*: to ensure that the seed is considered
            prtFactor = 0.
        sclFreVarP = PrtSclVar_2410(sclFreVar, freVarKeys, dBasSD['dRefRng'], dBasSD['dScl'], prtFactor=prtFactor)
        errorTP = SclSubVar2FTM_2502(sclFreVarP, theArgs)
        if errorTP > 0.4 * np.hypot(dBasSD['nc'], dBasSD['nr']):  # WATCH OUT: epsilon
            continue
        try:
            sclFreVarP = optimize.minimize(SclSubVar2FTM_2502, sclFreVarP, args=theArgs, callback=MinimizeStopper(5.)).x  # WATCH OUT: time = 1, 5, 10?
        except Exception:
            continue
        if not AreVarOK_2410(sclFreVarP, freVarKeys, areScl=True, dScl=dBasSD['dScl']):
            continue
        errorTP = SclSubVar2FTM_2502(sclFreVarP, theArgs)
        if errorTP < errorT:
            dMCSP = SclSubVar2DMCS_2502(dBasSD, sclFreVarP, freVarKeys, dGvnVar, incHor=True)  # 2025-04-14
            if True:  # avoidable check for readability
                assert np.isclose(errorTP, ErrorT_2410(dGH1, dMCSP, dMCSP['dHor']))
            nOfRightSide = len(XYZ2PossRightSideOfCamera_2410(dGH1['xs'], dGH1['ys'], dGH1['zs'], dBasSD['rangeC'], dCamVar=dMCSP['dAllVar'], ef=dMCSP['ef']))
            if not (nOfRightSide == len(dGH1['xs']) == len(dGH1['ys']) == len(dGH1['zs'])):
                continue
            sclFreVar, dMCS, errorT = sclFreVarP, dMCSP, errorTP
        counter += 1
        if errorT < min(1, 0.0001 * np.hypot(dBasSD['nc'], dBasSD['nr'])):  # WATCH OUT: epsilon
            break
    return dMCS, errorT
def Matches_2506(kps1, des1, kps2, des2, method='sift', max_error=np.inf, nOfStd=2.0, threshold_ratio=0.75, mutual_check=True):  # 1900-01-01; lm:2025-06-23; lr:2025-07-11
    if des1 is None or des2 is None:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    cs1, rs1 = [np.asarray([item.pt[pos] for item in kps1]) for pos in [0, 1]]
    cs2, rs2 = [np.asarray([item.pt[pos] for item in kps2]) for pos in [0, 1]]
    if method == 'sift':
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches12 = [m for m, n in bf.knnMatch(des1, des2, k=2) if m.distance < threshold_ratio * n.distance]
        matches21 = [m for m, n in bf.knnMatch(des2, des1, k=2) if m.distance < threshold_ratio * n.distance]
        if mutual_check:
            matches12 = [m12 for m12 in matches12 if any(m21.queryIdx == m12.trainIdx and m21.trainIdx == m12.queryIdx for m21 in matches21)]
            matches21 = []
    elif method == 'orb':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=not mutual_check)
        matches12 = bf.match(des1, des2)
        matches21 = bf.match(des2, des1)
        if mutual_check:
            matches12 = [m12 for m12 in matches12 if any(m21.queryIdx == m12.trainIdx and m21.trainIdx == m12.queryIdx for m21 in matches21)]
            matches21 = []
    poss1 = [m12.queryIdx for m12 in matches12] + [m21.trainIdx for m21 in matches21]
    poss2 = [m12.trainIdx for m12 in matches12] + [m21.queryIdx for m21 in matches21]
    cs1, rs1, cs2, rs2 = cs1[poss1], rs1[poss1], cs2[poss2], rs2[poss2]
    ers = np.asarray([m12.distance for m12 in matches12] + [m21.distance for m21 in matches21])
    if len(cs1) > 0:
        ds = np.hypot(cs1 - cs2, rs1 - rs2)
        possG = np.where((ers < max_error) & (ds < np.mean(ds) + nOfStd * np.std(ds) + 1.e-6))[0]
        cs1, rs1, cs2, rs2, ers = [item[possG] for item in [cs1, rs1, cs2, rs2, ers]]
    return cs1, rs1, cs2, rs2, ers
def MatrixK_2410(dCaSVar, rangeC):  # 1900-01-01; lm:2025-05-01; lr:2025-06-21
    if rangeC == 'close':  # K*
        sca, sra = [ClipWithSign(dCaSVar[item], 1.e-14, np.inf) for item in ['sca', 'sra']]
        K = np.asarray([[1/sca, 0, dCaSVar['oc']], [0, 1/sra, dCaSVar['or']], [0, 0, 1]])
    elif rangeC == 'long':  # K
        sc, sr = [ClipWithSign(dCaSVar[item], 1.e-14, np.inf) for item in ['sc', 'sr']]
        K = np.asarray([[1/sc, 0, dCaSVar['oc']], [0, 1/sr, dCaSVar['or']], [0, 0, 1]])
    else:
        raise Exception("Invalid input: 'rangeC' ('{}') must be 'close' or 'long'".format(rangeC))
    if False:  # avoidable check for readability
        assert np.isclose(K[0, 2], dCaSVar['oc']) and np.isclose(K[2, 0], 0)
        assert np.isclose(K[1, 2], dCaSVar['or']) and np.isclose(K[2, 1], 0)
    return K
def MatrixR_2502(dAngVar):  # 1900-01-01; lr:2025-04-30; lr:2025-06-21
    eu, ev, ef = UnitVectors_2502(dAngVar)
    R = np.asarray([eu, ev, ef])
    if False:  # avoidable check for readability
        assert np.allclose(eu, R[0, :]) and np.allclose(ev, R[1, :]) and np.allclose(ef, R[2, :])
    return R
def MatrixRt_2410(dExtVar, rangeC):  # 1900-01-01; lm:2025-04-30; lm:2025-06-21
    if rangeC == 'close':
        xH, yH, zH = [dExtVar[item] for item in ['xc', 'yc', 'zc']]
    elif rangeC == 'long':
        xH, yH, zH = [dExtVar[item] for item in ['x0', 'y0', 'z0']]
    else:
        raise Exception("Invalid input: 'rangeC' ('{}') must be 'close' or 'long'".format(rangeC))
    R = MatrixR_2502(dExtVar)  # dAngVar < dExtVar
    t = -np.dot(R, np.asarray([xH, yH, zH]))  # the rows of R are eu, ev and ef
    if False:  # avoidable check for readability
        eu, ev, ef = UnitVectors_2502(dExtVar)
        assert np.allclose(t, np.asarray([-xH*item[0]-yH*item[1]-zH*item[2] for item in [eu, ev, ef]]))
    Rt = np.zeros((3, 4))
    Rt[:, :3], Rt[:, 3] = R, t
    if False:  # avoidable check for readability
        assert Rt.shape == (3, 4) and np.allclose(Rt[:, :3], R) and np.allclose(Rt[:, 3], t)
    return Rt
def MeanAndSigmaOfImages(pathsImgs, fSigma=1.):  # 2000-10-21; lm:2025-07-02; lr:2025-07-13
    if not pathsImgs:
        raise ValueError("! pathsImgs is empty")
    for posPathImg, pathImg in enumerate(pathsImgs):  # Welford's algorithm; 2025-04-03 checked aside
        img = cv2.imread(pathImg).astype(np.float64)
        if posPathImg == 0:
            mean = img.copy()
            M2 = np.zeros_like(img)
        else:
            delta = img - mean
            mean += delta / (posPathImg + 1)
            M2 += delta * (img - mean)
    imgMean = np.clip(mean.round(), 0, 255).astype(np.uint8)
    imgSigma = np.clip((fSigma * np.sqrt(M2 / len(pathsImgs))).round(), 0, 255).astype(np.uint8)  # WATCH OUT: visual interest
    return imgMean, imgSigma
def MinVarKeys(rangeC):  # 2010-01-01; lr:2025-04-28; lr:2025-07-11
    if rangeC == 'close':
        minVarKeys = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'sca']  # WATCH OUT: cannot be changed
        if True:  # avoidable check for readability
            assert len(minVarKeys) == 7
    elif rangeC == 'long':
        minVarKeys = ['x0', 'y0', 'z0', 'ph', 'sg', 'ta', 'sc']  # WATCH OUT: cannot be changed
        if True:  # avoidable check for readability
            assert len(minVarKeys) == 7
    else:
        raise Exception("Invalid input: 'rangeC' ('{}') must be 'close' or 'long'".format(rangeC))
    return minVarKeys
def N2K_2410(n):  # 1900-01-01; lm:2025-05-27; lr:2025-07-11
    k = (n - 1) / 2
    return k
def NForRANSAC(pOutlier, pDesired, nOfPoints, eps=1.e-12):  # 1900-01-01; lm:2025-05-28; lr:2025-06-23
    num = np.log(np.clip(1 - pDesired, eps, 1-eps))
    den = np.log(np.clip(1 - (1 - pOutlier) ** nOfPoints, eps, 1-eps))
    N = int(np.ceil(num / den))
    return N
def OHorizon_2410():  # 2000-01-01; lm:2025-05-27; lr:2025-07-11
    oHorizon = 5
    return oHorizon
def OpenVideoOrFail_2504(pathVid):  # 2000-01-01; lm:2025-05-28; lr:2025-07-14
    vid = cv2.VideoCapture(pathVid)
    if not vid.isOpened():
        raise Exception("Invalid input: failed to read '{}'".format(pathVid))
    return vid
def PathFldVideoCal2FilterExtrinsic_2504(pathFld0, pathFld1, rangeC, filtering_length, nsOfStds=[5, 4, 3], length_stamp=12, ending='cal.txt'):  # lm:2025-06-26; lr:2025-07-13
    ts, *varss, pathsFns = PathFldVideoCal2TsAnd_2504(pathFld0, rangeC, length_stamp=length_stamp, ending=ending)
    if True:  # avoidable check for readability
        assert pathsFns == sorted(pathsFns)
    for nOfStd in sorted(nsOfStds)[::-1]:
        possBNOfStdAll = np.asarray([], dtype=int)
        for vars in varss:  # xcs, ycs, zcs, phs, sgs, tas
            varsF = FilterNotEquispacedXYData_2502(ts, vars, filtering_length)
            rmse = RMSE1D_2506(vars, varsF)
            possBNOfStdVar = np.where(np.abs(vars - varsF) > nOfStd * rmse)[0]
            possBNOfStdAll = np.unique(np.concatenate((possBNOfStdAll, possBNOfStdVar)))
        possBNOfStdAll = [item for item in possBNOfStdAll if item not in [0, len(ts)-1]]  # bad positions
        possGNOfStd = np.asarray([item for item in range(len(ts)) if item not in possBNOfStdAll], dtype=int)
        varss[:] = [np.interp(ts, ts[possGNOfStd], vars[possGNOfStd]) for vars in varss]
    for posPathFN, pathFn in enumerate(pathsFns):
        allVar, nc, nr, errorT = ReadCalTxt_2410(pathFn, rangeC)
        for pos in range(6):  # xcs, ycs, zcs, phs, sgs, tas
            allVar[pos] = varss[pos][posPathFN]  # WATCH OUT: a capon
        pathFn1 = os.path.join(pathFld1, os.path.split(pathFn)[1])
        WriteCalTxt_2410(pathFn1, allVar, nc, nr, errorT, rangeC)
    return None
def PathFldVideoCal2TsAnd_2504(pathFld, rangeC, length_stamp=12, ending='cal.txt'):  # 2000-01-01; lm:2025-06-04; lr:2025-07-13
    pathsFns = sorted([item.path for item in os.scandir(pathFld) if item.is_file() and item.name.endswith(ending)])
    ts, xcs, ycs, zcs, phs, sgs, tas = [[] for _ in range(7)]
    for pathFn in pathsFns:
        try:
            ts.append(int(pathFn[-length_stamp-len(ending):-len(ending)]) / 1000)  # WATCH OUT: milliseconds to seconds
        except Exception:
            raise Exception("Invalid input: unable to extract timestamp from '{}'".format(pathFn))
        xc, yc, zc, ph, sg, ta = ReadCalTxt_2410(pathFn, rangeC)[0][:6]  # IMP*: nomenclature
        xcs.append(xc); ycs.append(yc); zcs.append(zc)
        phs.append(ph); sgs.append(sg); tas.append(ta)
    ts, xcs, ycs, zcs, phs, sgs, tas = map(np.asarray, [ts, xcs, ycs, zcs, phs, sgs, tas])
    return ts, xcs, ycs, zcs, phs, sgs, tas, pathsFns
def PathImgOrImg2Img(img):  # 2000-01-01; lm:2025-05-27; lr:2025-07-01
    if isinstance(img, str):
        img = cv2.imread(img)  # WATCH OUT: can return None without raising an error
    if not IsImg_2504(img):
        raise Exception("Invalid input: invalid path or image")
    return img
def PathVid2AllFrames_2506(pathVid, pathFldFrames, stamp='millisecond', extImg='png'):  # 2010-01-01; lm:2025-06-11; lr:2025-06-30
    pathFldFramesP = Path(pathFldFrames)
    pathFldFramesP.mkdir(parents=True, exist_ok=True)
    pattern = str(pathFldFramesP / f"frame_%06d.{extImg}")
    cmd = ["ffmpeg", "-i", pathVid, "-vsync", "0", pattern]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    fnVidWE = os.path.splitext(os.path.split(pathVid)[1])[0]
    if stamp == 'millisecond':
        fpsVid = PathVid2FPS(pathVid)
    fns = sorted([item for item in os.listdir(pathFldFrames)])  # IMP*; sorted
    for posFn, fn in enumerate(fns):
        if stamp == 'millisecond':
            millisecond = int(posFn * 1000 / fpsVid)  # WATCH OUT; IMP*; int()
            fnNew = '{:}_{:}.{:}'.format(fnVidWE, str(millisecond).zfill(12), extImg)  # IMP*; nomenclature
        elif stamp == 'counter':  # counter
            fnNew = '{:}_{:}.{:}'.format(fnVidWE, str(posFn).zfill(12), extImg)  # IMP*; nomenclature
        else:
            raise Exception("Invalid input: 'stamp' must be 'millisecond' or 'counter'")
        os.rename(os.path.join(pathFldFrames, fn), os.path.join(pathFldFrames, fnNew))
    return None
def PathVid2FPS(pathVid):  # 2000-01-01; lm:2025-05-28; lr:2025-07-14
    vid = OpenVideoOrFail_2504(pathVid)
    fps = vid.get(cv2.CAP_PROP_FPS)
    vid.release()
    return fps
def PathVid2NOfFrames(pathVid):  # 2000-01-01; lm:2025-05-28; lr:2025-07-14
    vid = OpenVideoOrFail_2504(pathVid)
    nOfFrames = int(np.round(vid.get(cv2.CAP_PROP_FRAME_COUNT)))
    vid.release()
    return nOfFrames
def PathVid2NcNr(pathVid):  # 2000-01-01; lm:2025-05-28; lr:2025-07-09
    vid = OpenVideoOrFail_2504(pathVid)
    nc = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    nr = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid.release()
    return nc, nr
def Pixels2BandsInGrid_2506(cs, rs, nc, nr, nOfBands=None, nOfCBands=None, nOfRBands=None):  # 2000-01-01; lm:2025-07-07; lr:2025-07-13
    if len(cs) == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=int), np.asarray([], dtype=int)
    if min(cs) < -1/2 or max(cs) > nc-1/2 or min(rs) < -1/2 or max(rs) > nr-1/2:
        raise Exception("Invalid input: 'cs' and 'rs' must be within the image boundaries")
    if nOfBands is None and (nOfCBands is None or nOfRBands is None):
        raise Exception("Invalid input: either 'nOfBands' or both 'nOfCBands' and 'nOfRBands' must be provided")
    if nOfCBands is None:
        nOfCBands = nOfBands
    if nOfRBands is None:
        nOfRBands = nOfBands
    bandCs = ((cs+1/2) * nOfCBands / nc).astype(int)  # cs=-1/2 -> bandCs=0 starts; cs=nc-1/2 -> bandCs=nOfCBands starts
    bandRs = ((rs+1/2) * nOfRBands / nr).astype(int)  # rs=-1/2 -> bandRs=0 starts; rs=nr-1/2 -> bandRs=nOfRBands starts
    bandCs = np.clip(bandCs, 0, nOfCBands-1)  # points in the border
    bandRs = np.clip(bandRs, 0, nOfRBands-1)  # points in the border
    bandGs = bandCs * nOfRBands + bandRs
    return bandCs, bandRs, bandGs
def PlanCR2XY_2504(cs, rs, ACR2XY=None, xUL=None, yUL=None, angle=None, ppm=None, rtrnPossG=False, margin=0, nc=None, nr=None):  # lm:2025-03-31; lr:2025-07-02
    if ACR2XY is not None:  # apply affine transformation from cr to xy; it is actually simpler than an affine transformation
        xs, ys = ApplyAffineA01_2504(ACR2XY, cs, rs)
    else:  # IMP*; angle 0 = East; angle pi/2 = North
        xs = xUL + (np.cos(angle) * cs + np.sin(angle) * rs) / ppm
        ys = yUL + (np.sin(angle) * cs - np.cos(angle) * rs) / ppm
    if rtrnPossG:
        possG = CR2PossWithinImage_2502(cs, rs, nc, nr, margin=margin)
    else:
        possG = np.asarray([])
    return xs, ys, possG
def PlotCalibration_2504(img, dMCS, cDs, rDs, xs, ys, zs, cDhs, rDhs, pathImgOut):  # 2010-01-01; lm:2025-05-15; lr:2025-07-13
    img = PathImgOrImg2Img(img)
    nc = img.shape[1]
    cDsR, rDsR = XYZ2CDRD_2410(xs, ys, zs, dMCS)[:2]  # WATCH OUT: all positions
    img = DisplayCRInImage_2504(img, cDs, rDs, factor=0.75, colors=[[0, 0, 0]])  # black; clicked
    img = DisplayCRInImage_2504(img, cDsR, rDsR, factor=0.50, colors=[[0, 255, 255]])  # yellow; recovered from calibration and x, y and z
    if dMCS['rangeC'] == 'close':
        cDhsR, rDhsR = np.arange(nc), CDh2RDh_2410(np.arange(nc), dMCS['dHor'])[0]
        img = DisplayCRInImage_2504(img, cDhsR, rDhsR, factor=0.10, colors=[[0, 255, 255]])  # yellow; recovered from calibration
        img = DisplayCRInImage_2504(img, cDhs, rDhs, factor=0.75, colors=[[0, 0, 0]])  # black; clicked
    os.makedirs(os.path.split(pathImgOut)[0], exist_ok=True)
    cv2.imwrite(pathImgOut, img)
    return None
def Poss0AndPoss1InFind2DTransform_2504(n): # 1900-01-01; lm:2025-05-28; lr:2025-06-27
    aux = np.arange(n)  # array([], dtype=int64) if n == 0
    poss0 = 2 * aux + 0
    poss1 = 2 * aux + 1
    return poss0, poss1
def PrintDictionary_2506(theDictionary, margin=0, sB='*'):  # 2020-01-01; lm:2025-06-19; lr:2025-07-09
    if not theDictionary:
        print("{:{}}{} <empty dictionary>".format('', margin, sB))
        return None
    lMax = max(max(len(str(item)) for item in theDictionary) + 5, 30)  # WATCH OUT; epsilon
    for key in theDictionary:
        print("{:{}}{} __{:_<{}} {}".format('', margin, sB, key, lMax, theDictionary[key]))
    return None
def PrtFactorAndNOfSeeds_2502(errorT):  # related to dBasSD; 1900-01-01; lm:2025-05-13; lr:2025-07-14
    log10ErrorT = np.log10(max(errorT, 1))  # IMP*
    prtFactor = 0.2 + 0.8 * log10ErrorT  # IMP*
    nOfSeeds = int(5 + 15 * log10ErrorT)  # IMP*
    return prtFactor, nOfSeeds
def PrtSclVar_2410(sclVar, varKeys, dRefRng, dScl, prtFactor=1.):  # 2010-01-01; lm:2025-05-13; lr:2025-06-23
    var = SclVar_2410(sclVar, varKeys, dScl, 'unscale')
    dVar = Array2Dictionary(varKeys, var)
    for key in varKeys:
        dVar[key] = dVar[key] + prtFactor * np.random.uniform(-dRefRng[key], +dRefRng[key])
        if key == 'zc' and prtFactor > 0:
            dVar[key] = max(dVar[key], 1)  # WATCH OUT: epsilon
    var = Dictionary2Array(varKeys, dVar)
    sclVar = SclVar_2410(var, varKeys, dScl, 'scale')
    return sclVar
def REarth_2410():  # 2000-01-01; lm:2025-05-27; lr:2025-07-11
    rEarth = 6.371e+6
    return rEarth
def RMSE1D_2506(xs, xsR):  # lm:2025-06-18; lr:2025-07-13
    rmse = np.sqrt(np.mean((xs - xsR) ** 2))
    return rmse
def RMSE2D_2506(xs, ys, xsR, ysR):  # lm:2025-06-18; lr:2025-07-07
    rmse = np.sqrt(np.mean((xs - xsR) ** 2 + (ys - ysR) ** 2))
    return rmse
def ReadAFirstSeed_2410(dBasSD, dGH1, dGvnVar, dsMCSs):  # 2010-01-01; lm:2025-05-19; lr:2025-06-23
    selVarKeys = dBasSD['selVarKeys']
    freVarKeys = [item for item in selVarKeys if item not in dGvnVar]
    rangeC, nc, nr, zr, z0 = [dBasSD[item] for item in ['rangeC', 'nc', 'nr', 'zr', 'z0']]
    dMCSSeed, errorTSeed = None, np.inf
    for dMCSH in dsMCSs:
        if dMCSH == {} or dMCSH is None or dMCSH['rangeC'] != dBasSD['rangeC']:
            continue
        freVar = AllVar2SubVar_2410(dMCSH['allVar'], freVarKeys, rangeC)
        allVar = SubVar2AllVar_2410(freVar, freVarKeys, dGvnVar, selVarKeys, rangeC, nc=nc, nr=nr, z0=z0)
        dMCSH = AllVar2DMCS_2410(allVar, nc, nr, rangeC, incHor=True, zr=zr)
        errorTH = ErrorT_2410(dGH1, dMCSH, dMCSH['dHor'])
        if errorTH < errorTSeed:
            dMCSSeed, errorTSeed = dMCSH, errorTH  # WATCH OUT: there was copy.deepcopy
    return dMCSSeed, errorTSeed
def ReadCalTxt_2410(pathCalTxt, rangeC):  # 1900-01-01; lm:2025-05-01; lr:2025-07-14
    data = np.loadtxt(pathCalTxt, usecols=0, dtype=float, ndmin=1)
    if rangeC == 'close':
        lenAllVar = 14
    elif rangeC == 'long':
        lenAllVar = 8
    else:
        raise Exception("Invalid input: 'rangeC' ('{}') must be 'close' or 'long'".format(rangeC))
    if len(data) < lenAllVar+3:
        raise Exception("Invalid input: unable to read file at '{}'".format(pathCalTxt))
    allVar, nc, nr, errorT = data[:lenAllVar], int(np.round(data[lenAllVar])), int(np.round(data[lenAllVar+1])), data[lenAllVar+2]
    return allVar, nc, nr, errorT
def ReadCdgTxt_2502(pathCdgTxt, readCodes=False, readOnlyGood=True, nc=np.inf, nr=np.inf, margin=0):  # 1900-01-01; lm:2025-07-01; lr:2025-07-13
    if not os.path.exists(pathCdgTxt) or os.path.getsize(pathCdgTxt) == 0:
        raise Exception("Invalid input: failed to read '{}'; file does not exist or is empty".format(pathCdgTxt))
    try:
        data = np.loadtxt(pathCdgTxt, usecols=range(5), dtype=float, ndmin=2)
        cDs, rDs, xs, ys, zs = [data[:, item] for item in range(5)]
    except Exception:
        raise Exception("Invalid input: failed to read '{}'; file has an invalid format".format(pathCdgTxt))
    if readCodes:
        try:
            codes = np.loadtxt(pathCdgTxt, usecols=5, dtype=str, ndmin=1)
            assert len(set(codes)) == len(codes)
        except Exception:  # WATCH OUT
            codes = None  # IMP*
    else:
        codes = None  # IMP*
    if len(cDs) > 0 and readOnlyGood:
        possG = CR2PossWithinImage_2502(cDs, rDs, nc, nr, margin=margin)
        cDs, rDs, xs, ys, zs = [item[possG] for item in [cDs, rDs, xs, ys, zs]]
        if codes is not None:
            codes = codes[possG]
    return cDs, rDs, xs, ys, zs, codes
def ReadCdhTxt_2504(pathCdhTxt, readOnlyGood=True, nc=np.inf, nr=np.inf, margin=0):  # 1900-01-01; lm:2025-05-28; lr:2025-07-02
    if not os.path.exists(pathCdhTxt) or os.path.getsize(pathCdhTxt) == 0:  # IMP*
        cDhs, rDhs = [np.asarray([]) for _ in range(2)]
        return cDhs, rDhs
    try:
        data = np.loadtxt(pathCdhTxt, usecols=range(2), dtype=float, ndmin=2)
        cDhs, rDhs = [data[:, item] for item in range(2)]
    except Exception:
        raise Exception("Invalid input: failed to read '{}'".format(pathCdhTxt))
    if len(cDhs) > 0 and readOnlyGood:
        possG = CR2PossWithinImage_2502(cDhs, rDhs, nc, nr, margin=margin)
        cDhs, rDhs = [item[possG] for item in [cDhs, rDhs]]
    return cDhs, rDhs
def ReadPdfTxt_2410(pathFl):  # 2000-01-01; lm:2025-04-07; lr:2025-07-02
    data = np.loadtxt(pathFl, usecols=0, dtype=float)
    xUL, yUL, angleInDegrees, xyLengthInC, xyLengthInR, ppm = data[:6]  # IMP*
    angle = angleInDegrees * np.pi / 180  # IMP*
    return xUL, yUL, angle, xyLengthInC, xyLengthInR, ppm
def RecomputeFPS_2502(fpsGoal, fpsAvailable, round=True):  # 1900-01-01; lm:2025-06-06; lr:2025-07-11
    if not (0 < fpsGoal < fpsAvailable):
        fps = fpsAvailable
    else:
        factor = fpsAvailable / fpsGoal  # IMP*; >1
        if round:
            factor = int(np.round(factor))
        else:
            factor = int(factor)
        if True:  # avoidable check for readability
            assert isinstance(factor, int) and factor >= 1
        fps = fpsAvailable / factor  # float, but so that fpsAvailable / fps = factor = integer
    if True:  # avoidable check for readability
        assert 0 < fps <= fpsAvailable
    return fps
def RefinePolyline3D(pl, dl):  # lm:2025-07-02; lr:2025-07-14
    lens0 = np.linalg.norm(np.diff(np.column_stack((pl['xs'], pl['ys'], pl['zs'])), axis=0), axis=1)  # n-1; 2025-07-10
    cumLens0 = np.concatenate([[0], np.cumsum(lens0)])  # n
    cumLensR = np.linspace(0, cumLens0[-1], int(max(cumLens0) / dl + 1))
    plR = {key: np.interp(cumLensR, cumLens0, pl[key]) for key in ['xs', 'ys', 'zs']}
    return plR
def RndSclVar_2506(varKeys, dBasSD, dGvnVar, xs=None, ys=None, zs=None, xc=None, yc=None, zc=None, k1asa2=None, x0=None, y0=None):  # 2010-01-01; lm:2025-06-23; lr:2025-07-10
    dAux = {'xc': xc, 'yc': yc, 'zc': zc, 'x0': x0, 'y0': y0}
    dVar = {}
    for key in varKeys:
        if key in dGvnVar.keys():
            print(key, dGvnVar[key])
            val0 = dGvnVar[key]
            rng0 = 0.
        elif key in dAux and dAux[key] is not None:
            val0 = dAux[key]
            rng0 = dBasSD['dRefRng'][key]
        elif key in ['xc', 'x0'] and xs is not None:  # avoiding GCPs outliers
            val0 = np.median(xs)
            rng0 = min(10 * (np.percentile(xs, 75) - np.percentile(xs, 25)), 1000)
        elif key in ['yc', 'y0'] and ys is not None:  # avoiding GCPs outliers
            val0 = np.median(ys)
            rng0 = min(10 * (np.percentile(ys, 75) - np.percentile(ys, 25)), 1000)
        elif key in ['zc'] and zs is not None:
            val0 = max(np.median(zs), 1.e-12) + 40  # WATCH OUT: epsilon
            rng0 = 150
        else:
            val0 = dBasSD['dRefVal'][key]
            rng0 = dBasSD['dRefRng'][key]
        dVar[key] = val0 + np.random.uniform(-rng0, +rng0)
        if key == 'zc':
            dVar[key] = max(dVar[key], 1)  # WATCH OUT: epsilon
    if dBasSD['rangeC'] == 'close' and all(item in dVar for item in ['xc', 'yc', 'zc', 'sca']):
        if all(item is not None for item in [xs, ys, zs]):
            efx, efy, efz = np.median(xs) - dVar['xc'], np.median(ys) - dVar['yc'], np.median(zs) - dVar['zc']
            efm = np.hypot(np.hypot(efx, efy), efz)
            efx, efy, efz = [item / efm for item in [efx, efy, efz]]
            sta, cta = np.hypot(efx, efy), -efz
            dVar['ta'] = np.angle(cta + 1j * sta) + 0.1 * dBasSD['dRefRng']['ta']  # WATCH OUT: epsilon
            if np.isclose(sta, 0, atol=1.e-6):
                dVar['ph'] = np.random.uniform(-np.pi, +np.pi)
            else:
                dVar['ph'] = np.angle(efy + 1j * efx) + 0.1 * dBasSD['dRefRng']['ph']  # IMP*: from the definition of ef = (+sph * sta, +cph * sta, -cta)
        if k1asa2 is not None:
            dVar['k1a'] = k1asa2 / dVar['sca'] ** 2 + 0.1 * dBasSD['dRefRng']['k1a']
    var = Dictionary2Array(varKeys, dVar)
    sclVar = SclVar_2410(var, varKeys, dBasSD['dScl'], 'scale')
    return sclVar
def SclFreUVar2FTM_2502(sclFreUVar, theArgs):  # 2010-01-01; lm:2025-05-19; lr:2025-07-14
    dBasSD, dsGH1s, dGvnVar, freDVars, freDVarKeys, freUVarKeys = [theArgs[item] for item in ['dBasSD', 'dsGH1s', 'dGvnVar', 'freDVars', 'freDVarKeys', 'freUVarKeys']]
    nc, nr, rangeC, selVarKeys, zr, z0, dScl = [dBasSD[item] for item in ['nc', 'nr', 'rangeC', 'selVarKeys', 'zr', 'z0', 'dScl']]
    freUVar = SclVar_2410(sclFreUVar, freUVarKeys, dScl, 'unscale')
    dFreUVar = Array2Dictionary(freUVarKeys, freUVar)
    errorTs = np.zeros(len(dsGH1s))
    for pos in range(len(dsGH1s)):
        assert dsGH1s[pos]['nc'] == nc and dsGH1s[pos]['nr'] == nr
        allVar = SubVar2AllVar_2410(freDVars[pos], freDVarKeys, dGvnVar | dFreUVar, selVarKeys, rangeC, nc=nc, nr=nr, z0=z0)
        dMCS = AllVar2DMCS_2410(allVar, nc, nr, rangeC, incHor=True, zr=zr)
        errorTs[pos] = ErrorT_2410(dsGH1s[pos], dMCS, dMCS['dHor'])
    errorT = np.nanmean(errorTs)  # WATCH OUT: np.nanmean; IMP*
    return errorT
def SclSubVar2DMCS_2502(dBasSD, sclSubVar, subVarKeys, dRemVar, incHor=True):  # 2010-01-01; lm:2025-04-28; lr:2025-07-13
    assert set(subVarKeys).isdisjoint(dRemVar) and set(subVarKeys).union(dRemVar) == set(dBasSD['selVarKeys'])
    subVar = SclVar_2410(sclSubVar, subVarKeys, dBasSD['dScl'], 'unscale')
    allVar = SubVar2AllVar_2410(subVar, subVarKeys, dRemVar, dBasSD['selVarKeys'], dBasSD['rangeC'], nc=dBasSD['nc'], nr=dBasSD['nr'], z0=dBasSD['z0'])
    dMCS = AllVar2DMCS_2410(allVar, dBasSD['nc'], dBasSD['nr'], dBasSD['rangeC'], incHor=incHor, zr=dBasSD['zr'])
    return dMCS
def SclSubVar2ErrorT_2502(dBasSD, dGH1, sclSubVar, subVarKeys, dRemVar):  # 2010-01-01; lm:2025-04-28; lr:2025-06-23
    dMCS = SclSubVar2DMCS_2502(dBasSD, sclSubVar, subVarKeys, dRemVar, incHor=True)
    errorT = ErrorT_2410(dGH1, dMCS, dMCS['dHor'])
    return errorT
def SclSubVar2FTM_2502(sclSubVar, theArgs):  # 2010-01-01; lm:2025-04-28; lr:2025-06-23
    dBasSD, dGH1, subVarKeys, dRemVar = [theArgs[key] for key in ['dBasSD', 'dGH1', 'subVarKeys', 'dRemVar']]
    errorT = SclSubVar2ErrorT_2502(dBasSD, dGH1, sclSubVar, subVarKeys, dRemVar)
    return errorT
def SclVar_2410(var, varKeys, dScl, direction):  # 2000-01-01; lm:2025-05-13; lm:2025-07-14
    scl = Dictionary2Array(varKeys, dScl)
    if not (len(scl) == len(var) == len(varKeys) and min(np.abs(scl)) > 1.e-14):  # WATCH OUT: epsilon
        raise Exception("Invalid input: please check 'var', 'varKeys' and 'dScal'")
    if direction == 'scale':
        var = var / scl
    elif direction == 'unscale':
        var = var * scl
    else:
        raise Exception("Invalid input: 'direction' ('{}') must be 'scale' or 'unscale'".format(direction))
    return var
def SelVar2AllVar_2410(selVar, selVarKeys, rangeC, nc=None, nr=None, z0=None):  # 2010-01-01; lm:2025-04-28; lr:2025-06-23
    minVarKeys, allVarKeys = MinVarKeys(rangeC), AllVarKeys(rangeC)
    if not (set(minVarKeys) <= set(selVarKeys) <= set(allVarKeys)):
        raise Exception("Invalid input: invalid 'selVarKeys' value ('{}')".format(selVarKeys))
    dAllVar = Array2Dictionary(selVarKeys, selVar)  # initialize dAllVar as dSelVar
    for key in [item for item in allVarKeys if item not in selVarKeys]:  # WATCH OUT: only missing
        if key in ['k1a', 'k2a', 'p1a', 'p2a']:
            dAllVar[key] = 0.
        elif key == 'sra':
            dAllVar[key] = dAllVar['sca']
        elif key == 'oc':
            if nc is None:
                raise Exception("Invalid input: 'nc' must not be None")
            dAllVar[key] = N2K_2410(nc)
        elif key == 'or':
            if nr is None:
                raise Exception("Invalid input: 'nr' must not be None")
            dAllVar[key] = N2K_2410(nr)
        elif key == 'z0':
            if z0 is None:
                raise Exception("Invalid input: 'z0' must not be None")
            dAllVar[key] = z0
        elif key == 'sr':
            dAllVar[key] = dAllVar['sc']
    allVar = Dictionary2Array(allVarKeys, dAllVar)
    return allVar
def SelectPixelsInAGrid_2506(cs, rs, es, nc, nr, nOfBands=None, nOfCBands=None, nOfRBands=None):  # lm:2025-06-20; lr:2025-07-07
    if len(cs) == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=int), np.asarray([], dtype=int)
    bandCs, bandRs, bandGs = Pixels2BandsInGrid_2506(cs, rs, nc, nr, nOfBands=nOfBands, nOfCBands=nOfCBands, nOfRBands=nOfRBands)
    possS = []
    for bandGU in np.unique(bandGs):
        possTMP = np.where(bandGs == bandGU)[0]
        if len(possTMP) == 1:
            posS = possTMP[0]
        else:
            posS = possTMP[np.argmin(es[possTMP])]
        possS.append(posS)
    possS = np.asarray(possS, dtype=int)    
    bandCsS, bandRsS = bandCs[possS], bandRs[possS]
    return possS, bandCsS, bandRsS
def SubVar2AllVar_2410(subVar, subVarKeys, dRemVar, selVarKeys, rangeC, nc=None, nr=None, z0=None):  # 2010-01-01; lm:2025-04-28; lr:2025-06-23
    if not (set(subVarKeys).isdisjoint(dRemVar) and set(subVarKeys).union(dRemVar) == set(selVarKeys)):  # WATCH OUT: "=="
        raise Exception("Invalid input: please check 'subVarkeys', 'dRemVar' and 'selVarKeys'")
    dSelVar = Array2Dictionary(subVarKeys, subVar) | dRemVar
    selVar = Dictionary2Array(selVarKeys, dSelVar)
    allVar = SelVar2AllVar_2410(selVar, selVarKeys, rangeC, nc=nc, nr=nr, z0=z0)
    return allVar
def UDaVDa2UUaVUaAux_2410(uDas, vDas, uUas, vUas, dDtrVar, rangeC):  # 1900-01-01; lm:2025-05-01; lr:2025-07-01
    uDasN, vDasN = UUaVUa2UDaVDa_2410(uUas, vUas, dDtrVar, rangeC)[:2]  # WATCH OUT: all positions; distorted from the current undistorted
    uerrors, verrors = uDas - uDasN, vDas - vDasN  # IMP*: direction
    errors = np.hypot(uerrors, verrors)
    aux1s = uUas ** 2 + vUas ** 2
    aux1suUa = 2 * uUas
    aux1svUa = 2 * vUas
    aux2s = 1 + dDtrVar['k1a'] * aux1s + dDtrVar['k2a'] * aux1s ** 2
    aux2suUa = dDtrVar['k1a'] * aux1suUa + dDtrVar['k2a'] * 2 * aux1s * aux1suUa
    aux2svUa = dDtrVar['k1a'] * aux1svUa + dDtrVar['k2a'] * 2 * aux1s * aux1svUa
    aux3suUa = 2 * vUas
    aux3svUa = 2 * uUas
    aux4suUa = aux1suUa + 4 * uUas
    aux4svUa = aux1svUa
    aux5suUa = aux1suUa
    aux5svUa = aux1svUa + 4 * vUas
    JuUauUas = aux2s + uUas * aux2suUa + dDtrVar['p2a'] * aux4suUa + dDtrVar['p1a'] * aux3suUa
    JuUavUas = uUas * aux2svUa + dDtrVar['p2a'] * aux4svUa + dDtrVar['p1a'] * aux3svUa
    JvUauUas = vUas * aux2suUa + dDtrVar['p1a'] * aux5suUa + dDtrVar['p2a'] * aux3suUa
    JvUavUas = aux2s + vUas * aux2svUa + dDtrVar['p1a'] * aux5svUa + dDtrVar['p2a'] * aux3svUa
    dens = JuUauUas * JvUavUas - JuUavUas * JvUauUas
    dens = ClipWithSign(dens, 1.e-14, np.inf)  # WATCH OUT: epsilon
    duUas = (+JvUavUas * uerrors - JuUavUas * verrors) / dens
    dvUas = (-JvUauUas * uerrors + JuUauUas * verrors) / dens
    return duUas, dvUas, errors
def UDaVDa2UUaVUaParabolicDistortion_2410(uDas, vDas, k1a, rtrnPossG=False):  # undistort; Cardano; 1900-01-01; lm:2025-05-01; lr:2025-07-01
    xiDs = k1a * (uDas ** 2 + vDas ** 2)
    xiUs, possG = XiD2XiUCubicEquation_2410(xiDs, rtrnPossG=rtrnPossG)
    uUas, vUas = [item / (1 + xiUs) for item in [uDas, vDas]]
    return uUas, vUas, possG
def UDaVDa2UUaVUa_2410(uDas, vDas, dDtrVar, rangeC, rtrnPossG=False):  # undistort; potentially expensive; 1900-01-01; lm:2025-05-01; lm:2025-07-01
    if rangeC == 'long':
        if rtrnPossG:
            possG = np.arange(len(uDas))
        else:
            possG = np.asarray([], dtype=int)
        return uDas, vDas, possG
    elif rangeC == 'close':
        if len(uDas) == 0 or len(vDas) == 0:
            return np.full(uDas.shape, np.nan), np.full(uDas.shape, np.nan), np.asarray([], dtype=int)  # WATCH OUT
        uUas, vUas, possG = UDaVDa2UUaVUaParabolicDistortion_2410(uDas, vDas, dDtrVar['k1a'], rtrnPossG=rtrnPossG)
        if np.allclose([dDtrVar['k2a'], dDtrVar['p1a'], dDtrVar['p2a']], 0, atol=1.e-9):  # WATCH OUT: epsilon
            return uUas, vUas, possG
        errors, hasConverged, counter = 1.e+6 * np.ones(uDas.shape), False, 0
        while not hasConverged and counter <= 50:
            duUas, dvUas, errorsN = UDaVDa2UUaVUaAux_2410(uDas, vDas, uUas, vUas, dDtrVar, rangeC)
            possB = np.where(np.isnan(errorsN) | (errorsN > 4 * errors))[0]  # WATCH OUT: some points do not converge; epsilon
            if len(possB) == len(uDas):  # not hasConverged
                break
            uUas, vUas, errors, hasConverged, counter = uUas + duUas, vUas + dvUas, errorsN, np.nanmax(errorsN) < 1.e-9, counter + 1  # WATCH OUT: epsilon
            uUas[possB], vUas[possB] = np.nan, np.nan  # IMP*
        if not hasConverged:
            return np.full(uDas.shape, np.nan), np.full(uDas.shape, np.nan), np.asarray([], dtype=int)  # WATCH OUT
        if rtrnPossG:  # WATCH OUT: necessarily different than for parabolic: now it includes something like the solution xiU < -4/3; handles np.nan
            uDasR, vDasR = UUaVUa2UDaVDa_2410(uUas, vUas, dDtrVar, rangeC)[:2]  # WATCH OUT: all positions
            possG = np.where(np.hypot(uDasR - uDas, vDasR - vDas) < 1.e-9)[0]  # WATCH OUT: epsilon; this is also a check
        else:
            possG = np.asarray([], dtype=int)
    else:
        raise Exception("Invalid input: 'rangeC' ('{}') must be 'close' or 'long'".format(rangeC))
    return uUas, vUas, possG
def UUaVUa2UDaVDaParabolicDistortion_2410(uUas, vUas, k1a, rtrnPossG=False):  # distort; 1900-01-01; lm:2025-05-01; lm:2025-07-01
    xiUs = k1a * (uUas ** 2 + vUas ** 2)
    uDas, vDas = [item * (1 + xiUs) for item in [uUas, vUas]]
    xiDs, possG = XiU2XiDCubicEquation_2410(xiUs, rtrnPossG=rtrnPossG)
    if False:  # avoidable check for readability
        assert np.allclose(xiDs, k1a * (uDas ** 2 + vDas ** 2))
    return uDas, vDas, possG
def UUaVUa2UDaVDa_2410(uUas, vUas, dDtrVar, rangeC, rtrnPossG=False):  # distort; 1900-01-01; lm:2025-05-01; lr:2025-07-01
    if rangeC == 'long':
        if rtrnPossG:
            possG = np.arange(len(uUas))
        else:
            possG = np.asarray([], dtype=int)
        return uUas, vUas, possG
    elif rangeC == 'close':
        if np.allclose([dDtrVar['k2a'], dDtrVar['p1a'], dDtrVar['p2a']], 0, atol=1.e-9):  # WATCH OUT: epsilon
            uDas, vDas, possG = UUaVUa2UDaVDaParabolicDistortion_2410(uUas, vUas, dDtrVar['k1a'], rtrnPossG=rtrnPossG)
            return uDas, vDas, possG
        aux1s = uUas ** 2 + vUas ** 2  # = dUas**2 = d_{U*}**2
        aux2s = 1 + dDtrVar['k1a'] * aux1s + dDtrVar['k2a'] * aux1s ** 2
        aux3s = 2 * uUas * vUas
        aux4s = aux1s + 2 * uUas ** 2
        aux5s = aux1s + 2 * vUas ** 2
        uDas = uUas * aux2s + dDtrVar['p2a'] * aux4s + dDtrVar['p1a'] * aux3s
        vDas = vUas * aux2s + dDtrVar['p1a'] * aux5s + dDtrVar['p2a'] * aux3s
        if rtrnPossG:  # WATCH OUT: necessarily different than for parabolic: now it includes something like the solution xiU < -4/3
            uUasR, vUasR = UDaVDa2UUaVUa_2410(uDas, vDas, dDtrVar, rangeC)[:2]  # WATCH OUT: all positions
            possG = np.where(np.hypot(uUasR - uUas, vUasR - vUas) < 1.e-6)[0]  # WATCH OUT: epsilon
        else:
            possG = np.asarray([], dtype=int)
    else:
        raise Exception("Invalid input: 'rangeC' ('{}') must be 'close' or 'long'".format(rangeC))
    return uDas, vDas, possG
def UaVa2CR_2410(uas, vas, dCaSVar, rangeC):  # 1900-01-01; lm:2025-04-30; lr:2025-06-30
    if rangeC == 'close':
        sca, sra = [ClipWithSign(dCaSVar[item], 1.e-14, np.inf) for item in ['sca', 'sra']]  # WATCH OUT: epsilon
        cs = uas / sca + dCaSVar['oc']
        rs = vas / sra + dCaSVar['or']
    elif rangeC == 'long':
        sc, sr = [ClipWithSign(dCaSVar[item], 1.e-14, np.inf) for item in ['sc', 'sr']]  # WATCH OUT: epsilon
        cs = uas / sc + dCaSVar['oc']  # uas are actually us in this case
        rs = vas / sr + dCaSVar['or']  # vas are actually vs in this case
    else:
        raise Exception("Invalid input: 'rangeC' ('{}') must be 'close' or 'long'".format(rangeC))
    return cs, rs
def UnitVectors_2502(dAngVar):  # 1900-01-01; lm:2025-04-30; lr:2025-06-23
    sph, cph = np.sin(dAngVar['ph']), np.cos(dAngVar['ph'])
    ssg, csg = np.sin(dAngVar['sg']), np.cos(dAngVar['sg'])
    sta, cta = np.sin(dAngVar['ta']), np.cos(dAngVar['ta'])
    eux = +csg * cph - ssg * sph * cta
    euy = -csg * sph - ssg * cph * cta
    euz = -ssg * sta
    eu = np.asarray([eux, euy, euz])
    evx = -ssg * cph - csg * sph * cta
    evy = +ssg * sph - csg * cph * cta
    evz = -csg * sta
    ev = np.asarray([evx, evy, evz])
    efx = +sph * sta
    efy = +cph * sta
    efz = -cta
    ef = np.asarray([efx, efy, efz])
    if False:  # avoidable check for readability
        R = np.asarray([eu, ev, ef])
        assert np.allclose(np.dot(R, np.transpose(R)), np.eye(3)) and np.isclose(np.linalg.det(R), 1)
    return eu, ev, ef
def UpdateFreUVar(dBasSD, dsGH1s, dGvnVar, freDVars, freDVarKeys, freUVar, freUVarKeys, nOfSeeds=20):  # 2010-01-01; lm:2025-05-19; lr:2025-06-23
    theArgs = {'dBasSD': dBasSD, 'dsGH1s': dsGH1s, 'dGvnVar': dGvnVar, 'freDVars': freDVars, 'freDVarKeys': freDVarKeys, 'freUVarKeys': freUVarKeys}
    sclFreUVar, errorT = SclVar_2410(freUVar, freUVarKeys, dBasSD['dScl'], 'scale'), np.inf
    for _ in range(nOfSeeds):
        sclFreUVarH = sclFreUVar * np.random.uniform(0.8, 1.2, len(sclFreUVar))  # WATCH OUT: a capon
        sclFreUVarH = optimize.minimize(SclFreUVar2FTM_2502, sclFreUVarH, args=theArgs).x
        errorTH = SclFreUVar2FTM_2502(sclFreUVarH, theArgs)
        if errorTH < errorT * (1 - 1.e-6):  # WATCH OUT: epsilon
            sclFreUVar, errorT = sclFreUVarH, errorTH
    freUVar = SclVar_2410(sclFreUVar, freUVarKeys, dBasSD['dScl'], 'unscale')
    return freUVar, errorT
def UpdateOCOR(oc_=None, or_=None, nc=None, nr=None):  # lm:2025-06-20; lr:2025-07-09
    if oc_ is None:
        if nc is None:
            raise ValueError("! Invalid input: either 'nc' or 'oc_' must be provided (not both None)")
        oc_ = N2K_2410(nc)
    if or_ is None:
        if nr is None:
            raise ValueError("! Invalid input: either 'nr' or 'or_' must be provided (not both None)")
        or_ = N2K_2410(nr)
    return oc_, or_
def Vid2VidModified_2504(pathVid0, pathVid1, fps=0., round=True, scl=1., t0InSeconds=0., t1InSeconds=np.inf, overwrite=False):  # 2010-01-01; lm:2025-06-23; lr:2025-07-11
    if pathVid1 == pathVid0:
        raise Exception("Invalid input: 'pathVid0' and 'pathVid1' must be different")
    pathFld0, fnVid0 = os.path.split(pathVid0)
    pathFld1, fnVid1 = os.path.split(pathVid1)
    os.makedirs(pathFld1, exist_ok=True)
    if os.path.exists(pathVid1):
        if overwrite:
            os.remove(pathVid1)
        else:
            return None
    fnVidTMP = '000_{:}{:}'.format(''.join(random.choices(string.ascii_letters, k=20)), os.path.splitext(fnVid1)[1])
    pathVidTMP = os.path.join(pathFld0, fnVidTMP)  # IMP*; we always run in pathFld0
    if os.path.exists(pathVidTMP):  # miracle
        os.remove(pathVidTMP)
    (nc0, nr0), fps0, nOfFrames0 = PathVid2NcNr(pathVid0), PathVid2FPS(pathVid0), PathVid2NOfFrames(pathVid0)
    t0InSeconds0, t1InSeconds0 = 0, (nOfFrames0 - 1) / fps0  # IMP*
    fps1 = RecomputeFPS_2502(fps, fps0, round=round)
    if np.isclose(fps0, fps1):
        shutil.copy2(pathVid0, pathVidTMP)  # IMP*
    else:
        cmd = ['ffmpeg', '-i', fnVid0, '-filter:v', 'fps=fps={:.8f}'.format(fps1), fnVidTMP]
        subprocess.run(cmd, cwd=pathFld0, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    if not os.path.exists(pathVidTMP):
        raise Exception("Unexpected condition: unable to change fps of '{}'".format(pathVid0))
    t0InSeconds1, t1InSeconds1 = max(t0InSeconds, t0InSeconds0), min(t1InSeconds, t1InSeconds0)
    if not (np.isclose(t0InSeconds0, t0InSeconds1) and np.isclose(t1InSeconds0, t1InSeconds1)):
        t0 = '{:02}:{:02}:{:02}'.format(*map(int, [t0InSeconds1 // 3600, (t0InSeconds1 % 3600) // 60, t0InSeconds1 % 60]))
        t1 = '{:02}:{:02}:{:02}'.format(*map(int, [t1InSeconds1 // 3600, (t1InSeconds1 % 3600) // 60, t1InSeconds1 % 60]))
        fnAux = '{:}_aux{:}'.format(os.path.splitext(fnVidTMP)[0], os.path.splitext(fnVidTMP)[1])
        cmd = ['ffmpeg', '-ss', t0, '-i', fnVidTMP, '-to', t1, fnAux]
        subprocess.run(cmd, cwd=pathFld0, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        shutil.move(os.path.join(pathFld0, fnAux), pathVidTMP)
    nc1, nr1 = int(np.round(scl * nc0)), int(np.round(scl * nr0))
    if not (np.isclose(nc0, nc1) and np.isclose(nr0, nr1)):
        fnAux = '{:}_aux{:}'.format(os.path.splitext(fnVidTMP)[0], os.path.splitext(fnVidTMP)[1])
        cmd = ['ffmpeg', '-i', fnVidTMP, '-vf', f'scale={nc1}:{nr1}', '-c:v', 'libx264', '-crf', '23', '-preset', 'veryfast', fnAux]
        subprocess.run(cmd, cwd=pathFld0, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        shutil.move(os.path.join(pathFld0, fnAux), pathVidTMP)
    shutil.move(pathVidTMP, pathVid1)
    return None
def WriteCalTxt_2410(pathCalTxt, allVar, nc, nr, errorT, rangeC):  # lm:1900-01-01; lm:2025-05-01; lr:2025-07-11
    if not all(np.isclose(item, np.round(item), atol=1.e-3) for item in [nc, nr]):
        raise Exception("Invalid input: 'nc' and 'nr' must be integers")
    allVarKeys = AllVarKeys(rangeC)
    if not len(allVarKeys) == len(allVar):
        raise Exception("Invalid input: 'allVar' must be a NumPy array of length {}".format(len(allVarKeys)))
    os.makedirs(os.path.dirname(pathCalTxt), exist_ok=True)
    with open(pathCalTxt, 'w') as fileout:
        for posVar, var in enumerate(allVar):
            key = allVarKeys[posVar]
            if key in ['ph', 'sg', 'ta']:
                var = np.angle(np.exp(1j * var))  # IMP*: var in (-np.pi, np.pi]
            fileout.write('{:21.9f} \t {}\n'.format(var, key))  # WATCH OUT: formatting
        fileout.write('{:21.0f} \t nc\n'.format(np.round(nc)))  # WATCH OUT: formatting
        fileout.write('{:21.0f} \t nr\n'.format(np.round(nr)))  # WATCH OUT: formatting
        fileout.write('{:21.9f} \t error\n'.format(errorT))  # WATCH OUT: formatting
    return None
def WriteCdgTxt_2502(pathCdgTxt, cs, rs, xs, ys, zs, codes=None):  # 1900-01-01; lm:2025-05-28; lr:2025-07-14
    os.makedirs(os.path.dirname(pathCdgTxt), exist_ok=True)
    with open(pathCdgTxt, 'w') as fileout:
        for pos in range(len(cs)):
            fileout.write('{:18.6f} {:18.6f} {:15.3f} {:15.3f} {:15.3f}'.format(cs[pos], rs[pos], xs[pos], ys[pos], zs[pos]))  # WATCH OUT: formatting
            if codes is None or codes[pos] == 'c,':  # WATCH OUT: in case we have wrongly read 'c,' as code
                fileout.write(' \t c, r, x, y and z\n')  # WATCH OUT: formatting
            else:
                fileout.write(' {:>25} \t c, r, x, y, z and code\n'.format(codes[pos]))  # WATCH OUT: formatting
    return None
def WriteCdhTxt_2504(pathCdhTxt, chs, rhs):  # 1900-01-01; lm:2025-05-28; lr:2025-07-14
    os.makedirs(os.path.dirname(pathCdhTxt), exist_ok=True)
    with open(pathCdhTxt, 'w') as fileout:
        for pos in np.argsort(chs):
            fileout.write('{:18.6f} {:18.6f} \t c and r at the horizon\n'.format(chs[pos], rhs[pos]))  # WATCH OUT: formatting
    return None
def XYZ2A0(xs, ys, zs):  # 1900-01-01; lm:2025-05-28; lr:2025-07-07
    poss0, poss1 = Poss0AndPoss1InFind2DTransform_2504(len(xs))
    A0 = np.zeros((2 * len(xs), 8))  # IMP*: initialize with zeroes
    A0[poss0, 0], A0[poss0, 1], A0[poss0, 2], A0[poss0, 3] = xs, ys, zs, np.ones(xs.shape)
    A0[poss1, 4], A0[poss1, 5], A0[poss1, 6], A0[poss1, 7] = xs, ys, zs, np.ones(xs.shape)
    return A0
def XYZ2CDRD_2410(xs, ys, zs, dMCS, rtrnPossG=False, margin=0):  # explicit if not rtrnPossG; 2010-01-01; lm:2025-05-05; lr:2025-06-23
    Px, rangeC, dAllVar, ef, nc, nr = [dMCS[item] for item in ['Px', 'rangeC', 'dAllVar', 'ef', 'nc', 'nr']]
    cUs, rUs, possG = XYZ2CURU_2410(xs, ys, zs, Px, rangeC, rtrnPossG=rtrnPossG, dCamVar=dAllVar, ef=ef, nc=nc, nr=nr)
    cDs, rDs, possGH = CURU2CDRD_2410(cUs, rUs, dAllVar, dAllVar, rangeC, rtrnPossG=rtrnPossG, nc=nc, nr=nr, margin=margin)
    possG = np.intersect1d(possG, possGH, assume_unique=True)
    if rtrnPossG and len(possG) > 0:
        xsG, ysG, zsG, cDsG, rDsG = [item[possG] for item in [xs, ys, zs, cDs, rDs]]
        xsGR, ysGR = CDRDZ2XY_2410(cDsG, rDsG, zsG, dMCS)[:2]  # WATCH OUT: potentially expensive
        possGInPossG = np.where(np.hypot(xsG - xsGR, ysG - ysGR) < 1.e-6)[0]  # WATCH OUT: epsilon; could be 1.e-3 also
        possG = possG[possGInPossG]
    return cDs, rDs, possG
def XYZ2CURU_2410(xs, ys, zs, Px, rangeC, rtrnPossG=False, dCamVar=None, ef=None, nc=None, nr=None):  # 2010-01-01; lm:2025-05-28; lr:2025-07-05
    if rangeC == 'close':
        dens = Px[8] * xs + Px[9] * ys + Px[10] * zs + 1
        dens = ClipWithSign(dens, 1.e-14, np.inf)  # WATCH OUT: epsilon
        cUs = (Px[0] * xs + Px[1] * ys + Px[2] * zs + Px[3]) / dens
        rUs = (Px[4] * xs + Px[5] * ys + Px[6] * zs + Px[7]) / dens
    elif rangeC == 'long':
        cUs = Px[0] * xs + Px[1] * ys + Px[2] * zs + Px[3]
        rUs = Px[4] * xs + Px[5] * ys + Px[6] * zs + Px[7]
    else:
        raise Exception("Invalid input: 'rangeC' ('{}') must be 'close' or 'long'".format(rangeC))
    if rtrnPossG:
        possG = XYZ2PossRightSideOfCamera_2410(xs, ys, zs, rangeC, dCamVar=dCamVar, ef=ef)
        if len(possG) > 0 and nc is not None and nr is not None:
            cUsG, rUsG = [item[possG] for item in [cUs, rUs]]
            possGInPossG = CR2PossWithinImage_2502(cUsG, rUsG, nc, nr, margin=-max(nc, nr), case='')  # WATCH OUT: undistorted, large negative margin to relax
            possG = possG[possGInPossG]
    else:
        possG = np.asarray([], dtype=int)
    return cUs, rUs, possG
def XYZ2PossRightSideOfCamera_2410(xs, ys, zs, rangeC, dCamVar=None, ef=None):  # 2000-01-01; lr:2025-05-28; lr:2025-06-22
    if rangeC == 'close':
        xas, yas, zas = xs - dCamVar['xc'], ys - dCamVar['yc'], zs - dCamVar['zc']
        possRightSideOfCamera = np.where(xas * ef[0] + yas * ef[1] + zas * ef[2] > 0)[0]
    elif rangeC == 'long':
        possRightSideOfCamera = np.arange(len(xs))
    else:
        raise Exception("Invalid input: 'rangeC' ('{}') must be 'close' or 'long'".format(rangeC))
    return possRightSideOfCamera
def XiD2XiUCubicEquation_2410(xiDs, rtrnPossG=False):  # undistort; Cardano; 1900-01-01; lr:2025-05-28; lr:2025-06-23
    p, qs = -1 / 3, -(xiDs + 2 / 27)
    Deltas = (xiDs + 4 / 27) * xiDs  # Deltas <= 0 -> -4/27 <= xiD <= 0 -> several solutions
    possN = np.where(Deltas <= 0)[0]  # possN -> several solutions
    possP = np.asarray([item for item in np.arange(len(xiDs)) if item not in possN], dtype=int)  # possP -> unique solution
    auxsN = (qs[possN] + 1j * np.sqrt(np.abs(Deltas[possN]))) / 2
    auxsP = (qs[possP] + np.sqrt(Deltas[possP])) / 2
    ns = np.zeros(xiDs.shape) + 1j * np.zeros(xiDs.shape)
    ns[possN] = np.abs(auxsN) ** (1 / 3) * np.exp(1j * (np.abs(np.angle(auxsN)) + 2 * np.pi * 1) / 3)  # + 2 * pi * j for j = 0, *1*, 2
    ns[possP] = np.sign(auxsP) * (np.abs(auxsP) ** (1 / 3))
    xiUs = np.real(p / (3 * ns) - ns - 2 / 3)  # WATCH OUT
    if rtrnPossG:
        possG = np.where(xiDs >= -4 / 27)[0]  # works also if len(xiDs) = 0
    else:
        possG = np.asarray([], dtype=int)
    return xiUs, possG
def XiU2XiDCubicEquation_2410(xiUs, rtrnPossG=False):  # distort; 1900-01-01; lr:2025-05-28; lr:2025-07-01
    xiDs = xiUs ** 3 + 2 * xiUs ** 2 + xiUs
    if rtrnPossG:
        possG = np.where(xiUs >= -1 / 3)[0]  # works also if len(xiUs) = 0
    else:
        possG = np.asarray([], dtype=int)
    return xiDs, possG
