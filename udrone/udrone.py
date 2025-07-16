# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~ by Gonzalo Simarro and Daniel Calvete
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
import os
import cv2  # type: ignore
import itertools
import numpy as np  # type: ignore
import shutil
import sys
#
import ulises_udrone as uli  # type: ignore
import warnings
warnings.filterwarnings("ignore")
#
# ~~~~~~ data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
rangeC = 'close'
extsVids, extsImgs = ['mp4', 'avi', 'mov'], ['jpeg', 'jpg', 'png']
dGit = uli.GHLoadDGit()
#
freDVarKeys = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta']
lens2SelVarKeys = {}
lens2SelVarKeys = lens2SelVarKeys | {'parabolic': ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'sca']}
lens2SelVarKeys = lens2SelVarKeys | {'quartic': ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'sca']}
lens2SelVarKeys = lens2SelVarKeys | {'full': ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or']}
#
# ~~~~~~ main ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
def Inform_UDrone(pathFldMain, pos):  # lm:2025-06-19; lr:2025-07-14
    #
    # obtain par and inform
    par = uli.GHLoadPar(pathFldMain)
    uli.GHInform_2506('UDrone', pathFldMain, par, pos, margin=dGit['ind'], sB='', nFill=10)
    #
    return None
#
def Video2Frames(pathFldMain):  # lm:2025-06-27; lr:2025-07-14
    #
    # obtain par and extract video
    par = uli.GHLoadPar(pathFldMain)
    print("{}{} Extracting frames from video at /data directory".format(' '*dGit['ind'], dGit['sB1']))
    pathFldFrames = uli.GHDroneExtractVideoToPathFldFrames(pathFldMain, active=True, extsVids=extsVids, fps=par['frame_rate'], extImg='png', overwrite=par['overwrite_outputs'])  # WATCH OUT: png; stamp = 'millisecond'
    if uli.IsFldModified_2506(pathFldFrames):
        print("\033[F\033[K{}{} Video at /data successfully processed: {} frames extracted {}".format(' '*dGit['ind'], dGit['sB1'], len(os.listdir(pathFldFrames)), dGit['sOK']))
    else:
        print("\033[F\033[K{}{} Video at /data was already processed: {} frames found {}".format(' '*dGit['ind'], dGit['sB1'], len(os.listdir(pathFldFrames)), dGit['sOK']))
    #
    return None
#
def CalibrationOfBasisImages(pathFldMain):  # lm:2025-07-01; lr:2025-07-15
    #
    # obtain par
    par = uli.GHLoadPar(pathFldMain)
    #
    # obtain fnsImgs and fnsCalTxt0s and skip if calibrations already done
    fnsImgs = sorted([item for item in os.listdir(os.path.join(pathFldMain, 'data', 'basis')) if os.path.splitext(item)[1][1:].lower() in extsImgs])
    if os.path.exists(os.path.join(pathFldMain, 'scratch', 'numerics', 'calibration_basis')):
        fnsCalTxt0s = sorted([item for item in os.listdir(os.path.join(pathFldMain, 'scratch', 'numerics', 'calibration_basis')) if item.endswith('cal0.txt')])
    else:
        fnsCalTxt0s = []
    if set([os.path.splitext(item)[0] for item in fnsImgs]) == set([item[:-8] for item in fnsCalTxt0s]) and not par['overwrite_outputs']:
        errorTs = [uli.ReadCalTxt_2410(os.path.join(pathFldMain, 'scratch', 'numerics', 'calibration_basis', item), rangeC)[-1] for item in fnsCalTxt0s]
        print("{}{} Basis already calibrated: {} calibrations found with errors ranging from {:.2f} to {:.2f} pixels {}".format(' '*dGit['ind'], dGit['sB1'], len(fnsCalTxt0s), min(errorTs), max(errorTs), dGit['sOK']))  # WATCH OUT: formatting
        return None
    #
    # obtain selVarKeys
    selVarKeys = uli.GHLens2SelVarKeysOrFail(par['camera_lens_model'], lens2SelVarKeys)
    #
    # obtain calibration for each fnImg
    for fnImg in fnsImgs:
        # obtain fnImgWE and inform
        fnImgWE = os.path.splitext(fnImg)[0]
        print("{}{} Calibrating image {}".format(' '*dGit['ind'], dGit['sB1'], fnImgWE))
        # load nc, nr, dBasSI and dBasSD
        if 'nc' not in locals() or 'nr' not in locals():
            nc, nr = uli.PathVid2NcNr(os.path.join(pathFldMain, 'data', 'basis', fnImg))
            dBasSD = uli.LoadDBasSD_2506(selVarKeys, nc, nr, rangeC, zr=par['z_sea_level'])
        else:
            assert uli.PathVid2NcNr(os.path.join(pathFldMain, 'data', 'basis', fnImg)) == (nc, nr)
        # load GCPs
        pathCdgTxt = os.path.join(pathFldMain, 'data', 'basis', '{}cdg.txt'.format(fnImgWE))
        cDs, rDs, xs, ys, zs, codes = uli.ReadCdgTxt_2502(pathCdgTxt, readCodes=True, readOnlyGood=True, nc=nc, nr=nr)
        # load horizon points
        pathCdhTxt = os.path.join(pathFldMain, 'data', 'basis', '{}cdh.txt'.format(fnImgWE))
        cDhs, rDhs = uli.ReadCdhTxt_2504(pathCdhTxt, readOnlyGood=True, nc=nc, nr=nr)  # empty ndarrays if pathCdhTxt does not exist
        # load dsMCSs if available
        dsMCSs = []
        if os.path.exists(os.path.join(pathFldMain, 'scratch', 'numerics', 'calibration_basis')):
            for pathCalTxt in [item.path for item in os.scandir(os.path.join(pathFldMain, 'scratch', 'numerics', 'calibration_basis')) if any(item.name.endswith(ending) for ending in ['cal0.txt', 'cal.txt'])]:
                dMCS = uli.LoadDMCSFromCalTxt_2502(pathCalTxt, rangeC, incHor=False)
                dsMCSs.append(dMCS)
        # obtain calibration
        dGH1 = {'cDs': cDs, 'rDs': rDs, 'xs': xs, 'ys': ys, 'zs': zs, 'cDhs': cDhs, 'rDhs': rDhs, 'nc': nc, 'nr': nr}
        dMCS, errorT = uli.ManualCalibration_2502(dBasSD, dGH1, {}, dsMCSs)  # WATCH OUT: dGvnVar = {}
        if dMCS is None or errorT > par['max_reprojection_error_px']:
            print("\033[F\033[K{}{} Image {} could not be calibrated: error = {:.2f} pixels {}".format(' '*dGit['ind'], dGit['sB1'], fnImgWE, errorT, dGit['sKO']))
            if dMCS is not None:
                uli.InformErrorsG_2506(xs, ys, zs, cDs, rDs, dMCS, par['max_reprojection_error_px'], codes=codes, margin=2*dGit['ind'], sB=dGit['sB2'], sWO=dGit['sWO'])
            continue
        # inform and write
        print("\033[F\033[K{}{} Image {} successfully calibrated: error = {:.2f} pixels {}".format(' '*dGit['ind'], dGit['sB1'], fnImgWE, errorT, dGit['sOK']))  # WATCH OUT: formatting
        uli.InformErrorsG_2506(xs, ys, zs, cDs, rDs, dMCS, par['max_reprojection_error_px'], codes=codes, margin=2*dGit['ind'], sB=dGit['sB2'], sWO=dGit['sWO'])
        # write pathCal0Txt
        pathCal0Txt = os.path.join(pathFldMain, 'scratch', 'numerics', 'calibration_basis', '{}cal0.txt'.format(fnImgWE))
        uli.WriteCalTxt_2410(pathCal0Txt, dMCS['allVar'], nc, nr, errorT, rangeC)
        # write pathScrJpg
        if par['generate_scratch_plots']:
            pathScrJpg = os.path.join(pathFldMain, 'scratch', 'plots', 'calibration_basis', '{}cal0.jpg'.format(fnImgWE))
            img = cv2.imread(os.path.join(pathFldMain, 'data', 'basis', fnImg))
            uli.PlotCalibration_2504(img, dMCS, cDs, rDs, xs, ys, zs, cDhs, rDhs, pathScrJpg)
    #
    # check
    if not all(os.path.exists(os.path.join(pathFldMain, 'scratch', 'numerics', 'calibration_basis', '{}cal0.txt'.format(os.path.splitext(item)[0]))) for item in fnsImgs):
        print("! Some calibrations failed: rerun, check GCPs, or try another lens model {}".format(dGit['sKO']))
        sys.exit()
    #
    return None
#
def CalibrationOfBasisImagesConstantIntrinsic(pathFldMain):  # lm:2025-07-02; lr:2025-07-15
    #
    # obtain par
    par = uli.GHLoadPar(pathFldMain)
    #
    # obtain fnsImgs and fnsCalTxts and skip if calibrations already done
    fnsImgs = sorted([item for item in os.listdir(os.path.join(pathFldMain, 'data', 'basis')) if os.path.splitext(item)[1][1:].lower() in extsImgs])
    if os.path.exists(os.path.join(pathFldMain, 'scratch', 'numerics', 'calibration_basis')):
        fnsCalTxts = sorted([item for item in os.listdir(os.path.join(pathFldMain, 'scratch', 'numerics', 'calibration_basis')) if item.endswith('cal.txt')])
    else:
        fnsCalTxts = []
    if set([os.path.splitext(item)[0] for item in fnsImgs]) == set([item[:-7] for item in fnsCalTxts]) and not par['overwrite_outputs']:
        errorTs = [uli.ReadCalTxt_2410(os.path.join(pathFldMain, 'scratch', 'numerics', 'calibration_basis', item), rangeC)[-1] for item in fnsCalTxts]
        print("{}{} Basis already calibrated: {} calibrations found with errors ranging from {:.2f} to {:.2f} pixels {}".format(' '*dGit['ind'], dGit['sB1'], len(fnsCalTxts), min(errorTs), max(errorTs), dGit['sOK']))  # WATCH OUT: formatting
        return None
    #
    # obtain selVarKeys and freUVarKeys
    selVarKeys = uli.GHLens2SelVarKeysOrFail(par['camera_lens_model'], lens2SelVarKeys)
    freUVarKeys = [item for item in selVarKeys if item not in freDVarKeys]  # IMP*: (U)nique, the same for all frames
    #
    # load dsMCSs, codess, dsGH1s and dBasSD
    dsMCSs, codess, dsGH1s = [[] for _ in range(3)]  # IMP*: lists
    for posFnImg, fnImg in enumerate(fnsImgs):
        # obtain fnImgWE
        fnImgWE = os.path.splitext(fnImg)[0]
        # load nc, nr and dBasSD and update dsMCSs
        pathCal0Txt = os.path.join(pathFldMain, 'scratch', 'numerics', 'calibration_basis', '{}cal0.txt'.format(fnImgWE))  # IMP*: exists
        dMCS = uli.LoadDMCSFromCalTxt_2502(pathCal0Txt, rangeC, incHor=True, zr=par['z_sea_level'])
        if 'nc' not in locals() or 'nr' not in locals():
            nc, nr = dMCS['nc'], dMCS['nr']
            dBasSD = uli.LoadDBasSD_2506(selVarKeys, nc, nr, rangeC, zr=par['z_sea_level'])
        else:
            assert dMCS['nc'] == nc and dMCS['nr'] == nr and dMCS['rangeC'] == rangeC
        dsMCSs.append(dMCS)
        # load GCPs and update codess
        pathCdgTxt = os.path.join(pathFldMain, 'data', 'basis', '{}cdg.txt'.format(fnImgWE))
        cDs, rDs, xs, ys, zs, codes = uli.ReadCdgTxt_2502(pathCdgTxt, readCodes=True, readOnlyGood=True, nc=nc, nr=nr)
        codess.append(codes)
        # load horizon points
        pathCdhTxt = os.path.join(pathFldMain, 'data', 'basis', '{}cdh.txt'.format(fnImgWE))
        cDhs, rDhs = uli.ReadCdhTxt_2504(pathCdhTxt, readOnlyGood=True, nc=nc, nr=nr)  # empty ndarrays if pathCdhTxt does not exist
        # update dsGH1s
        dGH1 = {'cDs': cDs, 'rDs': rDs, 'xs': xs, 'ys': ys, 'zs': zs, 'cDhs': cDhs, 'rDhs': rDhs, 'nc': nc, 'nr': nr}
        dsGH1s.append(dGH1)
    #
    # obtain forced calibrations
    dsMCSs, errorTs = uli.ManualCalibrationOfSeveralImages_2502(dBasSD, dsGH1s, {}, freDVarKeys, freUVarKeys, dsMCSs, nOfSeeds=10)  # WATCH OUT: dGvnVar = {}; output lists
    assert np.allclose(errorTs, np.asarray([uli.ErrorT_2410(dsGH1s[pos], dsMCSs[pos], dsMCSs[pos]['dHor']) for pos in range(len(fnsImgs))]))  # avoidable
    for posFnImg, fnImg in enumerate(fnsImgs):
        fnImgWE = os.path.splitext(fnImg)[0]
        if errorTs[posFnImg] <= par['max_reprojection_error_px']:
            print("{}{} Image {} successfully calibrated: error = {:.2f} pixels {}".format(' '*dGit['ind'], dGit['sB1'], fnImgWE, errorTs[posFnImg], dGit['sOK']))
        else:
            print("{}{} Image {} could not be calibrated: error = {:.2f} pixels {}".format(' '*dGit['ind'], dGit['sB1'], fnImgWE, errorTs[posFnImg], dGit['sKO']))
        xsH, ysH, zsH, cDsH, rDsH = [dsGH1s[posFnImg][item] for item in ['xs', 'ys', 'zs', 'cDs', 'rDs']]
        uli.InformErrorsG_2506(xsH, ysH, zsH, cDsH, rDsH, dsMCSs[posFnImg], par['max_reprojection_error_px'], codes=codess[posFnImg], margin=2*dGit['ind'], sB=dGit['sB2'], sWO=dGit['sWO'])
    if any(errorTs > par['max_reprojection_error_px']):
        print("! Some calibrations failed: rerun, check GCPs, or try another lens model {}".format(dGit['sKO']))
        sys.exit()
    #
    # write forced calibrations
    for posFnImg, fnImg in enumerate(fnsImgs):
        # obtain fnImgWE
        fnImgWE = os.path.splitext(fnImg)[0]
        # write pathCalTxt
        pathCalTxt = os.path.join(pathFldMain, 'scratch', 'numerics', 'calibration_basis', '{}cal.txt'.format(fnImgWE))
        uli.WriteCalTxt_2410(pathCalTxt, dsMCSs[posFnImg]['allVar'], nc, nr, errorTs[posFnImg], rangeC)
        # write pathScrJpg
        if par['generate_scratch_plots']:
            pathScrJpg = os.path.join(pathFldMain, 'scratch', 'plots', 'calibration_basis', '{}cal.jpg'.format(fnImgWE))
            img = cv2.imread(os.path.join(pathFldMain, 'data', 'basis', fnImg))
            cDs, rDs, xs, ys, zs, cDhs, rDhs = [dsGH1s[posFnImg][item] for item in ['cDs', 'rDs', 'xs', 'ys', 'zs', 'cDhs', 'rDhs']]
            uli.PlotCalibration_2504(img, dsMCSs[posFnImg], cDs, rDs, xs, ys, zs, cDhs, rDhs, pathScrJpg)
    #
    return None
#
def AutoCalibrationOfFramesViaGCPs(pathFldMain):  # lm:2025-07-02; lm:2025-07-02
    #
    # obtain par and pathFldSAutoCal
    par = uli.GHLoadPar(pathFldMain)
    pathFldSAutoCal = os.path.join(pathFldMain, 'scratch', 'numerics', 'autocalibrations')
    #
    # obtain autoDone; IMP*: do not skip if autocalibrations already done, to allow easy changes in 'filtering'
    pathFldFrames = uli.GHDroneExtractVideoToPathFldFrames(pathFldMain, active=False)
    fnsFrames = sorted([item for item in os.listdir(pathFldFrames) if os.path.splitext(item)[1][1:] in extsImgs])
    if os.path.exists(os.path.join(pathFldMain, 'scratch', 'numerics', 'autocalibrations')):
        fnsCalTxts = sorted([item for item in os.listdir(os.path.join(pathFldMain, 'scratch', 'numerics', 'autocalibrations')) if item.endswith('cal.txt')])
    else:
        fnsCalTxts = []
    if set([os.path.splitext(item)[0] for item in fnsFrames]) == set([item[:-7] for item in fnsCalTxts]) and not par['overwrite_outputs']:
        errorTs = [uli.ReadCalTxt_2410(os.path.join(pathFldMain, 'scratch', 'numerics', 'autocalibrations', item), rangeC)[-1] for item in fnsCalTxts]
        print("{}{} All video frames already calibrated: {} calibrations found with errors ranging from {:.2f} to {:.2f} pixels {}".format(' '*dGit['ind'], dGit['sB1'], len(fnsCalTxts), min(errorTs), max(errorTs), dGit['sOK']))
        autoDone = True
    else:
        autoDone = False
    #
    # obtain selVarKeys and freUVarKeys
    selVarKeys = uli.GHLens2SelVarKeysOrFail(par['camera_lens_model'], lens2SelVarKeys)
    freUVarKeys = [item for item in selVarKeys if item not in freDVarKeys]  # IMP*: (U)nique, the same for all frames
    #
    # obtain autocalibrations
    if not autoDone:
        #
        # load basis information and dBasSD
        imgsB, kpssB, dessB, cDssB, rDssB, xssB, yssB, zssB, dsMCSsB, cUssB, rUssB = [{} for _ in range(11)]
        fnsImgsB = sorted([item for item in os.listdir(os.path.join(pathFldMain, 'data', 'basis')) if os.path.splitext(item)[1][1:] in extsImgs])
        for fnImgB in fnsImgsB:
            #
            # obtain fnImgBWE
            fnImgBWE = os.path.splitext(fnImgB)[0]
            #
            # update imgsB, kpssB and dessB
            img = cv2.imread(os.path.join(pathFldMain, 'data', 'basis', fnImgB))
            ncH, nrH, kps, des, ctrl = uli.Keypoints_2506(img, method=par['feature_detection_method'], nOfFeatures=par['max_features'])
            if not ctrl:
                continue
            imgsB[fnImgB], kpssB[fnImgB], dessB[fnImgB] = img, kps, des
            #
            # obtain dBasSD
            if 'nc' not in locals() or 'nr' not in locals():
                nc, nr = ncH, nrH
                dBasSD = uli.LoadDBasSD_2506(selVarKeys, nc, nr, rangeC, zr=par['z_sea_level'])
            else:
                assert nc == ncH and nr == nrH
            #
            # pdate cDssB, rDssB, xssB, yssB, zssB
            pathCdgTxt = os.path.join(pathFldMain, 'data', 'basis', '{}cdg.txt'.format(fnImgBWE))
            cDssB[fnImgB], rDssB[fnImgB], xssB[fnImgB], yssB[fnImgB], zssB[fnImgB] = uli.ReadCdgTxt_2502(pathCdgTxt, nc=nc, nr=nr)[:5]  # default readOnlyGood
            #
            # load calibrations and update dsMCSsB
            pathCalTxt = os.path.join(pathFldMain, 'scratch', 'numerics', 'calibration_basis', '{}cal.txt'.format(fnImgBWE))  # IMP*: forced calibrations
            allVar, ncH, nrH = uli.ReadCalTxt_2410(pathCalTxt, rangeC)[:3]
            assert (nrH, ncH) == (nr, nc)  # avoidable
            dsMCSsB[fnImgB] = uli.AllVar2DMCS_2410(allVar, nc, nr, rangeC, incHor=False)
            if 'dMCS0' not in locals():
                freUVar = uli.AllVar2SubVar_2410(allVar, freUVarKeys, rangeC)  # to be gvnVar here
                dFreUVar = uli.Array2Dictionary(freUVarKeys, freUVar)  # to be dGvnVar here
                dMCS0 = dsMCSsB[fnImgB]
            else:
                assert all(np.isclose(dsMCSsB[fnImgB][key], dMCS0[key]) for key in freUVarKeys)
            #
            # load cUssB and rUssB
            cUssB[fnImgB], rUssB[fnImgB] = uli.CDRD2CURU_2410(cDssB[fnImgB], rDssB[fnImgB], dMCS0, dMCS0, rangeC)[:2]  # WATCH OUT: all positions; only intrinsic, constant, is used from dMCS0
        #
        # obtain window
        window = int(0.025 * np.sqrt(nc * nr))  # IMP*: WATCH OUT: epsilon
        #
        # obtain autocalibration for each fnFrame
        for fnFrame in fnsFrames:
            #
            # obtain fnFrameWE and inform
            fnFrameWE = os.path.splitext(fnFrame)[0]
            print("{}{} Calibration of {}".format(' '*dGit['ind'], dGit['sB1'], fnFrameWE))
            #
            # obtain pathCalTxt and disregard
            pathCalTxt = os.path.join(pathFldMain, 'scratch', 'numerics', 'autocalibrations', '{}cal.txt'.format(fnFrameWE))
            if os.path.exists(pathCalTxt) and not par['overwrite_outputs']:
                errorT = uli.ReadCalTxt_2410(pathCalTxt, rangeC)[-1]
                if errorT <= par['max_reprojection_error_px']:
                    print("\033[F\033[K{}{} Frame {} was already calibrated: error = {:.2f} pixels {}".format(' '*dGit['ind'], dGit['sB1'], fnFrameWE, errorT, dGit['sOK']))  # WATCH OUT: formatting
                    continue
            #
            # obtain img and keypoints
            img = cv2.imread(os.path.join(pathFldFrames, fnFrame))
            ncH, nrH, kps, des, ctrl = uli.Keypoints_2506(img, method=par['feature_detection_method'], nOfFeatures=par['max_features'])
            if not ctrl:
                print("\033[F\033[K{}{} Frame {} could not be calibrated {}".format(' '*dGit['ind'], dGit['sB1'], fnFrameWE, dGit['sKO']))
                continue
            assert ncH == nc and nrH == nr
            #
            # obtain GCPs for the image to calibrate
            cDsGCP, rDsGCP, xsGCP, ysGCP, zsGCP = [[] for _ in range(5)]
            for fnImgB in fnsImgsB:
                # obtain pairs of distorted pixels
                cDs, rDs, cDsB, rDsB, ers = uli.Matches_2506(kps, des, kpssB[fnImgB], dessB[fnImgB], method=par['feature_detection_method'], nOfStd=2.0)
                possTMP = uli.SelectPixelsInAGrid_2506(cDs, rDs, ers, nc, nr, nOfBands=10)[0]
                cDs, rDs, cDsB, rDsB, ers = [item[possTMP] for item in [cDs, rDs, cDsB, rDsB, ers]]
                # obtain pairs of undistorted pixels
                cUs, rUs = uli.CDRD2CURU_2410(cDs, rDs, dMCS0, dMCS0, rangeC)[:2]  # only intrinsic, constant, is used from dMCS0
                cUsB, rUsB = uli.CDRD2CURU_2410(cDsB, rDsB, dMCS0, dMCS0, rangeC)[:2]  # only intrinsic, constant, is used from dMCS0
                # find homography from cUsB to cUs
                Ha = uli.FindHomographyH01ViaRANSAC_2504(cUsB, rUsB, cUs, rUs, par['max_reprojection_error_px'], margin=0.2)[0]  # margin is for RANSAC
                if Ha is None:
                    continue
                # obtain approximated pixel positions of the GCPs via the homography
                cUsApprox, rUsApprox = uli.ApplyHomographyH01_2504(Ha, cUssB[fnImgB], rUssB[fnImgB])
                cDsApprox, rDsApprox = uli.CURU2CDRD_2410(cUsApprox, rUsApprox, dMCS0, dMCS0, rangeC)[:2]  # only intrinsic, constant, is used from dMCS0
                # obtain refined pixel positions of the GCPs; homography is not valid since the camera is moving
                for pos in range(len(cDsApprox)):
                    # crop basis image
                    c0, r0 = int(cDssB[fnImgB][pos]), int(rDssB[fnImgB][pos])
                    if not (c0 > window+1 and nc-c0 > window+1 and r0 > window+1 and nr-r0 > window+1):
                        continue
                    img0 = imgsB[fnImgB][r0-window:r0+window, c0-window:c0+window, :]
                    # crop image to calibrate
                    c1, r1 = int(cDsApprox[pos]), int(rDsApprox[pos])
                    if not (c1 > window+1 and nc-c1 > window+1 and r1 > window+1 and nr-r1 > window+1):
                        continue
                    img1 = img[r1-window:r1+window, c1-window:c1+window, :]
                    # apply sift/orb
                    kps0, des0, ctrl0 = uli.Keypoints_2506(img0, method=par['feature_detection_method'], nOfFeatures=par['max_features'])[2:]
                    kps1, des1, ctrl1 = uli.Keypoints_2506(img1, method=par['feature_detection_method'], nOfFeatures=par['max_features'])[2:]
                    if not ctrl0 or not ctrl1:
                        continue
                    try:
                        cDs0, rDs0, cDs1, rDs1, ers = uli.Matches_2506(kps0, des0, kps1, des1, method=par['feature_detection_method'], nOfStd=1.0)
                    except Exception:
                        continue
                    if len(cDs0) < 5:
                        continue
                    dc, dr = np.mean(cDs1-cDs0), np.mean(rDs1-rDs0)  # ?IMPROVE?
                    # update xsGCP, ysGCP, zsGCP, csGCP and rsGCP
                    cDsGCP.append(c1+dc); rDsGCP.append(r1+dr); xsGCP.append(xssB[fnImgB][pos]); ysGCP.append(yssB[fnImgB][pos]); zsGCP.append(zssB[fnImgB][pos])
            cDsGCP, rDsGCP, xsGCP, ysGCP, zsGCP = map(np.asarray, [cDsGCP, rDsGCP, xsGCP, ysGCP, zsGCP])
            #
            # disregard
            if len(xsGCP) < 5:
                print("\033[F\033[K{}{} Frame {} could not be calibrated {}".format(' '*dGit['ind'], dGit['sB1'], fnFrameWE, dGit['sKO']))
                continue
            #
            # clean GCPs
            if False:
                possG = uli.GoodGCPs_2504(cDsGCP, rDsGCP, xsGCP, ysGCP, zsGCP, 1.0*par['max_reprojection_error_px'], rangeC, nc=nc, nr=nr)[0]  # WATCH OUT: epsilon
                cDsGCP, rDsGCP, xsGCP, ysGCP, zsGCP = [item[possG] for item in [cDsGCP, rDsGCP, xsGCP, ysGCP, zsGCP]]
            #
            # write GCPs
            pathCdgTxt = os.path.join(pathFldMain, 'scratch', 'numerics', 'autocalibrations', '{}cdg.txt'.format(fnFrameWE))
            uli.WriteCdgTxt_2502(pathCdgTxt, cDsGCP, rDsGCP, xsGCP, ysGCP, zsGCP)
            #
            # obtain first (auto)calibration, only with GCPs
            dGH1 = {'cDs': cDsGCP, 'rDs': rDsGCP, 'xs': xsGCP, 'ys': ysGCP, 'zs': zsGCP, 'cDhs': np.asarray([]), 'rDhs': np.asarray([]), 'nc': nc, 'nr': nr}
            dMCS, errorT = uli.ManualCalibration_2502(dBasSD, dGH1, dFreUVar, list(dsMCSsB.values()))  # IMP*: seeds from basis
            #
            # write first pathCalTxt and inform
            if dMCS is not None and errorT <= par['max_reprojection_error_px']:
                print("\033[F\033[K{}{} Frame {} successfully calibrated: error = {:.2f} pixels {}".format(' '*dGit['ind'], dGit['sB1'], fnFrameWE, errorT, dGit['sOK']))  # WATCH OUT: formatting
                uli.WriteCalTxt_2410(pathCalTxt, dMCS['allVar'], nc, nr, errorT, rangeC)
            else:
                print("\033[F\033[K{}{} Frame {} could not be calibrated {}".format(' '*dGit['ind'], dGit['sB1'], fnFrameWE, dGit['sKO']))
                continue
            #
            # obtain and write pathCdhTxt
            if par['enable_horizon_detection']:
                quality_min = 0.15
                pathCdhTxt = os.path.join(pathFldMain, 'scratch', 'numerics', 'autocalibrations', '{}cdh.txt'.format(fnFrameWE))
                dMCSApp = uli.LoadDMCSFromCalTxt_2502(pathCalTxt, rangeC, incHor=True, zr=par['z_sea_level'])
                cDhs, rDhs, quality = uli.DetectHorizon(img, dMCSApp['dHor'], quality_min=quality_min)
                if quality > quality_min:  # WATCH OUT: epsilon
                    cDhs, rDhs = [item[::int(len(cDhs)/100)] for item in [cDhs, rDhs]]  # WATCH OUT: epsilon; 100 points at most
                    uli.WriteCdhTxt_2504(pathCdhTxt, cDhs, rDhs)
            #
            # obtain and write second pathCalTxt
            if par['enable_horizon_detection'] and os.path.exists(pathCdhTxt):
                dGH1 = {'cDs': cDsGCP, 'rDs': rDsGCP, 'xs': xsGCP, 'ys': ysGCP, 'zs': zsGCP, 'cDhs': cDhs, 'rDhs': rDhs, 'nc': nc, 'nr': nr}
                dMCS, errorT = uli.ManualCalibration_2502(dBasSD, dGH1, dFreUVar, list(dsMCSsB.values()))
                if dMCS is not None and errorT < par['max_reprojection_error_px']:
                    print("\033[F\033[K{}{} Frame {} successfully calibrated using the horizon: error = {:.2f} pixels {}".format(' '*dGit['ind'], dGit['sB1'], fnFrameWE, errorT, dGit['sOK']))  # WATCH OUT: formatting
                    uli.WriteCalTxt_2410(pathCalTxt, dMCS['allVar'], nc, nr, errorT, rangeC)
            #
            # write pathScrJpg
            if par['generate_scratch_plots']:
                pathScrJpg = os.path.join(pathFldMain, 'scratch', 'plots', 'autocalibrations', '{}cal.jpg'.format(fnFrameWE))
                img = cv2.imread(os.path.join(pathFldFrames, fnFrame))
                cDs, rDs, xs, ys, zs, cDhs, rDhs = [dGH1[item] for item in ['cDs', 'rDs', 'xs', 'ys', 'zs', 'cDhs', 'rDhs']]
                uli.PlotCalibration_2504(img, dMCS, cDs, rDs, xs, ys, zs, cDhs, rDhs, pathScrJpg)
    #
    # obtain and write filtered calibrations
    if par['outlier_filtering_window_sec'] > 0:  # IMP*: rewrites always; perfomance
        pathFldSAutoCalF = os.path.join(pathFldMain, 'scratch', 'numerics', 'autocalibrations_filtered')
        uli.PathFldVideoCal2FilterExtrinsic_2504(pathFldSAutoCal, pathFldSAutoCalF, rangeC, par['outlier_filtering_window_sec'], nsOfStds=[5, 4, 3], length_stamp=12, ending='cal.txt')
        # write pathsScrJpgs
        if par['generate_scratch_plots']:
            for fnFrame in fnsFrames:
                fnFrameWE = os.path.splitext(fnFrame)[0]
                pathCalfTxt = os.path.join(pathFldMain, 'scratch', 'numerics', 'autocalibrations_filtered', '{}cal.txt'.format(fnFrameWE))
                pathCdgTxt = os.path.join(pathFldMain, 'scratch', 'numerics', 'autocalibrations', '{}cdg.txt'.format(fnFrameWE))
                if not os.path.exists(pathCalfTxt) or not os.path.exists(pathCdgTxt):
                    continue
                dMCS = uli.LoadDMCSFromCalTxt_2502(pathCalfTxt, rangeC, incHor=True, zr=par['z_sea_level'])
                cDs, rDs, xs, ys, zs = uli.ReadCdgTxt_2502(pathCdgTxt, nc=dMCS['nc'], nr=dMCS['nr'])[:5]
                pathCdhTxt = os.path.join(pathFldMain, 'scratch', 'numerics', 'autocalibrations', '{}cdh.txt'.format(fnFrameWE))
                cDhs, rDhs = uli.ReadCdhTxt_2504(pathCdhTxt, nc=dMCS['nc'], nr=dMCS['nr'])
                pathScrJpg = os.path.join(pathFldMain, 'scratch', 'plots', 'autocalibrations_filtered', '{}cal.jpg'.format(fnFrameWE))
                img = cv2.imread(os.path.join(pathFldFrames, fnFrame))
                uli.PlotCalibration_2504(img, dMCS, cDs, rDs, xs, ys, zs, cDhs, rDhs, pathScrJpg)
    #
    # plot pathScrJpg
    pathScrJpg = os.path.join(pathFldMain, 'scratch', 'plots', 'extrinsic_parameters.jpg')
    if par['outlier_filtering_window_sec'] > 0:  # IMP*: rewrites always; perfomance
        uli.GHDronePlotExtrinsic_2504(pathFldSAutoCal, pathScrJpg, dGit['fw'], dGit['fh'], dGit['fontsize'], dGit['dpiHQ'], pathFldB=pathFldSAutoCalF, length_stamp=12, ending='cal.txt')
    else:
        if not os.path.exists(pathScrJpg) or par['overwrite_outputs']:
            uli.GHDronePlotExtrinsic_2504(pathFldSAutoCal, pathScrJpg, dGit['fw'], dGit['fh'], dGit['fontsize'], dGit['dpiHQ'], pathFldB=None, length_stamp=12, ending='cal.txt')
    #
    return None
#
def PlanviewsFromImages(pathFldMain):  # lm:2025-07-02; lr:2025-07-10
    #
    # obtain par, pathFldFrames and pathFldSAutoCal
    par = uli.GHLoadPar(pathFldMain)
    pathFldFrames = uli.GHDroneExtractVideoToPathFldFrames(pathFldMain, active=False)
    if par['outlier_filtering_window_sec'] > 0:
        pathFldSAutoCal = os.path.join(pathFldMain, 'scratch', 'numerics', 'autocalibrations_filtered')
    else:
        pathFldSAutoCal = os.path.join(pathFldMain, 'scratch', 'numerics', 'autocalibrations')
    #
    # skip if planviews already done
    if os.path.exists(os.path.join(pathFldMain, 'output', 'plots', 'planviews')):
        fnsPlws = os.listdir(os.path.join(pathFldMain, 'output', 'plots', 'planviews'))
    else:
        fnsPlws = []
    if os.path.exists(os.path.join(pathFldMain, 'scratch', 'numerics', 'autocalibrations')):
        fnsCalTxts = sorted([item for item in os.listdir(os.path.join(pathFldMain, 'scratch', 'numerics', 'autocalibrations')) if item.endswith('cal.txt')])
    else:
        fnsCalTxts = []
    tcks0 = set(np.loadtxt(os.path.join(pathFldMain, 'data', 'timestacks_xyz.txt'), usecols=0, dtype=str))
    if os.path.exists(os.path.join(pathFldMain, 'output', 'plots', 'timestacks')):
        tcks = [os.path.splitext(item)[0].split('_')[1] for item in os.listdir(os.path.join(pathFldMain, 'output', 'plots', 'timestacks'))]
    else:
        tcks = []
    if set([os.path.splitext(item)[0][:-3] for item in fnsPlws]) == set([item[:-7] for item in fnsCalTxts]) and set(tcks0) == set(tcks) and not par['overwrite_outputs']:
        print("{}{} All planviews and timestacks were already generated: {} planviews and {} timestacks found {}".format(' '*dGit['ind'], dGit['sB1'], len(fnsPlws), len(tcks), dGit['sOK']))
        return None
    #
    # planviews preliminaries: obtain dPdf
    try:
        dataTMP = np.loadtxt(os.path.join(pathFldMain, 'data', 'planviews_xy.txt'), usecols=range(2), dtype=float, ndmin=2)
    except Exception as eTMP:
        print("! Unable to read planviews_xy.txt in /data directory: ({}) {}".format(eTMP, dGit['sKO']))
        sys.exit()
    if dataTMP.shape[0] < 3:
        print("! File planviews_xy.txt in /data directory must contain at least 3 points {}".format(dGit['sKO']))
        sys.exit()
    xsCloud, ysCloud = dataTMP[:, 0], dataTMP[:, 1]
    dPdf = uli.CloudAndPpm2DPdf_2504(xsCloud, ysCloud, par['ppm_for_planviews'])
    #
    # planviews preliminaries; write planviews_crxyz.txt
    pathTMPTxt = os.path.join(pathFldMain, 'output', 'numerics', 'planviews', 'planviews_crxyz.txt')
    os.makedirs(os.path.dirname(pathTMPTxt), exist_ok=True)
    with open(pathTMPTxt, 'w') as fileout:
        assert len(dPdf['csC']) == 4  # avoidable check
        for pos in range(len(dPdf['csC'])):
            fileout.write('{:12.0f} {:12.0f} {:15.3f} {:15.3f} {:15.3f}\t c, r, x, y and z\n'.format(dPdf['csC'][pos], dPdf['rsC'][pos], dPdf['xsC'][pos], dPdf['ysC'][pos], par['z_sea_level']))  # IMP*: formatting
    if par['generate_ubathy_data']:
        pathTMPUBathyTxt = os.path.join(pathFldMain, 'output', 'ubathy', 'planviews_crxyz.txt')
        os.makedirs(os.path.dirname(pathTMPUBathyTxt), exist_ok=True)
        shutil.copy2(pathTMPTxt, pathTMPUBathyTxt)
    #
    # timestacks preliminaries; obtain dXsT, dYsT and dZsT and write timestack_{}_cxyz.txt
    try:
        dXsT, dYsT, dZsT = uli.GHDroneReadTimestacksTxt(pathFldMain, par['ppm_for_timestacks'])
    except Exception as eTMP:
        print("! Unable to read timestacks_xyz.txt in /data directory: ({}) {}".format(eTMP, dGit['sKO']))
        sys.exit()
    #
    # planviews and timestacks preliminaries; obtain fnsFrames, fnsFramesCal, fnsFramesT, nrT and write timestacks_rt.txt
    fnsFrames = sorted([item for item in os.listdir(pathFldFrames) if os.path.splitext(item)[1][1:].lower() in extsImgs])
    fnsFramesCal = [item for item in fnsFrames if os.path.exists(os.path.join(pathFldSAutoCal, '{}cal.txt'.format(os.path.splitext(item)[0])))]  # IMP*
    if par['include_gaps_in_timestack']:
        fnsFramesT = fnsFrames
    else:
        fnsFramesT = fnsFramesCal
    pathTMPTxt = os.path.join(pathFldMain, 'output', 'numerics', 'timestacks', 'timestacks_rt.txt')
    os.makedirs(os.path.dirname(pathTMPTxt), exist_ok=True)
    with open(pathTMPTxt, 'w') as fileout:
        for posFnFrameT, fnFrameT in enumerate(fnsFramesT):
            fileout.write('{:12.0f} {:>50}\t r and filename\n'.format(posFnFrameT, fnFrameT))  # WATCH OUT: formatting
    #
    # initialize dImgT; dictionary of timestack images
    dImgT = {}
    for code in dXsT:
        dImgT[code] = np.zeros((len(fnsFramesT), len(dXsT[code]), 3), dtype=float)  # nr x nc x 3; IMP*: floats
    #
    # obtain and write planviews and timestacks
    for fnFrame in fnsFrames:
        #
        # obtain posInT and load img
        if fnFrame in fnsFramesCal:  # IMP*
            posInT = fnsFramesT.index(fnFrame)
            img = cv2.imread(os.path.join(pathFldFrames, fnFrame))
        else:
            continue
        #
        # obtain fnImgWE and inform
        fnFrameWE = os.path.splitext(fnFrame)[0]
        print("{}{} Creating planview for frame {}".format(' '*dGit['ind'], dGit['sB1'], fnFrameWE))
        #
        # obtain pathCalTxt and dMCS
        pathCalTxt = os.path.join(pathFldSAutoCal, '{}cal.txt'.format(fnFrameWE))  # exists
        dMCS = uli.LoadDMCSFromCalTxt_2502(pathCalTxt, rangeC, incHor=True, zr=par['z_sea_level'])
        #
        # obtain and write planview and pathScrJpg
        pathPlwPng = os.path.join(pathFldMain, 'output', 'plots', 'planviews', '{}plw.png'.format(fnFrameWE))  # WATCH OUT: plw.png, nomenclature
        if os.path.exists(pathPlwPng) and not par['overwrite_outputs']:
            print("\033[F\033[K{}{} Planview for frame {} already exists {}".format(' '*dGit['ind'], dGit['sB1'], fnFrameWE, dGit['sOK']))
        else:
            # write pathPlwPng and inform
            dPlwPC = uli.DPlwPC_2504({'01': dMCS}, dPdf, par['z_sea_level'])
            imgPlw = uli.CreatePlw_2504(dPlwPC, {'01': img})
            os.makedirs(os.path.dirname(pathPlwPng), exist_ok=True)
            cv2.imwrite(pathPlwPng, imgPlw)
            print("\033[F\033[K{}{} Planview for frame {} successfully generated {}".format(' '*dGit['ind'], dGit['sB1'], fnFrameWE, dGit['sOK']))
            # write pathScrJpg
            if par['generate_scratch_plots']:
                pathScrJpg = os.path.join(pathFldMain, 'scratch', 'plots', 'planviews', '{}.jpg'.format(fnFrameWE))
                csTMP, rsTMP = uli.XYZ2CDRD_2410(dPdf['xs'], dPdf['ys'], par['z_sea_level']*np.ones(dPdf['xs'].shape), dMCS)[:2]  # WATCH OUT: all positions
                uli.DisplayCRInImage_2504(img, csTMP, rsTMP, factor=0.1, colors=[[0, 255, 255]], pathOut=pathScrJpg)  # IMP*: formatting
        if par['generate_ubathy_data']:
            pathPlwPngUBathy = os.path.join(pathFldMain, 'output', 'ubathy', 'planviews', '{}plw.png'.format(fnFrameWE))  # WATCH OUT: plw.png, nomenclature
            os.makedirs(os.path.dirname(pathPlwPngUBathy), exist_ok=True)
            shutil.copy2(pathPlwPng, pathPlwPngUBathy)
        #
        # update timestacks and write pathScrJpg
        for code in dXsT:
            cDsH, rDsH, possGH = uli.XYZ2CDRD_2410(dXsT[code], dYsT[code], dZsT[code], dMCS, rtrnPossG=True, margin=1)
            csIA, rsIA, wsA = uli.CR2CRIntegerAroundAndWeights_2504(cDsH, rDsH)
            for pos, posCorner in itertools.product(possGH, range(4)):  
                dImgT[code][posInT, pos, :] += wsA[pos, posCorner] * img[rsIA[pos, posCorner], csIA[pos, posCorner], :]
            if par['generate_scratch_plots']:
                pathScrJpg = os.path.join(pathFldMain, 'scratch', 'plots', 'timestacks', 'timestack_{}'.format(code), '{}.jpg'.format(fnFrameWE))
                uli.DisplayCRInImage_2504(img, cDsH[possGH], rDsH[possGH], factor=0.2, colors=[[0, 255, 255]], pathOut=pathScrJpg)  # IMP*: formatting
    #
    # write timestacks
    for code in sorted(dXsT):
        pathTckPng = os.path.join(pathFldMain, 'output', 'plots', 'timestacks', 'timestack_{}.png'.format(code))  # WATCH OUT: plw.png, nomenclature
        os.makedirs(os.path.dirname(pathTckPng), exist_ok=True)
        cv2.imwrite(pathTckPng, dImgT[code].astype(np.uint8))
        print("{}{} Timestack {} successfully generated {}".format(' '*dGit['ind'], dGit['sB1'], code, dGit['sOK']))
    #
    # obtain and write mean and sigma images
    if os.path.exists(os.path.join(pathFldMain, 'output', 'plots', 'planviews')):
        pathsImgs = [item.path for item in os.scandir(os.path.join(pathFldMain, 'output', 'plots', 'planviews')) if item.is_file() and os.path.splitext(item.name)[1][1:] in extsImgs]
        imgMea, imgSig = uli.MeanAndSigmaOfImages(pathsImgs)
        pathMeaPng = os.path.join(pathFldMain, 'output', 'plots', 'mean.png')
        os.makedirs(os.path.dirname(pathMeaPng), exist_ok=True)
        cv2.imwrite(pathMeaPng, imgMea)
        pathSigPng = os.path.join(pathFldMain, 'output', 'plots', 'sigma.png')
        os.makedirs(os.path.dirname(pathSigPng), exist_ok=True)
        cv2.imwrite(pathSigPng, imgSig)
    #
    return None
    #
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
