#'''
# Created on 2021 by Gonzalo Simarro and Daniel Calvete
#'''
#
import os
import cv2
import numpy as np
import shutil
#
import ulises_udrone as ulises
#
def Video2Frames(pathFolderVideo , pathFolderSnaps, fps):
    #
    # obtain pathVideo
    pathVideo = [pathFolderVideo + os.sep + item for item in os.listdir(pathFolderVideo) if item[item.rfind('.')+1:] in ['mp4', 'MP4', 'avi', 'AVI']][0]
    print('... frame extraction from video {:}'.format(pathVideo[pathVideo.rfind(os.sep)+1:]))
    #
    # load video and obtain fps
    fpsOfVideo = cv2.VideoCapture(pathVideo).get(cv2.CAP_PROP_FPS)
    if fps > fpsOfVideo:
        print('*** Frames per second of the video ({:2.1f}) smaller than given FPS = {:2.1f}'.format(fpsOfVideo, fps))
        print('*** Select a smaller FPS ***'); exit()
    if fps == 0:
        fps = fpsOfVideo
    else :
        fps = min([fpsOfVideo / int(fpsOfVideo / fps), fpsOfVideo])
    #
    # write frames
    ulises.Video2Snaps(pathVideo, pathFolderSnaps, fps)
    #
    return None
#
def CalibrationOfBasisImages(pathBasis, errorTCritical, model, verbosePlot):
    #
    # manage model
    model2SelectedVariablesKeys = {'parabolic':['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'sca'], 'quartic':['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'sca'], 'full':['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or']}
    if model not in model2SelectedVariablesKeys.keys():
        print('*** Invalid calibration model {:}'.format(model))
        print('*** Choose one of the following calibration models: {:}'.format(list(model2SelectedVariablesKeys.keys()))); exit()
    #
    # obtain calibrations
    fnsImages = sorted([item for item in os.listdir(pathBasis) if '.' in item and item[item.rfind('.')+1:] in ['jpeg', 'JPEG', 'jpg', 'JPG', 'png', 'PNG']])
    for posFnImage, fnImage in enumerate(fnsImages):
        #
        print('... calibration of {:}'.format(fnImage), end='', flush=True)
        #
        # load image information and dataBasic
        if posFnImage == 0:
            nr, nc = cv2.imread(pathBasis + os.sep + fnImage).shape[0:2]
            dataBasic = ulises.LoadDataBasic0(options={'nc':nc, 'nr':nr, 'selectedVariablesKeys':model2SelectedVariablesKeys[model]})
        else:
            assert cv2.imread(pathBasis + os.sep + fnImage).shape[0:2] == (nr, nc)
        #
        # load GCPs
        pathCdgTxt = pathBasis + os.sep + fnImage[0:fnImage.rfind('.')] + 'cdg.txt'
        cs, rs, xs, ys, zs = ulises.ReadCdgTxt(pathCdgTxt, options={'readCodes':False, 'readOnlyGood':True})[0:5]
        #
        # load horizon points
        pathCdhTxt = pathBasis + os.sep + fnImage[0:fnImage.rfind('.')] + 'cdh.txt'
        if os.path.exists(pathCdhTxt):
            chs, rhs = ulises.ReadCdhTxt(pathCdhTxt, options={'readOnlyGood':True})
        else:
            chs, rhs = [np.asarray([]) for item in range(2)]
        #
        # load dataForCal and obtain calibration (aG, aH, mainSetSeeds are in dataForCal)
        dataForCal = {'nc':nc, 'nr':nr, 'cs':cs, 'rs':rs, 'xs':xs, 'ys':ys, 'zs':zs, 'aG':1., 'mainSetSeeds':[]} # IMP* to initialize mainSetSeeds
        if len(chs) == len(rhs) > 0:
            dataForCal['chs'], dataForCal['rhs'], dataForCal['aH'] = chs, rhs, 1.
        for filename in [item for item in os.listdir(pathBasis) if 'cal' in item and item[-3:] == 'txt']:
            allVariablesH, ncH, nrH = ulises.ReadCalTxt(pathBasis + os.sep + filename)[0:3]
            dataForCal['mainSetSeeds'].append(ulises.AllVariables2MainSet(allVariablesH, ncH, nrH, options={}))
        subsetVariablesKeys, subCsetVariablesDictionary = model2SelectedVariablesKeys[model], {}
        mainSet, errorT = ulises.NonlinearManualCalibration(dataBasic, dataForCal, subsetVariablesKeys, subCsetVariablesDictionary, options={})
        #
        # inform and write
        if errorT <= 1. * errorTCritical:
            print(' success')
            # check errorsG
            csR, rsR = ulises.XYZ2CDRD(mainSet, xs, ys, zs, options={})[0:2]
            errorsG = np.sqrt((csR - cs) ** 2 + (rsR - rs) ** 2)
            for pos in range(len(errorsG)):
                if errorsG[pos] > errorTCritical:
                    print('*** the error of GCP at x = {:8.2f}, y = {:8.2f} and z = {:8.2f} is {:5.1f} > critical error = {:5.1f}: consider to check or remove it'.format(xs[pos], ys[pos], zs[pos], errorsG[pos], errorTCritical))
            # write pathCal0Txt
            pathCal0Txt = pathBasis + os.sep + fnImage[0:fnImage.rfind('.')] + 'cal0.txt'
            ulises.WriteCalTxt(pathCal0Txt, mainSet['allVariables'], mainSet['nc'], mainSet['nr'], errorT)
            # manage verbosePlot
            if verbosePlot:
                pathTMP = pathBasis + os.sep + '..' + os.sep + 'TMP'
                ulises.MakeFolder(pathTMP)
                ulises.PlotMainSet(pathBasis + os.sep + fnImage, mainSet, cs, rs, xs, ys, zs, chs, rhs, pathTMP + os.sep + fnImage.replace('.', 'cal0_check.'))
        else:
            print(' failed (error = {:6.1f})'.format(errorT))
            print('*** re-run and, if it keeps failing, check quality of the GCP or try another calibration model ***')
    #
    return None
#
def CalibrationOfBasisImagesConstantIntrinsic(pathBasis, model, verbosePlot):
    #
    # manage model
    model2SelectedVariablesKeys = {'parabolic':['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'sca'], 'quartic':['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'sca'], 'full':['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or']}
    if model not in model2SelectedVariablesKeys.keys():
        print('*** Invalid calibration model {:}'.format(model))
        print('*** Choose one of the following calibration models: {:}'.format(list(model2SelectedVariablesKeys.keys()))); exit()
    #
    # load basis information
    ncs, nrs, css, rss, xss, yss, zss, chss, rhss, allVariabless, mainSets, errorTs = [[] for item in range(12)]
    fnsImages = sorted([item for item in os.listdir(pathBasis) if '.' in item and item[item.rfind('.')+1:] in ['jpeg', 'JPEG', 'jpg', 'JPG', 'png', 'PNG']])
    for posFnImage, fnImage in enumerate(fnsImages):
        #
        # load image information and dataBasic
        if posFnImage == 0:
            nr, nc = cv2.imread(pathBasis + os.sep + fnImage).shape[0:2]
            dataBasic = ulises.LoadDataBasic0(options={'nc':nc, 'nr':nr, 'selectedVariablesKeys':model2SelectedVariablesKeys[model]})
        else:
            assert cv2.imread(pathBasis + os.sep + fnImage).shape[0:2] == (nr, nc)
        ncs.append(nc); nrs.append(nr)
        #
        # load GCPs
        pathCdgTxt = pathBasis + os.sep + fnImage[0:fnImage.rfind('.')] + 'cdg.txt'
        cs, rs, xs, ys, zs = ulises.ReadCdgTxt(pathCdgTxt, options={'readCodes':False, 'readOnlyGood':True})[0:5]
        css.append(cs); rss.append(rs); xss.append(xs); yss.append(ys); zss.append(zs)
        #
        # load horizon points
        pathCdhTxt = pathBasis + os.sep + fnImage[0:fnImage.rfind('.')] + 'cdh.txt'
        if os.path.exists(pathCdhTxt):
            chs, rhs = ulises.ReadCdhTxt(pathCdhTxt, options={'readOnlyGood':True})
        else:
            chs, rhs = [np.asarray([]) for item in range(2)]
        chss.append(chs); rhss.append(rhs)
        #
        # load allVariables, mainSet and errorT
        pathCal0Txt = pathBasis + os.sep + fnImage[0:fnImage.rfind('.')] + 'cal0.txt'
        allVariables, nc, nr, errorT = ulises.ReadCalTxt(pathCal0Txt)
        mainSet = ulises.AllVariables2MainSet(allVariables, nc, nr, options={})
        allVariabless.append(allVariables); mainSets.append(mainSet); errorTs.append(errorT)
    #
    # obtain calibrations and write pathCalTxts forcing unique xc, yc, zc and intrinsic
    if len(fnsImages) == 1:
        pathCal0Txt = pathBasis + os.sep + fnsImages[0][0:fnsImages[0].rfind('.')] + 'cal0.txt'
        pathCalTxt = pathBasis + os.sep + fnsImages[0][0:fnsImages[0].rfind('.')] + 'cal.txt'
        shutil.copyfile(pathCal0Txt, pathCalTxt)
    else:
        subsetVariabless, subsetVariablesKeys = [], ['xc', 'yc', 'zc', 'ph', 'sg', 'ta']
        subCsetVariabless, subCsetVariablesKeys = [], [item for item in model2SelectedVariablesKeys[model] if item not in subsetVariablesKeys]
        for pos in range(len(fnsImages)):
            subsetVariabless.append(ulises.AllVariables2SubsetVariables(dataBasic, allVariabless[pos], subsetVariablesKeys, options={}))
            subCsetVariabless.append(ulises.AllVariables2SubsetVariables(dataBasic, allVariabless[pos], subCsetVariablesKeys, options={}))
        mainSets, errorTs = ulises.NonlinearManualCalibrationForcingUniqueSubCset(dataBasic, ncs, nrs, css, rss, xss, yss, zss, chss, rhss, subsetVariabless, subsetVariablesKeys, subCsetVariabless, subCsetVariablesKeys, options={'aG':1., 'aH':1.}) # IMP* aG and aH
        for pos in range(len(fnsImages)):
            ulises.WriteCalTxt(pathBasis + os.sep + fnsImages[pos][0:fnsImages[pos].rfind('.')] + 'cal.txt', mainSets[pos]['allVariables'], mainSets[pos]['nc'], mainSets[pos]['nr'], errorTs[pos])
    #
    # manage verbosePlot
    if verbosePlot:
        pathTMP = pathBasis + os.sep + '..' + os.sep + 'TMP'
        ulises.MakeFolder(pathTMP)
        for pos in range(len(fnsImages)):
            ulises.PlotMainSet(pathBasis + os.sep + fnsImages[pos], mainSets[pos], css[pos], rss[pos], xss[pos], yss[pos], zss[pos], chss[pos], rhss[pos], pathTMP + os.sep + fnsImages[pos].replace('.', 'cal_check.'))
    #
    return None
#
def AutoCalibrationOfFramesViaGCPs(pathBasis, pathFrames, verbosePlot):
    #
    nOfFeaturesORB, aG = 20000, 1.
    #
    # load basic information
    pathCalTxt = [pathBasis + os.sep + item for item in os.listdir(pathBasis) if 'cal' in item and 'cal0' not in item and item[-3:] == 'txt'][0] # only the intrinsic, constant, is used
    allVariables, nc, nr = ulises.ReadCalTxt(pathCalTxt)[0:3]
    dataBasic = ulises.LoadDataBasic0({'nc':nc, 'nr':nr, 'selectedVariablesKeys':['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or']})
    subsetVariablesKeys, subCsetVariablesKeys = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta'], ['k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or'] # IMP*
    subCsetVariables = ulises.AllVariables2SubsetVariables(dataBasic, allVariables, subCsetVariablesKeys, options={})
    subCsetVariablesDictionary = ulises.Array2Dictionary(subCsetVariablesKeys, subCsetVariables)
    window = np.int(0.025 * np.sqrt(nc * nr)) 
    #
    # load basis information
    print('... loading basis information')
    imgsB, kpssB, dessB, cssB, rssB, xssB, yssB, zssB, mainSetsB, cUssB, rUssB = [{} for item in range(11)]
    fnsBasisImages = sorted([item for item in os.listdir(pathBasis) if '.' in item and item[item.rfind('.')+1:] in ['jpeg', 'JPEG', 'jpg', 'JPG', 'png', 'PNG']])
    for fnBasisImage in fnsBasisImages:
        #
        print('... loading information for {:}'.format(fnBasisImage))
        #
        # load image and obtain keypoints
        img = cv2.imread(pathBasis + os.sep + fnBasisImage)
        assert img.shape[0:2] == (nr, nc)
        kps, des, ctrl = ulises.ORBKeypoints(img, {'nOfFeatures':nOfFeaturesORB})[2:] # nc and nr are not loaded
        if not ctrl:
            continue
        imgsB[fnBasisImage], kpssB[fnBasisImage], dessB[fnBasisImage] = img, kps, des
        #
        # load GCPs (distorted pixels and xyz coordinates)
        pathCdgTxt = pathBasis + os.sep + fnBasisImage[0:fnBasisImage.rfind('.')] + 'cdg.txt'
        cssB[fnBasisImage], rssB[fnBasisImage], xssB[fnBasisImage], yssB[fnBasisImage], zssB[fnBasisImage] = ulises.ReadCdgTxt(pathCdgTxt)[0:5]
        #
        # load calibrations
        pathCalTxt = pathBasis + os.sep + fnBasisImage[0:fnBasisImage.rfind('.')] + 'cal.txt'
        allVariables, ncH, nrH = ulises.ReadCalTxt(pathCalTxt)[0:3]
        assert ncH == nc and nrH == nr
        mainSetsB[fnBasisImage] = ulises.AllVariables2MainSet(allVariables, nc, nr, options={}) #!
        #
        # load GCPs (undistorted pixels)
        cUssB[fnBasisImage], rUssB[fnBasisImage] = ulises.CDRD2CURU(mainSetsB[fnBasisImage], cssB[fnBasisImage], rssB[fnBasisImage]) # only the intrinsic, constant, is used
    #
    # obtain calibrations and write pathCalTxts
    fnsFrames = sorted([item for item in os.listdir(pathFrames) if '.' in item and item[item.rfind('.')+1:] in ['jpeg', 'JPEG', 'jpg', 'JPG', 'png', 'PNG']])
    for fnFrame in fnsFrames:
        #
        print('... calibration of {:}'.format(fnFrame), end='', flush=True)
        #
        # obtain pathCalTxt and check if already exists
        pathCalTxt = pathFrames + os.sep + fnFrame[0:fnFrame.rfind('.')] + 'cal.txt'
        if os.path.exists(pathCalTxt):
            print(' already calibrated'); continue
        #
        # obtain keypoints for the image to calibrate
        img = cv2.imread(pathFrames + os.sep + fnFrame)
        kps, des, ctrl = ulises.ORBKeypoints(img, {'nOfFeatures':nOfFeaturesORB})[2:] # nc and nr are not loaded
        if not ctrl:
            print(' not calibratable (lack of keypoints)'); continue
        #
        # find GCPs for the image to calibrate
        csGCP, rsGCP, xsGCP, ysGCP, zsGCP = [[] for item in range(5)] # image to calibrate
        for fnBasisImage in fnsBasisImages:
            # obtain pairs of distorted pixels (ORB)
            try:
                cs, rs, csB, rsB, ers = ulises.ORBMatches(kps, des, kpssB[fnBasisImage], dessB[fnBasisImage], {'erMaximum':30., 'nOfStd':1.}) # WATCH OUT
                poss06 = ulises.SelectPixelsInGrid(6, nc, nr, cs, rs, ers)[0]
                assert len(poss06) >= 6
                cs, rs, csB, rsB, ers = [item[poss06] for item in [cs, rs, csB, rsB, ers]]
            except:
                continue
            # obtain pairs of undistorted pixels
            cUs, rUs = ulises.CDRD2CURU(mainSetsB[fnBasisImage], cs, rs) # only the intrinsic, constant, is used
            cUsB, rUsB = ulises.CDRD2CURU(mainSetsB[fnBasisImage], csB, rsB) # only the intrinsic, constant, is used
            # find homography from cUsB(1) to cUs(2)
            parametersRANSAC = {'p':0.99999, 'e':0.8, 's':4, 'errorC':2.0}
            Ha, possGood = ulises.FindHomographyHa01ViaRANSAC(cUsB, rUsB, cUs, rUs, parametersRANSAC)
            # obtain approximated pixel positions of the GCPs via the homography
            cUsApprox, rUsApprox = ulises.ApplyHomographyHa01(Ha, cUssB[fnBasisImage], rUssB[fnBasisImage])
            csApprox, rsApprox = ulises.CURU2CDRD(mainSetsB[fnBasisImage], cUsApprox, rUsApprox) # only the intrinsic, constant, is used
            # obtain refined pixel positions of the GCPs via the homography
            for pos in range(len(csApprox)):
                # crop basis image
                c0, r0 = int(cssB[fnBasisImage][pos]), int(rssB[fnBasisImage][pos])
                if not (c0 > window+1 and nc-c0 > window+1 and r0 > window+1 and nr-r0 > window+1):
                    continue
                img0 = imgsB[fnBasisImage][r0-window:r0+window, c0-window:c0+window, :]
                # crop image to calibrate
                c1, r1 = int(csApprox[pos]), int(rsApprox[pos])
                if not (c1 > window+1 and nc-c1 > window+1 and r1 > window+1 and nr-r1 > window+1):
                    continue
                img1 = img[r1-window:r1+window, c1-window:c1+window, :]
                # apply ORB
                nc0, nr0, kps0, des0, ctrl0 = ulises.ORBKeypoints(img0, options={'nOfFeatures':1000})
                nc1, nr1, kps1, des1, ctrl1 = ulises.ORBKeypoints(img1, options={'nOfFeatures':1000})
                if not (ctrl0 and ctrl1):
                    continue
                cs0, rs0, cs1, rs1, ers = ulises.ORBMatches(kps0, des0, kps1, des1, options={'erMaximum':50., 'nOfStd':1.0}) # WATCH OUT
                if len(cs0) < 5:
                    continue
                dc, dr = np.mean(cs1-cs0), np.mean(rs1-rs0)
                # update xsGCP, ysGCP, zsGCP, csGCP and rsGCP
                csGCP.append(c1+dc); rsGCP.append(r1+dr)
                xsGCP.append(xssB[fnBasisImage][pos]); ysGCP.append(yssB[fnBasisImage][pos]); zsGCP.append(zssB[fnBasisImage][pos])
        xsGCP, ysGCP, zsGCP, csGCP, rsGCP = [np.asarray(item) for item in [xsGCP, ysGCP, zsGCP, csGCP, rsGCP]]
        #
        # obtain (auto)calibration
        dataForCal = {'nc':nc, 'nr':nr, 'cs':csGCP, 'rs':rsGCP, 'xs':xsGCP, 'ys':ysGCP, 'zs':zsGCP, 'aG':aG}
        dataForCal['mainSetSeeds'] = [mainSetsB[item] for item in fnsBasisImages]
        mainSet, errorT = ulises.NonlinearManualCalibration(dataBasic, dataForCal, subsetVariablesKeys, subCsetVariablesDictionary, options={})
        #
        # write pathCalTxt
        if mainSet is not None and errorT is not None and errorT < 10: # WATCH OUT Daniel
            print(' success')
            ulises.WriteCalTxt(pathCalTxt, mainSet['allVariables'], mainSet['nc'], mainSet['nr'], errorT)
        else:
            print(' failed'); continue
        #
        # manage verbosePlot
        if mainSet is not None and errorT is not None and verbosePlot: 
            pathTMP = pathFrames + os.sep + '..' + os.sep + 'TMP'
            ulises.MakeFolder(pathTMP)
            imgVerbose = ulises.DisplayCRInImage(img, csGCP, rsGCP, options={'colors':[[0, 0, 0]], 'size':4})
            cs, rs = ulises.XYZ2CDRD(mainSet, xsGCP, ysGCP, zsGCP)[0:2]
            imgVerbose = ulises.DisplayCRInImage(imgVerbose, cs, rs, options={'colors':[[0, 255, 255]], 'size':2})
            chs = np.arange(0, nc, 1);
            rhs = ulises.CDh2RDh(mainSet['horizonLine'], chs, options={})[0]
            imgVerbose = ulises.DisplayCRInImage(imgVerbose, chs, rhs, options={'colors':[[0, 255, 255]], 'size':1})
            cv2.imwrite(pathTMP + os.sep + fnFrame.replace('.', 'cal_check.'), imgVerbose)
    #
    return None
#
def PlanviewsFromImages(pathImages, pathPlanviews, z0, ppm, verbosePlot):
    #
    # obtain the planview domain from the cloud of points
    if not os.path.exists(pathPlanviews):
        print('*** folder {:} not found'.format(pathPlanviews)); exit()
    if not os.path.exists(pathPlanviews + os.sep + 'xy_planview.txt'):
        print('*** file xy_planview.txt not found in {:}'.format(pathPlanviews)); exit()
    rawData = np.asarray(ulises.ReadRectangleFromTxt(pathPlanviews + os.sep + 'xy_planview.txt', options={'c1':2, 'valueType':'float'}))
    xsCloud, ysCloud = rawData[:, 0], rawData[:, 1]
    angle, xUL, yUL, H, W = ulises.Cloud2Rectangle(xsCloud, ysCloud)
    dataPdfTxt = ulises.LoadDataPdfTxt(options={'xUpperLeft':xUL, 'yUpperLeft':yUL, 'angle':angle, 'xYLengthInC':W, 'xYLengthInR':H, 'ppm':ppm})
    csCloud, rsCloud = ulises.XY2CR(dataPdfTxt, xsCloud, ysCloud)[0:2] # only useful if verbosePlot
    #
    # write the planview domain
    fileout = open(pathPlanviews + os.sep + 'crxyz_planview.txt', 'w')
    for pos in range(4):
        fileout.write('{:6.0f} {:6.0f} {:8.2f} {:8.2f} {:8.2f}\t c, r, x, y and z\n'.format(dataPdfTxt['csC'][pos], dataPdfTxt['rsC'][pos], dataPdfTxt['xsC'][pos], dataPdfTxt['ysC'][pos], z0))
    fileout.close()
    #
    # obtain and write planviews
    fnsImages = sorted([item for item in os.listdir(pathImages) if '.' in item and item[item.rfind('.')+1:] in ['jpeg', 'JPEG', 'jpg', 'JPG', 'png', 'PNG']])
    for fnImage in fnsImages:
        #
        print('... planview of {:}'.format(fnImage), end='', flush=True)
        #
        # obtain pathPlw and check if already exists
        pathPlw = pathPlanviews + os.sep + fnImage.replace('.', 'plw.')
        if os.path.exists(pathPlw):
            print(' already exists'); continue
        #
        # load calibration and obtain and write planview
        pathCalTxt = pathImages + os.sep + fnImage[0:fnImage.rfind('.')] + 'cal.txt'
        if os.path.exists(pathCalTxt):
            # load calibration
            allVariables, nc, nr, errorT = ulises.ReadCalTxt(pathCalTxt)
            mainSet = ulises.AllVariables2MainSet(allVariables, nc, nr, options={})
            # obtain and write planview
            imgPlanview = ulises.CreatePlanview(ulises.PlanviewPrecomputations({'01':mainSet}, dataPdfTxt, z0), {'01':cv2.imread(pathImages + os.sep + fnImage)})
            cv2.imwrite(pathPlw, imgPlanview)
            print(' success')
            # manage verbosePlot
            if verbosePlot:
                pathTMP = pathPlanviews + os.sep + '..' + os.sep + 'TMP'
                ulises.MakeFolder(pathTMP)
                imgTMP = ulises.DisplayCRInImage(imgPlanview, csCloud, rsCloud, options={'colors':[[0, 255, 255]], 'size':10})
                cv2.imwrite(pathTMP + os.sep + fnImage.replace('.', 'plw_check.'), imgTMP)
                cs, rs = ulises.XYZ2CDRD(mainSet, xsCloud, ysCloud, z0)[0:2]
                img = cv2.imread(pathImages + os.sep + fnImage)
                imgTMP = ulises.DisplayCRInImage(img, cs, rs, options={'colors':[[0, 255, 255]], 'size':10})
                cv2.imwrite(pathTMP + os.sep + fnImage.replace('.', '_checkplw.'), imgTMP)
        else:
            print(' failed')
    #
    return None
#
def CheckGCPs(pathBasisCheck, errorCritical, model):
    #
    errorRANSAC, nOfRANSAC, aG = errorCritical, 100, 1.
    #
    # manage model
    model2SelectedVariablesKeys = {'parabolic':['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'sca'], 'quartic':['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'sca'], 'full':['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or']}
    if model not in model2SelectedVariablesKeys.keys():
        print('*** Invalid calibration model {:}'.format(model))
        print('*** Choose one of the following calibration models: {:}'.format(list(model2SelectedVariablesKeys.keys()))); exit()
    #
    # check GCPs
    fnsImages = sorted([item for item in os.listdir(pathBasisCheck) if '.' in item and item[item.rfind('.')+1:] in ['jpeg', 'JPEG', 'jpg', 'JPG', 'png', 'PNG']])
    for posFnImage, fnImage in enumerate(fnsImages):
        #
        print('... checking of {:}'.format(fnImage))
        #
        # load image information and dataBasic
        if posFnImage == 0:
            nr, nc = cv2.imread(pathBasisCheck + os.sep + fnImage).shape[0:2]
            dataBasic = ulises.LoadDataBasic0(options={'nc':nc, 'nr':nr, 'selectedVariablesKeys':model2SelectedVariablesKeys[model]})
        else:
            assert cv2.imread(pathBasisCheck + os.sep + fnImage).shape[0:2] == (nr, nc)
        #
        # load GCPs
        pathCdgTxt = pathBasisCheck + os.sep + fnImage[0:fnImage.rfind('.')] + 'cdg.txt'
        cs, rs, xs, ys, zs = ulises.ReadCdgTxt(pathCdgTxt, options={'readCodes':False, 'readOnlyGood':True})[0:5]
        #
        # load dataForCal and obtain good GCPs via RANSAC
        dataForCal = {'nc':nc, 'nr':nr, 'cs':cs, 'rs':rs, 'xs':xs, 'ys':ys, 'zs':zs, 'aG':aG} # IMP* only GCPs
        subsetVariablesKeys, subCsetVariablesDictionary = model2SelectedVariablesKeys[model], {}
        possGood = ulises.ObtainGoodGCPsRANSAC(dataBasic, dataForCal, subsetVariablesKeys, subCsetVariablesDictionary, options={'errorRANSAC':errorRANSAC, 'nOfRANSAC':nOfRANSAC, 'verbose':True})
        #
        # inform
        if len(possGood) < len(cs):
            print('... re-run or consider to ignore the following GCPs')
            for pos in [item for item in range(len(cs)) if item not in possGood]:
                c, r, x, y, z = [item[pos] for item in [cs, rs, xs, ys, zs]]
                print('... c = {:8.2f}, r = {:8.2f}, x = {:8.2f}, y = {:8.2f}, z = {:8.2f}'.format(c, r, x, y, z))
        else:
            print('... all the GCPs for {:} are OK'.format(fnImage))
    #
    return None
#
def TimexAngSigma(pathImages):
    #
    # obtain and write mean and sigma images
    pathsImages = [pathImages + os.sep + item for item in os.listdir(pathImages) if item not in ['mean.png', 'sigma.png'] and '.' in item and item[item.rfind('.')+1:] in ['jpeg', 'JPEG', 'jpg', 'JPG', 'png', 'PNG']]
    imgMean, imgSigma = ulises.MeanAndSigmaOfImages(pathsImages)
    cv2.imwrite(pathImages + os.sep + 'mean.png', imgMean)
    cv2.imwrite(pathImages + os.sep + 'sigma.png', imgSigma)
    #
    return None
#
def TimestackFromImages(pathImages, pathFolderTimestack, ppm, includeNotCalibrated, verbosePlot):
    #
    # obtain pathTimestack and check if already exists
    pathTimestack = pathFolderTimestack + os.sep + 'timestack.png'
    if os.path.exists(pathTimestack):
        print('... timestack already exists'); return None
    #
    # obtain the timestack points
    if not os.path.exists(pathFolderTimestack):
        print('*** folder {:} not found'.format(pathFolderTimestack)); exit()
    if not os.path.exists(pathFolderTimestack + os.sep + 'xyz_timestack.txt'):
        print('*** file xyz_timestack.txt not found in {:}'.format(pathFolderTimestack)); exit()
    rawData = np.asarray(ulises.ReadRectangleFromTxt(pathFolderTimestack + os.sep + 'xyz_timestack.txt', options={'c1':3, 'valueType':'float'}))
    xs, ys, zs = rawData[:, 0], rawData[:, 1], rawData[:, 2]
    #
    # obtain ppm and interpolated xs, ys and zs
    lengthsOfInitialSegments = ulises.LengthsOfSegmentsOfAPolyline({'xs':xs, 'ys':ys})
    length = np.sum(lengthsOfInitialSegments)
    ppm = length / (int(length * ppm) + 1) # length of each final segment
    nOfDesiredSegments = int(np.round(length / ppm))
    cumulativeLengthsOfInitialSegments = np.array([0.] + list(np.cumsum(lengthsOfInitialSegments)))
    cumulativeLengthsOfFinalSegments = length / nOfDesiredSegments * np.arange(0., nOfDesiredSegments + 1)
    xsI, ysI, zsI = [], [], []
    xsI.append(xs[0]); ysI.append(ys[0]); zsI.append(zs[0])
    for cumulativeLengthOfFinalSegments in cumulativeLengthsOfFinalSegments[1:-1]: # each of the middel points
        posInitialSegment = np.where(cumulativeLengthOfFinalSegments >= cumulativeLengthsOfInitialSegments)[0][-1]
        auxiliaryLength = cumulativeLengthOfFinalSegments - cumulativeLengthsOfInitialSegments[posInitialSegment]
        #print('{:5.2f} {:5.0f} {:5.2f} {:5.2f}'.format(cumulativeLengthOfFinalSegments, posInitialSegment, lengthsOfInitialSegments[posInitialSegment], auxiliaryLength))
        #lengthOfInitialSegment = lengthsOfInitialSegments[posInitialSegment]
        xsI.append(xs[posInitialSegment] + auxiliaryLength * (xs[posInitialSegment+1] - xs[posInitialSegment]) / lengthsOfInitialSegments[posInitialSegment])
        ysI.append(ys[posInitialSegment] + auxiliaryLength * (ys[posInitialSegment+1] - ys[posInitialSegment]) / lengthsOfInitialSegments[posInitialSegment])
        zsI.append(zs[posInitialSegment] + auxiliaryLength * (zs[posInitialSegment+1] - zs[posInitialSegment]) / lengthsOfInitialSegments[posInitialSegment])
    xsI.append(xs[-1]); ysI.append(ys[-1]); zsI.append(zs[-1])
    xsI, ysI, zsI = [np.asarray(item) for item in [xsI, ysI, zsI]]
    #
    # write timestack xyz interpolated coordinates
    fileout = open(pathFolderTimestack + os.sep + 'cxyz_timestack.txt', 'w')
    for pos in range(len(xsI)):
        fileout.write('{:6.0f} {:8.2f} {:8.2f} {:8.2f}\t c, x, y and z\n'.format(pos, xsI[pos], ysI[pos], zsI[pos]))
    fileout.close()
    #
    # obtain useful images
    fnsImages = sorted([item for item in os.listdir(pathImages) if '.' in item and item[item.rfind('.')+1:] in ['jpeg', 'JPEG', 'jpg', 'JPG', 'png', 'PNG']])    
    if not includeNotCalibrated:
        fnsImages = sorted([item for item in fnsImages if os.path.exists(pathImages + os.sep + item[0:item.rfind('.')] + 'cal.txt')])
    imgTimestack = np.zeros((len(fnsImages), len(xsI), 3), np.uint8)
    #
    # write timestack "times"
    fileout = open(pathFolderTimestack + os.sep + 'rt_timestack.txt', 'w')
    for pos in range(len(fnsImages)):
        fileout.write('{:6.0f} {:>30}\t r and filename\n'.format(pos, fnsImages[pos]))
    fileout.close()
    #
    # obtain and write timestack
    for posFnImage, fnImage in enumerate(fnsImages):
        #
        # load image and calibration
        img = cv2.imread(pathImages + os.sep + fnImage)
        pathCalTxt = pathImages + os.sep + fnImage[0:fnImage.rfind('.')] + 'cal.txt'
        #pathCalTxt = pathImages + os.sep + fnsImages[0][0:fnsImages[0].rfind('.')] + 'cal.txt'
        #
        # update timestack
        if os.path.exists(pathCalTxt):
            allVariables, nc, nr, errorT = ulises.ReadCalTxt(pathCalTxt)
            mainSet = ulises.AllVariables2MainSet(allVariables, nc, nr, options={}) #!
            # update timestack
            cs, rs, possGood = ulises.XYZ2CDRD(mainSet, xsI, ysI, zsI, options={'imgMargins':{'c0':2, 'c1':2, 'r0':2, 'r1':2, 'isComplete':True}, 'returnGoodPositions':True})
            csIA, rsIA, wsA = ulises.CR2CRIntegerAroundAndWeights(cs, rs)
            for pos in possGood:
                for posBis in range(4):
                    imgTimestack[posFnImage, pos, :] = imgTimestack[posFnImage, pos, :] + wsA[pos, posBis] * img[int(rsIA[pos, posBis]), int(csIA[pos, posBis]), :]
            # manage verbosePlot
            if verbosePlot:
                pathTMP = pathFolderTimestack + os.sep + '..' + os.sep + 'TMP'
                ulises.MakeFolder(pathTMP)
                imgTMP = ulises.DisplayCRInImage(img, cs[possGood], rs[possGood], options={'colors':[[0, 255, 255]], 'size':2})
                cv2.imwrite(pathTMP + os.sep + fnImage.replace('.', '_checktimestack.'), imgTMP)
        else:
            continue
    #
    # write timestack
    imgTimestack = imgTimestack.astype(np.uint8)
    cv2.imwrite(pathFolderTimestack + os.sep + 'timestack.png', imgTimestack)
    #
    return None
#
