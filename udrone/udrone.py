'''
Created on 2021 by Gonzalo Simarro and Daniel Calvete
'''
#
import ulises_udrone as ulises
#
import os
import cv2
import datetime
import numpy as np
#
#
#
def Video2Frames(pathFolderVideo , pathFolderSnaps, fps): # OK
    #
    extensions = ['mp4', 'avi']
    pathVideo = [pathFolderVideo + os.sep + item for item in os.listdir(pathFolderVideo) if any([item.endswith(itemBis) for itemBis in extensions])][0]
    fnVideo = pathVideo[pathVideo.rfind(os.sep)+1:]
    print('... frame extraction from video {:}'.format(fnVideo))
    #
    # load video and obtain fps
    fpsOfVideo = cv2.VideoCapture(pathVideo).get(cv2.CAP_PROP_FPS)
    if fps > fpsOfVideo:
        print('*** Frames per second of the video ({:2.1f}) small than FPS={:2.1f}'.format(fpsOfVideo, fps))
        print('*** Select smaller FPS ***')
        exit()
    if fps == 0:
        fps = fpsOfVideo
    else :
        fps = min([fpsOfVideo / int(fpsOfVideo / fps), fpsOfVideo])
    ulises.Video2Snaps(pathVideo, pathFolderSnaps, fps)
    return None
#
def calibrationOfBasisImages(pathBasis, model): # OK
    #
    fnsImgsBas = sorted([item for item in os.listdir(pathBasis) if 'TMP' not in item and item[-3:] == 'png'])
    aG, aH = 1., 1.
    #
    model2SelectedVariablesKeys = {'parabolic':['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'sca'], 'quartic':['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'sca'], 'full':['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or']}
    if model not in model2SelectedVariablesKeys:
        print('*** No case for calibration model {:}'.format(model))
        print('*** Choose one of the following calibration models: {:}'.format(list(model2SelectedVariablesKeys.keys())))
        exit()
    #
    dataBasic = ulises.LoadDataBasic0(options={'selectedVariablesKeys':model2SelectedVariablesKeys[model]})
    #
    fnsImgsBas = sorted([item for item in os.listdir(pathBasis) if 'TMP' not in item and item[-3:] == 'png'])
    for posFnImgBas, fnImgBas in enumerate(fnsImgsBas):
        print('... calibration of {:} '.format(fnImgBas),end='', flush=True)
        #
        # load images size
        img = cv2.imread(pathBasis + os.sep + fnImgBas)
        if posFnImgBas == 0:
            nr, nc = img.shape[0:2]
        else:
            assert img.shape[0:2] == (nr, nc)
        #
        # load cdg (GCPs)
        pathCdgTxt = pathBasis + os.sep + fnImgBas[0:fnImgBas.rfind('.')] + 'cdg.txt'
        cs, rs, xs, ys, zs = ulises.ReadCdgTxt(pathCdgTxt)
        #
        # load cdh (horizon)
        pathCdhTxt = pathCdgTxt.replace('cdg', 'cdh')
        if os.path.exists(pathCdhTxt):
            chs, rhs = ulises.ReadCdhTxt(pathCdhTxt)
        #
        # load data for calibration
        dataForCal = {}
        dataForCal['nc'], dataForCal['nr'] = nc, nr
        dataForCal['cs'], dataForCal['rs'] = cs, rs
        dataForCal['xs'], dataForCal['ys'], dataForCal['zs'] = xs, ys, zs
        dataForCal['aG'] = aG
        if os.path.exists(pathCdhTxt):
            dataForCal['chs'], dataForCal['rhs'] = chs, rhs
            dataForCal['aH'] = aH
        pathsCalTxts = [pathBasis + os.sep + item for item in os.listdir(pathBasis) if 'cal' in item and item.endswith('txt')]
        if len(pathsCalTxts) > 0:
            dataForCal['mainSetSeeds'] = []
            for pathCalTxt in pathsCalTxts:
                allVariables, nc, nr = ulises.ReadCalTxt(pathCalTxt)[0:3]
                dataForCal['mainSetSeeds'].append(ulises.AllVariables2MainSet(allVariables, nc, nr, options={}))
        #
        # obtain calibration
        mainSet, errorT = ulises.NonlinearManualCalibration(dataBasic, dataForCal, model2SelectedVariablesKeys[model], {}, options={})
        if mainSet is not None and errorT < 10.:
            pathCal0Txt = pathCdgTxt.replace('cdg', 'cal0')
            ulises.WriteCalTxt(pathCal0Txt, mainSet['allVariables'], mainSet['nc'], mainSet['nr'], errorT)
            print('success')
        else:
            print('failed')
            print('****** re-run and, if it keeps failing, check quality of the GCP or try another calibration model ******')
    return None
#
def calibrationOfBasisImagesConstantIntrinsic(pathBasis, model): # OK
    #
    fnsImgsBas = sorted([item for item in os.listdir(pathBasis) if 'TMP' not in item and item[-3:] == 'png'])
    aG, aH = 1., 1.
    #
    model2SelectedVariablesKeys = {'parabolic':['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'sca'], 'quartic':['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'sca'], 'full':['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or']}
    if model not in model2SelectedVariablesKeys:
        print('*** No case for calibration model {:}'.format(model))
        print('*** Choose one of the following calibration models: {:}'.format(list(model2SelectedVariablesKeys.keys())))
        exit()
    #
    dataBasic = ulises.LoadDataBasic0(options={'selectedVariablesKeys':model2SelectedVariablesKeys[model]})
    #
    subsetVariablesKeys = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta'] # variable
    subCsetVariablesKeys = [item for item in model2SelectedVariablesKeys[model] if item not in subsetVariablesKeys] # unique
    css, rss, xss, yss, zss, chss, rhss, subsetVariabless, subCsetVariabless = [[] for item in range(9)]
    for posFnImgBas, fnImgBas in enumerate(fnsImgsBas):
        #
        # load images size
        img = cv2.imread(pathBasis + os.sep + fnImgBas)
        if posFnImgBas == 0:
            nr, nc = img.shape[0:2]
        else:
            assert img.shape[0:2] == (nr, nc)
        #
        # load cdg (GCPs)
        pathCdgTxt = pathBasis + os.sep + fnImgBas[0:fnImgBas.rfind('.')] + 'cdg.txt'
        cs, rs, xs, ys, zs = ulises.ReadCdgTxt(pathCdgTxt)
        css.append(cs); rss.append(rs); xss.append(xs); yss.append(ys); zss.append(zs)
        #
        # load cdh (horizon)
        pathCdhTxt = pathCdgTxt.replace('cdg', 'cdh')
        if os.path.exists(pathCdhTxt):
            chs, rhs = ulises.ReadCdhTxt(pathCdhTxt)
        else:
            chs, rhs = None, None # IMP*
        chss.append(chs); rhss.append(rhs)
        #
        pathCal0Txt = pathCdgTxt.replace('cdg', 'cal0')
        allVariables, nc, nr = ulises.ReadCalTxt(pathCal0Txt)[0:3]
        subsetVariabless.append(ulises.AllVariables2SubsetVariables(dataBasic, allVariables, subsetVariablesKeys, options={})) 
        subCsetVariabless.append(ulises.AllVariables2SubsetVariables(dataBasic, allVariables, subCsetVariablesKeys, options={}))
    #
    #print(subsetVariabless[0])
    #print(subsetVariablesKeys)
    #print(subCsetVariabless[0])
    #print(subCsetVariablesKeys)
    ncs, nrs = np.asarray([nc for item in range(len(fnsImgsBas))]), np.asarray([nr for item in range(len(fnsImgsBas))])
    mainSets, errorTs = ulises.NonlinearManualCalibrationForcingUniqueSubCset(dataBasic, ncs, nrs, css, rss, xss, yss, zss, chss, rhss, subsetVariabless, subsetVariablesKeys, subCsetVariabless, subCsetVariablesKeys, options={'aG':aG, 'aH':aH})
    if mainSets is not None:   
        for posFnImgBas, fnImgBas in enumerate(fnsImgsBas):
            pathCalTxt = pathBasis + os.sep + fnImgBas[0:fnImgBas.rfind('.')] + 'cal.txt'
            ulises.WriteCalTxt(pathCalTxt, mainSets[posFnImgBas]['allVariables'], mainSets[posFnImgBas]['nc'], mainSets[posFnImgBas]['nr'], errorTs[posFnImgBas])
        print('... intrinsic success')
    else:
        print('... intrinsic failed')
    return None
#
def autoCalibrationOfFrames(pathBasis, pathSnaps, verboseTMP): # OK
    #
    pathCalib = pathSnaps
    ulises.MakeFolder(pathSnaps)
    if  verboseTMP:
        pathTMP = pathSnaps + os.sep + '..' + os.sep + 'TMP'
        ulises.MakeFolder(pathTMP)
    #
    dataBasic = ulises.LoadDataBasic0()
    nOfFeatures, aG = 20000, 1.
    #
    # load basis information
    pathCalTxt = [pathBasis + os.sep + item for item in os.listdir(pathBasis) if 'cal' in item and 'cal0' not in item and item.endswith('txt')][0]
    allVariablesTMP, nc, nr = ulises.ReadCalTxt(pathCalTxt)[0:3]
    selectedVariablesKeys = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or']
    subsetVariablesKeys, subCsetVariablesKeys = ['xc', 'yc', 'zc', 'ph', 'sg', 'ta'], ['k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or']
    subCsetVariables = ulises.AllVariables2SubsetVariables(dataBasic, allVariablesTMP, subCsetVariablesKeys, options={}) # intrinsic here
    subCsetVariablesDictionary = ulises.Array2Dictionary(subCsetVariablesKeys, subCsetVariables)
    window = np.int(0.025*np.sqrt(nc*nr)) 
    #
    # load basis information
    print('... loading basis information')
    fnsBas = sorted([item[0:-4] for item in os.listdir(pathBasis) if 'TMP' not in item and item[-3:] == 'png'])
    imgsBas, kpssBas, dessBas, cssBas, rssBas, xssBas, yssBas, zssBas, mainSetsBas, cUssBas, rUssBas = [{} for item in range(11)]
    for posFnBas, fnBas in enumerate(fnsBas):
        print('... loading basis information: {:}'.format(fnBas))
        img = cv2.imread(pathBasis + os.sep + fnBas + '.png')
        assert img.shape[0:2] == (nr, nc)
        kps, des, ctrl = ulises.ORBKeypoints(img, {'nOfFeatures':nOfFeatures})[2:]
        if not ctrl:
            continue
        imgsBas[fnBas], kpssBas[fnBas], dessBas[fnBas] = img, kps, des
        #
        pathCdgTxt = pathBasis + os.sep + fnBas + 'cdg.txt'
        cssBas[fnBas], rssBas[fnBas], xssBas[fnBas], yssBas[fnBas], zssBas[fnBas] = ulises.ReadCdgTxt(pathCdgTxt)
        #
        pathCalTxt = pathBasis + os.sep + fnBas + 'cal.txt'
        allVariables, ncR, nrR = ulises.ReadCalTxt(pathCalTxt)[0:3]
        assert ncR == nc and nrR == nr
        mainSetsBas[fnBas] = ulises.AllVariables2MainSet(allVariables, nc, nr, options={}) #!
        #
        cUssBas[fnBas], rUssBas[fnBas] = ulises.CDRD2CURU(mainSetsBas[fnBas], cssBas[fnBas], rssBas[fnBas]) # only the intrinsic, constant, is used
    #    
    for fnSnap in sorted([item[0:-4] for item in os.listdir(pathSnaps) if item[-3:] == 'png']):
        #
        print('... calibration of {:} '.format(fnSnap),end='', flush=True)
        if os.path.exists(pathCalib + os.sep + fnSnap + 'cal.txt'):
            print('already calibrated') 
            continue
        #
        xsGCP, ysGCP, zsGCP, csGCP, rsGCP = [[] for item in range(5)] # image to calibrate
        #
        # compute keypoints for the image to calibrate
        img = cv2.imread(pathSnaps + os.sep + fnSnap + '.png')
        kps, des, ctrl = ulises.ORBKeypoints(img, {'nOfFeatures':nOfFeatures})[2:]
        if not ctrl:
            print('not calibratable')
            print('****** lack of points keypoints ******')
            continue
        #
        # find GCPs for the image to calibrate
        for fnBas in fnsBas:
            try:
                cs, rs, csBas, rsBas, ers = ulises.ORBMatches(kps, des, kpssBas[fnBas], dessBas[fnBas], {'erMaximum':20., 'nOfStd':1.})
                poss06 = ulises.SelectPixelsInGrid(6, nc, nr, cs, rs, ers)[0]
                assert len(poss06) >= 6
                cs, rs, csBas, rsBas, ers = [item[poss06] for item in [cs, rs, csBas, rsBas, ers]]
            except:
                continue
            #
            cUs, rUs = ulises.CDRD2CURU(mainSetsBas[fnBas], cs, rs) # only the intrinsic, constant, is used
            cUsBas, rUsBas = ulises.CDRD2CURU(mainSetsBas[fnBas], csBas, rsBas) # only the intrinsic, constant, is used
            #
            # find homography from cUsBas(1) to cUs(2)
            parametersRANSAC = {'p':0.99999, 'e':0.8, 's':4, 'errorC':2.0}
            Ha, possGood = ulises.FindHomographyHa01ViaRANSAC(cUsBas, rUsBas, cUs, rUs, parametersRANSAC)
            #
            # approximate position of the GCPs
            assert Ha[2, 2] == 1.
            den =  Ha[2, 0] * cUssBas[fnBas] + Ha[2, 1] * rUssBas[fnBas] + 1.
            cUsApprox = (Ha[0, 0] * cUssBas[fnBas] + Ha[0, 1] * rUssBas[fnBas] + Ha[0, 2]) / den
            rUsApprox = (Ha[1, 0] * cUssBas[fnBas] + Ha[1, 1] * rUssBas[fnBas] + Ha[1, 2]) / den
            csApprox, rsApprox = ulises.CURU2CDRD(mainSetsBas[fnBas], cUsApprox, rUsApprox) # only the intrinsic, constant, is used
            #
            # refined position of the GCPs
            for pos in range(len(csApprox)):
                # image bas
                c0, r0 = int(cssBas[fnBas][pos]), int(rssBas[fnBas][pos])
                if not (c0 > window+1 and nc-c0 > window+1 and r0 > window+1 and nr-r0 > window+1):
                    continue
                img0 = imgsBas[fnBas][r0-window:r0+window, c0-window:c0+window, :]
                # image to calibrate
                c1, r1 = int(csApprox[pos]), int(rsApprox[pos])
                if not (c1 > window+1 and nc-c1 > window+1 and r1 > window+1 and nr-r1 > window+1):
                    continue
                img1 = img[r1-window:r1+window, c1-window:c1+window, :]
                nc0, nr0, kps0, des0, ctrl0 = ulises.ORBKeypoints(img0, options={'nOfFeatures':1000})
                nc1, nr1, kps1, des1, ctrl1 = ulises.ORBKeypoints(img1, options={'nOfFeatures':1000})
                if not (ctrl0 and ctrl1):
                    continue
                cs0, rs0, cs1, rs1, ers = ulises.ORBMatches(kps0, des0, kps1, des1, options={'erMaximum':50., 'nOfStd':1.0})
                if len(cs0) < 5:
                    continue
                dc, dr = np.mean(cs1-cs0), np.mean(rs1-rs0)
                #
                xsGCP.append(xssBas[fnBas][pos]) 
                ysGCP.append(yssBas[fnBas][pos])
                zsGCP.append(zssBas[fnBas][pos])
                csGCP.append(c1+dc)
                rsGCP.append(r1+dr)
        #
        xsGCP, ysGCP, zsGCP, csGCP, rsGCP = [np.asarray(item) for item in [xsGCP, ysGCP, zsGCP, csGCP, rsGCP]]
        #
        # obtain auto calibration
        dataForCal = {}
        dataForCal['nc'], dataForCal['nr'] = nc, nr
        dataForCal['cs'], dataForCal['rs'] = csGCP, rsGCP
        dataForCal['xs'], dataForCal['ys'], dataForCal['zs'] = xsGCP, ysGCP, zsGCP
        dataForCal['aG'] = aG
        dataForCal['mainSetSeeds'] = []
        for fnBas in fnsBas:
            dataForCal['mainSetSeeds'].append(mainSetsBas[fnBas])
        mainSet, errorT = ulises.NonlinearManualCalibration(dataBasic, dataForCal, subsetVariablesKeys, subCsetVariablesDictionary, options={})
        #
        if mainSet is not None and verboseTMP: 
            imgVerbose = ulises.DisplayCRInImage(img, csGCP, rsGCP, options={'colors':[[0, 0, 0]], 'size':4})
            cds, rds = ulises.XYZ2CDRD(mainSet, xsGCP, ysGCP, zsGCP)[0:2]
            imgVerbose = ulises.DisplayCRInImage(imgVerbose, cds, rds, options={'colors':[[0, 255, 255]], 'size':2})
            cs = np.arange(0, nc, 1);
            rs = ulises.CDh2RDh(mainSet['horizonLine'], cs, options={})[0] #!
            imgVerbose = ulises.DisplayCRInImage(imgVerbose, cs, rs, options={'colors':[[0, 255, 255]], 'size':1})
            cv2.imwrite(pathTMP + os.sep + fnSnap + '_check.png', imgVerbose)
        #
        # write calibration
        if mainSet is not None and errorT < 10:
            print('success')
            pathCalTxt = pathCalib + os.sep + fnSnap + 'cal.txt'
            ulises.WriteCalTxt(pathCalTxt, mainSet['allVariables'], mainSet['nc'], mainSet['nr'], errorT)
        else:
            print('failed')
            print('****** check quality of the GCP or try another calibration model ******')
    return None
#
def planviewsFromSnaps(pathMain, pathFrames, pathPlanviews, z0, ppm, verboseTMP):
    #
    pathCalib = pathFrames
    ulises.MakeFolder(pathPlanviews)
    if verboseTMP:
        pathTMP = pathPlanviews + os.sep + '..' + os.sep + 'TMP'
        ulises.MakeFolder(pathTMP)
    #
    # load cloud of points
    rawData = np.asarray(ulises.ReadRectangleFromTxt(pathMain + os.sep + 'xy_planview.txt', options={'c1':2, 'valueType':'float'}))
    x_cloud, y_cloud = rawData[:, 0], rawData[:, 1]
    angle, xUL, yUL, H, W = ulises.Cloud2Rectangle(x_cloud, y_cloud) #1
    dataPdfTxt = ulises.LoadDataPdfTxt(options={'xUpperLeft':xUL, 'yUpperLeft':yUL, 'angle':angle, 'xYLengthInC':W, 'xYLengthInR':H, 'ppm':ppm})
    # pixels of the cloud of points
    c_cloud, r_cloud, _ = ulises.XY2CR(dataPdfTxt, x_cloud, y_cloud)
    #
    # write pixel coordinates
    fileout = open(pathPlanviews + os.sep + 'crxy_Planviews.txt', 'w')
    C_min, C_max, R_min, R_max= min(dataPdfTxt['cs']), max(dataPdfTxt['cs']), min(dataPdfTxt['rs']), max(dataPdfTxt['rs'])
    C_write, R_write = C_min, R_min
    fileout.write('{:6.0f} {:6.0f} {:8.2f} {:8.2f} {:8.2f}\n'.format(C_write, R_write, ulises.CR2XY(dataPdfTxt, C_write, R_write)[0], ulises.CR2XY(dataPdfTxt, C_write, R_write)[1], z0))
    C_write, R_write = C_max, R_min
    fileout.write('{:6.0f} {:6.0f} {:8.2f} {:8.2f} {:8.2f}\n'.format(C_write, R_write, ulises.CR2XY(dataPdfTxt, C_write, R_write)[0], ulises.CR2XY(dataPdfTxt, C_write, R_write)[1], z0))
    C_write, R_write = C_min, R_max
    fileout.write('{:6.0f} {:6.0f} {:8.2f} {:8.2f} {:8.2f}\n'.format(C_write, R_write, ulises.CR2XY(dataPdfTxt, C_write, R_write)[0], ulises.CR2XY(dataPdfTxt, C_write, R_write)[1], z0))
    C_write, R_write = C_max, R_max
    fileout.write('{:6.0f} {:6.0f} {:8.2f} {:8.2f} {:8.2f}\n'.format(C_write, R_write, ulises.CR2XY(dataPdfTxt, C_write, R_write)[0], ulises.CR2XY(dataPdfTxt, C_write, R_write)[1], z0))
    fileout.close()
    #
    # write planview images
    dataBasic = ulises.LoadDataBasic0()
    for fnFrame in sorted([item for item in os.listdir(pathFrames) if item[-3:] == 'png']):
        print('... planview {:} '.format(fnFrame[:-4]),end='', flush=True)
        if os.path.exists(pathFrames + os.sep + fnFrame.replace('.png', 'cal.txt')) :
            pathCalTxt = pathFrames + os.sep + fnFrame.replace('.png', 'cal.txt')
            allVariables, nc, nr, errorT = ulises.ReadCalTxt(pathCalTxt)
            mainSet = ulises.AllVariables2MainSet(allVariables, nc, nr, options={})
            imgPlanview = ulises.CreatePlanview(ulises.PlanviewPrecomputations({'01':mainSet}, dataPdfTxt, z0), {'01':cv2.imread(pathFrames + os.sep + fnFrame)})
            cv2.imwrite(pathPlanviews + os.sep + fnFrame.replace('.png', 'plw.png'), imgPlanview)
            print('success')
            if verboseTMP:
                imgVerbose = ulises.DisplayCRInImage(imgPlanview, c_cloud, r_cloud, options={'colors':[[0, 255, 255]], 'size':10})
                cv2.imwrite(pathTMP + os.sep + fnFrame.replace('.png', 'plw_check.png'), imgVerbose)
                cs, rs = ulises.XYZ2CDRD(mainSet, x_cloud, y_cloud, z0)[0:2]
                img = cv2.imread(pathFrames + os.sep + fnFrame)
                imgVerbose = ulises.DisplayCRInImage(img, cs, rs, options={'colors':[[0, 255, 255]], 'size':10})
                cv2.imwrite(pathTMP + os.sep + fnFrame.replace('.png', '_checkplw.png'), imgVerbose)
        else:
            print('failed')
    return None
#
def checkGCPs(pathBasisCheck):
#
    pathBasis = pathBasisCheck
    errorRANSAC, nOfRANSAC, model, aG, aH = 5., 100, 'parabolic', 1., 1.
    #
    model2SelectedVariablesKeys = {'parabolic':['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'sca'], 'quartic':['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'sca'], 'full':['xc', 'yc', 'zc', 'ph', 'sg', 'ta', 'k1a', 'k2a', 'p1a', 'p2a', 'sca', 'sra', 'oc', 'or']}
    dataBasic = ulises.LoadDataBasic0(options={'selectedVariablesKeys':model2SelectedVariablesKeys[model]}) #!
    #
    # analyze images
    fnsImgsBas = sorted([item for item in os.listdir(pathBasis) if item[-3:] == 'png'])
    for posFnImgBas, fnImgBas in enumerate(fnsImgsBas):
        #
        print('*************** analysis of {:} ***************'.format(fnImgBas[0:fnImgBas.rfind('.')]))
        #
        # load images size
        img = cv2.imread(pathBasis + os.sep + fnImgBas)
        if posFnImgBas == 0:
            nr, nc = img.shape[0:2]
        else:
            assert img.shape[0:2] == (nr, nc)
        #
        # load cdg (GCPs)
        pathCdgTxt = pathBasis + os.sep + fnImgBas[0:fnImgBas.rfind('.')] + 'cdg.txt'
        cs, rs, xs, ys, zs = ulises.ReadCdgTxt(pathCdgTxt)
        #
        # load data for calibration
        dataForCal = {}
        dataForCal['nc'], dataForCal['nr'] = nc, nr
        dataForCal['cs'], dataForCal['rs'] = cs, rs
        dataForCal['xs'], dataForCal['ys'], dataForCal['zs'] = xs, ys, zs
        dataForCal['aG'] = aG
        #
        # obtain good GCPs via RANSAC (no seeds)
        possGood = ulises.ObtainGoodGCPsRANSAC(dataBasic, dataForCal, model2SelectedVariablesKeys[model], {}, options={'errorRANSAC':errorRANSAC, 'nOfRANSAC':nOfRANSAC, 'verbose':True})
        print ('')
        if len(possGood) < len(dataForCal['cs']):
            print('... consider to ignore the following GCPs')
            for pos in [item for item in range(len(dataForCal['cs'])) if item not in possGood]:
                c, r, x, y, z = [dataForCal[item][pos] for item in ['cs', 'rs', 'xs', 'ys', 'zs']]
                print('... line {:2}: c={:8.2f} r={:8.2f} x={:8.2f} y={:8.2f} z={:8.2f}'.format(pos+1, c, r, x, y, z))
        else:
            print('... the GCPs are OK')
    return None
