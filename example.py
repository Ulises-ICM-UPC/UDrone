'''
Created on 2021 by Gonzalo Simarro and Daniel Calvete
'''
import sys
sys.path.insert(0, 'udrone')
# own modules
import udrone
import os

pathMain = 'example'
pathFolderVideo = pathMain
pathFolderFrames = pathMain + os.sep + 'frames'

FPS = 2.00

print('Video extraction')
udrone.Video2Frames(pathFolderVideo, pathFolderFrames, FPS)

pathFolderBasis = pathMain + os.sep + 'basis'
calibrationModel = 'parabolic'
print('Intrinsic calibration')
print('Calibration of the basis')
udrone.calibrationOfBasisImages(pathFolderBasis, calibrationModel)
print('Optimal intrinsic parameters of the camera')
udrone.calibrationOfBasisImagesConstantIntrinsic(pathFolderBasis, calibrationModel)

verbosePlot = True
print('Calibration of the frames')
udrone.autoCalibrationOfFrames(pathFolderBasis, pathFolderFrames, verbosePlot)

pathFolderPlanv = pathMain + os.sep + 'planviews'
z0 = 1.5
ppm = 1.0
verbosePlanviews = True
print('Generation of planviews')
udrone.planviewsFromSnaps(pathMain, pathFolderFrames, pathFolderPlanv, z0, ppm, verbosePlanviews)

pathBasisCheck = pathMain + os.sep + 'basis_check'
print('Check GCP for basis calibration')
udrone.checkGCPs(pathBasisCheck)
