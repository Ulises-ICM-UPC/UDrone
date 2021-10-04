'''
Created on 2021 by Gonzalo Simarro and Daniel Calvete
'''
# Import modules
import sys
sys.path.insert(0, 'udrone')
import udrone
import os

pathMain = 'example' # USER DEFINED (main folder)

'''
Video extraction
'''
pathFolderVideo = pathMain # USER DEFINED (folder where the video is located)
pathFolderFrames = pathMain + os.sep + 'frames' # USER DEFINED (folder where the frames extractes fron the video wwil be placed)
FPS = 2.00 # USER DEFINED (extraction rate of the frames )
print('Video extraction')
udrone.Video2Frames(pathFolderVideo, pathFolderFrames, FPS)

'''
Intrinsic calibration
'''
pathFolderBasis = pathMain + os.sep + 'basis' # USER DEFINED (path where files for calibrating the basis are located)
calibrationModel = 'parabolic' # USER DEFINED (intrinsic camera calibration model [parabolic,quartic,full])
print('Intrinsic calibration')
print('Calibration of the basis')
udrone.calibrationOfBasisImages(pathFolderBasis, calibrationModel)
print('Optimal intrinsic parameters of the camera')
udrone.calibrationOfBasisImagesConstantIntrinsic(pathFolderBasis, calibrationModel)

'''
Automatic calibration
'''
verbosePlot = True # USER DEFINED
print('Calibration of the frames')
udrone.autoCalibrationOfFrames(pathFolderBasis, pathFolderFrames, verbosePlot)

'''
Planviews
'''
z0 = 1.5 # USER DEFINED (height at which the projection is made)
ppm = 1.0 # USER DEFINED (resolution of the planviews, pixels-per-meter)
verbosePlanviews = True # USER DEFINED 
pathFolderPlanv = pathMain + os.sep + 'planviews' # USER DEFINED (path where the planview images will be located)
print('Generation of planviews')
udrone.planviewsFromSnaps(pathMain, pathFolderFrames, pathFolderPlanv, z0, ppm, verbosePlanviews)

'''
GCP check
'''
pathBasisCheck = pathMain + os.sep + 'basis_check' # USER DEFINED (path where GCP-files to be checked are located)
print('Check GCP for basis calibration')
udrone.checkGCPs(pathBasisCheck)
