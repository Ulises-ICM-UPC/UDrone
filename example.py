#'''
# Created on 2021 by Gonzalo Simarro, Daniel Calvete and Paola Souto
#'''
#
import sys
import os
#
sys.path.insert(0, 'udrone')
import udrone as udrone
#
pathFolderMain = 'example' # USER DEFINED example TMP_Cali_201908161521
assert os.path.exists(pathFolderMain)
#
#''' --------------------------------------------------------------------------
# Extraction of the video
#''' --------------------------------------------------------------------------
#
pathFolderVideo = pathFolderMain # USER DEFINED (folder where the video is located)
pathFolderFrames = pathFolderMain + os.sep + 'frames' # USER DEFINED (folder where the frames extractes fron the video wwil be placed)
FPS = 2.00 # USER DEFINED (extraction rate of the frames )
#
print('Video extraction')
udrone.Video2Frames(pathFolderVideo, pathFolderFrames, FPS)
#
#''' --------------------------------------------------------------------------
# Calibration of the basis
#''' --------------------------------------------------------------------------
#
pathFolderBasis = pathFolderMain + os.sep + 'basis' # USER DEFINED
eCritical, calibrationModel = 5., 'parabolic' # USER DEFINED (eCritical is in pixels, calibrationModel = 'parabolic', 'quartic' or 'full')
verbosePlot = True # USER DEFINED
#
print('Calibration of the basis')
udrone.CalibrationOfBasisImages(pathFolderBasis, eCritical, calibrationModel, verbosePlot)
print('Calibration of the basis forcing a unique intrinsic parameters')
udrone.CalibrationOfBasisImagesConstantIntrinsic(pathFolderBasis, calibrationModel, verbosePlot)
#
#''' --------------------------------------------------------------------------
# (Auto)Calibration of the frames
#''' --------------------------------------------------------------------------
#
pathFolderBasis = pathFolderMain + os.sep + 'basis' # USER DEFINED
pathFolderFrames = pathFolderMain + os.sep + 'frames' # USER DEFINED
verbosePlot = True # USER DEFINED
#
print('Autocalibration of the frames')
udrone.AutoCalibrationOfFramesViaGCPs(pathFolderBasis, pathFolderFrames, verbosePlot)
#
#''' --------------------------------------------------------------------------
# Plot planviews
#''' --------------------------------------------------------------------------
#
#pathFolderFrames = pathFolderMain + os.sep + 'frames' # USER DEFINED
pathFolderPlanviews = pathFolderMain + os.sep + 'planviews' # USER DEFINED
z0, ppm = 3.2, 2.0 # USER DEFINED
verbosePlot = True # USER DEFINED
#
print('Generation of planviews')
udrone.PlanviewsFromImages(pathFolderFrames, pathFolderPlanviews, z0, ppm, verbosePlot)
#
#''' --------------------------------------------------------------------------
# Plot mean (timex) and sigma images of planviews
#''' --------------------------------------------------------------------------
#
#pathFolderPlanviews = pathFolderMain + os.sep + 'planviews' # USER DEFINED
#
print('Generation of mean and sigma images for the planviews')
udrone.TimexAngSigma(pathFolderPlanviews)
#
#''' --------------------------------------------------------------------------
# Plot timestacks
#''' --------------------------------------------------------------------------
#
#pathFolderFrames = pathFolderMain + os.sep + 'frames' # USER DEFINED
pathFolderTimestack = pathFolderMain + os.sep + 'timestack' # USER DEFINED
ppm = 10. # USER DEFINED
includeNotCalibrated = True # USER DEFINED
verbosePlot = True # USER DEFINED
#
print('Generation of timestack')
udrone.TimestackFromImages(pathFolderFrames, pathFolderTimestack, ppm, includeNotCalibrated, verbosePlot)
#
#''' --------------------------------------------------------------------------
# check basis images
#''' --------------------------------------------------------------------------
#
pathFolderBasisCheck = pathFolderMain + os.sep + 'basis_check' # USER DEFINED
#eCritical, calibrationModel = 5., 'parabolic' # USER DEFINED (eCritical is in pixels, calibrationModel = 'parabolic', 'quartic' or 'full')
#
print('Checking of the basis')
udrone.CheckGCPs(pathFolderBasisCheck, eCritical, calibrationModel)
#
