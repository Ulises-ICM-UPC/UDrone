# UDrone

`UDrone` is an open source software written in Python for automatic image calibration of drone video images from a set of images that are manually calibrated.

### Description
The calibration algorithm assumes that the intrinsic parameters of the camera remain unchanged while the extrinsic parameters (position and orientation) vary. The result of the process is the common intrinsic camera parameters for all images and the extrinsic parameters for each individual images extracted from a video. In addition, for each image a planview can be generated and, for the video, planviews for mean (_timex_) and sigma (_variance_) of the video images and a timestack can be generated as well. The development of this software is suitable for processing videos obtained from moving cameras such as those acquired from drones. Details on the algorithm and methodology are described in
> *Simarro, G.; Calvete, D.; Plomaritis, T.A.; Moreno-Noguer, F.; Giannoukakou-Leontsini, I.; Montes, J.; Durán, R. The Influence of Camera Calibration on Nearshore Bathymetry Estimation from UAV Videos. Remote Sens. 2021, 13, 150. https://doi.org/10.3390/rs13010150*

The automatic calibration process consists of the following steps:

 1. [Video frame extraction](#video-extraction)
 2. [Intrinsic camera calibration](#intrinsic-calibration)
 3. [Automatic frame calibration](#automatic-calibration)
 
Further `UDrone` allows to generate planviews for the calibrated images, mean and sigma images for the planviews and a timestack of the video:

 4. [Planview generation](#planviews)
 5. [Mean as sigma generation](#mean-and-sigma)
 6. [Timestack generation](#timestack)
 
A code to verify the quality of the GCPs used in the manual calibration of the basis images is also provided:

 7. [Check GCP for basis calibration](#gcp-check)

### Requirements and project structure
To run the software it is necessary to have Python (3.8) and install the following dependencies:
- cv2 (4.2.0)
- numpy (1.19.5)
- scipy (1.3.3)

In parenthesis we indicate the version with which the software has been tested. It is possible that it works with older versions. 

The structure of the project is the following:
* `example.py`
* `example_notebook.py`
* **`udrone`**
  * `udrone.py`
  * `ulises_udrone.py`
* **`example`**
  * `videoFile.mp4` (or .avi)
  * **`basis`**
    * `videoFile_000000000000.png`
    * `videoFile_000000000000cal0.txt`
    * `videoFile_000000000000cal.txt`
    * `videoFile_000000000000cdg.txt`
    * `videoFile_000000000000cdh.txt`
    * . . .
  * **`basis_check`**
    * `videoFile_000000000000.png`
    * `videoFile_000000000000cdg.txt`
    * . . .
  * **`frames`**
    * `videoFile_000000000000.png`
    * `videoFile_000000000000cal.txt`
    * . . .
  * **`planviews`**
    * `crxyz_planviews.txt`
    * `xy_planview.txt`
    * `videoFile__000000000000plw.png`
    * `mean.png`
    * `sigma.png`
    * . . .
  * **`timestack`**
    * `cxyz_timestack.txt`
    * `rt_timestack.txt`
    * `xyz_timestack.txt`
    * `videoFile__000000000000plw.png`
    * `mean.png`
    * `timestack.png`
  * **`TMP`**
    * `videoFile_000000000000cal0_check.png`
    * `videoFile_000000000000cal_check.png`
    * `videoFile_000000000000_checkplw.png`
    * `videoFile_000000000000plw_check.png`
    * `videoFile_000000000000_checktimestack.png`
    * . . .


The local modules of `UDrone` are located in the **`udrone`** folder.

To run a demo with the video in folder **`example`** and a basis of frames in **`basis`** using a Jupyter Notebook we provide the file `example_notebook.ipynb`. For experienced users, the `example.py` file can be run in a terminal. `UDrone` handles `MP4` (recommended) and `AVI` image formats.

## Video extraction

A selection of frames are extracted from the video to be analysed at a certain frame-rate.

### Run video extraction
Import modules:


```python
import sys
import os
sys.path.insert(0, 'udrone')
import udrone as udrone
```

Set the main path, where video is located, and the path of the folder where the frames will be placed:


```python
pathFolderMain = 'example'
pathFolderVideo = pathFolderMain
pathFolderFrames = pathFolderMain + os.sep + 'frames'
```

Set the extraction rate of the frames:

|  | Parameter | Suggested value | Units |
|:--|:--:|:--:|:--:|
| Extraction framerate | `FPS` | _2.0_ | _1/s_ |
Set FPS=0 to extract all frames from the video.


```python
FPS = 2.00
```

Run the code to extract frames from de video:


```python
udrone.Video2Frames(pathFolderVideo, pathFolderFrames, FPS)
```

As a result, images of each extracted frame `<frame>.png` are generated in the **`frames`** folder with the format `<videoFilename>_<milliseconds>.png`.

## Intrinsic calibration
The intrinsic parameters of the camera are determined by a manual calibration of selected frames that will also be used in the automatic calibration of all the extracted frames. To manually calibrate the frames selected for the basis, placed in the folder **`basis`**, it is necessary that each image `<basisFrame>.png` is supplied with a file containing the Ground Control Points (GCP) and, optionally, the Horizon Points (HP). The structure of each of these files is the following:
* `<basisFrame>cdg.txt`: For each GCP one line with (minimum 6)
>`pixel-column`, `pixel-row`, `x-coordinate`, `y-coordinate`, `z-coordinate`
* `<basisFrame>cdh.txt`: For each HP one line with (minimum 3)
>`pixel-column`, `pixel-row`

Quantities must be separated by at least one blank space between them and the last record should not be continued with a newline (return).


### Run basis calibration
Set the folder path where the basis is located:


```python
pathFolderBasis = pathFolderMain + os.sep + 'basis'
```

Set the value of maximum error allowed for the basis calibration:

|  | Parameter | Suggested value | Units |
|:--|:--:|:--:|:--:|
| Critical reprojection pixel error | `eCritical` | _5._ | _pixel_ |



```python
eCritical = 5.
```

Select an intrinsic camera calibration model.

| Camara model | `parabolic` | `quartic`| `full` |
|:--|:--:|:--:|:--:|
| Lens radial distortion | parabolic | parabolic + quartic | parabolic + quartic |
| Lens tangential distortion | no | no | yes |
| Square pixels | yes | yes | no |
| Decentering | no | no | yes |

The `parabolic` model is recommended by default, unless the images are highly distorted.


```python
calibrationModel = 'parabolic'
```

To facilitate the verification that the GCPs have been correctly selected in each image of the basis, images showing the GCPs and HPs (black), the reprojection of GCPs (yellow) and the horizon line (yellow) on the images can be generated. Set parameter `verbosePlot = True`, and to `False` otherwise. Images (`<basisFrame>cal0_check.png`) will be placed on a TMP folder.


```python
verbosePlot = True
```

Run the initial calibration algorithm for each image of the basis:


```python
udrone.CalibrationOfBasisImages(pathFolderBasis, eCritical, calibrationModel, verbosePlot)
```

In case that the reprojection error of a GCP is higher than the error `eCritical` for a certain image `<basisFrame>`, a message will appear suggesting to re-run the calibration of the basis or to modify the values or to delete points in the file `<basisFrame>cdg.txt`. If the calibration error of an image exceeds the error `eCritical` the calibration is given as _failed_. Consider re-run the calibration of the basis or verify the GPCs and HPs.

Then, run the algorithm to obtain the optimal intrinsic parameters of the camera.


```python
udrone.CalibrationOfBasisImagesConstantIntrinsic(pathFolderBasis, calibrationModel, verbosePlot)
```

As a result of the calibration, the calibration file `<basisFrame>cal.txt` is generated in the **`basis`** directory for each of the frames. This file contains the following parameters:

| Magnitudes | Variables | Units |
|:--|:--:|:--:|
| Camera position coordinates | `xc`, `yc`, `zc` | _m_ |
| Camera orientation angles | `ph`, `sg`, `ta` | _rad_ |
| Lens radial distortion (parabolic, quartic) | `k1a`, `k2a` | _-_ |
| Lens tangential distortion (parabolic, quartic) | `p1a`, `p2a` | _-_ |
| Pixel size | `sc`, `sr` | _-_ |
| Decentering | `oc`, `or` | _pixel_ |
| Image size | `nc`, `nr` | _pixel_ |
| Calibration error | `errorT`| _pixel_ |

The different calibration files `<basisFrame>cal.txt` differ only in extrinsic paramaters (`xc`, `yc`, `zc`, `ph`, `sg`, `ta`) and the calibration error (`errorT`). A `<basisFrame>cal0.txt` file with the manual calibration parameters for each frame of the basis will also have been generated.

## Automatic calibration

In this step, each of the frames `<frame>.png` in the folder **`frames`** will be automatically calibrated. To facilitate the verification that the GCPs have been correctly identified in each frame, images showing the reprojection of the GCPs can be generated. Set parameter `verbosePlot = True`, and to `False` otherwise. Images (`<frame>cal_check.png`) will be placed on a **`TMP`** folder.


```python
verbosePlot = True
```

Run the algorithm to calibrate frames automatically:


```python
udrone.AutoCalibrationOfFramesViaGCPs(pathFolderBasis, pathFolderFrames, verbosePlot)
```

For each of the frames `<frame>.png` in folder **`frames`**, a calibration file `<frame>cal.txt` with the same characteristics as the one described above will be obtained. In case autocalibration process fails, it is reported that the calibration of the frame `<frame>.png` has _failed_. Increasing the basis frames can improve the calibration process. 

## Planviews

Once the frames have been calibrated, planviews can be generated. The region of the planview is the one delimited by the minimum area rectangle containing the points of the plane specified in the file `xy_planview.txt` in the folder **`planviews`**. The planview image will be oriented so that the nearest corner to the point of the first of the file  `xy_planview.txt` will be placed in the upper left corner of the image. The structure of this file is the following:
* `xy_planview.txt`: For each points one line with 
> `x-coordinate`, `y-coordinate`

A minimum number of three not aligned points is required. These points are to be given in the same coordinate system as the GCPs.

Set the folder path where the file `xy_planview.txt` is located and the value of `z0`.


```python
pathFolderPlanviews = pathFolderMain + os.sep + 'planviews'
z0 = 3.2
```

The resolution of the planviews is fixed by the pixels-per-meter established in the parameter `ppm`. To help verifying that the points for setting the planview are correctly placed, it is possible to show such points on the frames and on the planviews. Set the parameter `verbosePlot = True`, and to `False` otherwise. The images (`<frame>_checkplw.png` and `<frame>plw_check.png`) will be placed in a TMP folder.


```python
ppm = 1.0
verbosePlot = True
```

Run the algorithm to generate the planviews:


```python
udrone.PlanviewsFromImages(pathFolderFrames, pathFolderPlanviews, z0, ppm, verbosePlot)
```

As a result, for each of the calibrated frames `<frame>.png` in folder **`frames`**, a planview `<frame>plw.png` will be placed in the folder **`planviews`**. Note that objects outside the plane at height `z0` will show apparent displacements due to real camera movement. In the same folder, the file `crxyz_planviews.txt` will be located, containing the coordinates of the corner of the planviews images:
* `crxyz_planviews.txt`: For each corner one line with 
>`pixel-column`, `pixel-row`, `x-coordinate`, `y-coordinate`, `z-coordinate`

## Mean and sigma

From the planview of each calibrated frame, time exposure (_timex_) and sigma images can be generated by computing the mean value (`mean.png`) and the standard deviation (`sigma.png`) of all images in the folder **`planviews`**, respectively.


```python
udrone.TimexAngSigma(pathFolderPlanviews)
```

## Timestack

In order to obtain time series of the pixel values of the frames along a path in the space, a file with the coordinates of points along the path must be provided. The valies are obtained along the straight segments bounded by consecutive points in the file. The structure of this file, located in the folder **`timestack`**, is the following:
* `xyz_timestack.txt`: For each point one line with
>`x-coordinate`, `y-coordinate`, `z-coordinate`

A minimum number of two points is required. These points are to be given in the same coordinate system as the GCPs. Set the folder path of the file `xyz_timestack.txt`.


```python
pathFolderTimestack = pathFolderMain + os.sep + 'timestack'
```

The resolution of the timestack is fixed by the pixels-per-meter established in the parameter `ppm`. In case a frame has no calibration, set the parameter `includeNotCalibrated = True` to include a black line for that image or `False` to be completely ignored in the timestack generation. To help verifying that the points for setting the timestack are correctly placed, it is possible to show such points on the frames (`<frame>_checktimestack.png` in **`TMP`** folder). Set the parameter `verbosePlot = True`, and to `False` otherwise. 


```python
ppm = 10.
includeNotCalibrated = False
verbosePlot = True
```

Run the algorithm to generate the timestack:


```python
udrone.TimestackFromImages(pathFolderFrames, pathFolderTimestack, ppm, includeNotCalibrated, verbosePlot)
```

As a result, a timestack `timestack.png` will be placed in the folder  **`timestack`**. In the same folder, a file `cxyz_timestack.txt` containing the spatial coordinates of each column and a file `rt_timestack.txt` containing the file (time) of each row of the timestack will be located. The structure of these files are the following:
* `cxyz_timestack.txt`: 
>`pixel-column`, `x-coordinate`, `y-coordinate`, `z-coordinate`
* `rt_timestack.txt`: 
>`pixel-row`, `<frame>.png`

## GCP check

To verify the quality of the GCPs used in the manual calibration of the basis frames, a RANSAC (RANdom SAmple Consensus) is performed. Points of the files `<basisImage>cdg.txt` located at the **`basis_check`** folder will be tested. The calibration of the points is done with a `calibrationModel` and requires a minimum error `eCritical`. Set the folder and run the RANSAC algorithm:


```python
pathFolderBasisCheck = pathFolderMain + os.sep + 'basis_check'
udrone.CheckGCPs(pathFolderBasisCheck, eCritical, calibrationModel)
```

For each file `<basisFrame>cdg.txt`, the GPCs that should be revised or excluded will be reported.

## Contact us

Are you experiencing problems? Do you want to give us a comment? Do you need to get in touch with us? Please contact us!

To do so, we ask you to use the [Issues section](https://github.com/Ulises-ICM-UPC/UDrone/issues) instead of emailing us.

## Contributions

Contributions to this project are welcome. To do a clean pull request, please follow these [guidelines](https://github.com/MarcDiethelm/contributing/blob/master/README.md)

## License

UCalib is released under a [AGPL-3.0 license](https://github.com/Ulises-ICM-UPC/UDrone/blob/master/LICENSE). If you use UDrone in an academic work, please cite:

    @Article{rs13010150,
      AUTHOR = {Simarro, Gonzalo and Calvete, Daniel and Plomaritis, Theocharis A. and Moreno-Noguer, Francesc and Giannoukakou-Leontsini, Ifigeneia and Montes, Juan and Durán, Ruth},
      TITLE = {The Influence of Camera Calibration on Nearshore Bathymetry Estimation from UAV Videos},
      JOURNAL = {Remote Sensing},
      VOLUME = {13},
      YEAR = {2021},
      NUMBER = {1},
      ARTICLE-NUMBER = {150},
      URL = {https://www.mdpi.com/2072-4292/13/1/150},
      ISSN = {2072-4292},
      DOI = {10.3390/rs13010150}
      }

    @Online{ulisesdrone, 
      author = {Simarro, Gonzalo and Calvete, Daniel},
      title = {UDrone},
      year = 2021,
      url = {https://github.com/Ulises-ICM-UPC/UDrone}
      }

