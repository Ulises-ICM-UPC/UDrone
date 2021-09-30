# UDrone

`UDrone` is an open source software written in Python for automatic calibration of drone video images from a set of manually calibrated frames.

### Description
The calibration algorithm assumes that the intrinsic camera parameters remain unchanged while the extrinsic parameters (position and orientation) vary. The result of the process is the common intrinsic camera parameters for all images and the extrinsic parameters for each individual images extracted from a video. In addition, planviews can be generated for each image. The development of this software is suitable for processing videos obtained from moving cameras such as those acquired from drones. Details on the algorithm and methodology are described in
> *Simarro, G.; Calvete, D.; Plomaritis, T.A.; Moreno-Noguer, F.; Giannoukakou-Leontsini, I.; Montes, J.; Durán, R. The Influence of Camera Calibration on Nearshore Bathymetry Estimation from UAV Videos. Remote Sens. 2021, 13, 150. https://doi.org/10.3390/rs13010150*

The automatic calibration process consists of the following steps:

 1. [Video frame extraction](#video-extraction)
 2. [Intrinsic camera calibration](#intrinsic-calibration)
 3. [Automatic frame calibration](#automatic-calibration)
 4. [Planview generation](#planviews)
 
A code to verify the quality of the GCPs used in the manual calibration of the basis images is also provided:

 5. [Check GCP for basis calibration](#gcp-check)

### Requirements and project structure
To run the software it is necessary to have Python (3.8) and install the following dependencies:
- cv2 (4.2.0)
- numpy (1.19.5)
- scipy (1.3.3)

In parenthesis we indicate the version with which the software has been tested. It is possible that it works with older versions. 

The structure of the project is the following:
* `example.py`
* **`udrone`**
  * `udrone.py`
  * `ulises_udrone.py`
* **`example`**
  * `videoFile.mp4` (or .avi)
  * `xy_planview.txt`
  * **`basis`**
    * `videoFile_000000000000cal.txt`
    * `videoFile_000000000000cal0.txt`
    * `videoFile_000000000000cdg.txt`
    * `videoFile_000000000000cdh.txt`
    * `videoFile_000000000000.png`
    * . . .
  * **`basis_check`**
    * `videoFile_000000000000cdg.txt`
    * `videoFile_000000000000.png`
    * . . .
  * **`frames`**
    * `videoFile_000000000000cal.txt`
    * `videoFile_000000000000.png`
    * . . .
  * **`planviews`**
    * `crxy_Planviews.txt`
    * `videoFile__000000000000plv.png`
    * . . .
  * **`TMP`**
    * `videoFile_000000000000_check.png`
    * `videoFile_000000000000_checkplw.png`
    * `videoFile_000000000000plw_check.png`
    * . . .


The local modules of `UDrone` are located in the **`udrone`** folder.

To run a demo with the video in folder **`example`** and a basis of frames in **`basis`** using a Jupyter Notebook we provide the file `example_notebook.ipynb`. For experienced users, the `example.py` file can be run in a terminal. 

## Video extraction

A selection of frames are extracted from the video to be analysed at a certain framerate. Supported video formats are `.mp3` and `.avi`.

### Run video extraction
Import modules:


```python
import sys
sys.path.insert(0, 'udrone')
import udrone
import os
```

Set the folder path where video is located, the filename of the video and the folder path where the frames will be placed:


```python
pathMain = 'example'
pathFolderVideo = pathMain
pathFolderFrames = pathMain + os.sep + 'frames'
```

Set the extraction rate of the frames:

|  | Parameter | Suggested value | Units |
|:--|:--:|:--:|:--:|
| Extraction framerate | `FPS` | _0.5_ | _1/s_ |
Set FPS=0 to extract all frames from the video.


```python
FPS=2.00
```

Run the code to extract frames from de video:


```python
udrone.Video2Frames(pathFolderVideo, pathFolderFrames, FPS)
```

As a result, images of each extracted frame `<frame>.png` are generated in the **`frames`** folder with the format `<videoFilename>_<milliseconds>.png`.

## Intrinsic calibration
The intrinsic parameters of the camera are determined by a manual calibration of selected frames that will also be used in the automatic calibration of all the extracted frames. To manually calibrate the frames selected for the basis, placed in the folder **`basis`**, it is necessary that each image `<basisFrame>.png` is supplied with a file containing the Ground Control Points (GCP) and, optionally, the Horizon Points (HP). The structure of each of these files is the following:
* `<basisFrame>cdg.txt`: For each GCP one line with (minumun 6)
>`pixel-column`, `pixel-row`, `x-coordinate`, `y-coordinate`, `z-coordinate`
* `<basisFrame>cdh.txt`: For each HP one line with (minumun 3)
>`pixel-column`, `pixel-row`

Quantities must be separated by at least one blank space between them and the last record should not be continued with a newline (return).


### Run intrinsic calibration
Set the folder path where files for calibrating the basis are located:


```python
pathFolderBasis = pathMain + os.sep + 'basis'
```

Select an intrinsic camera calibration model.

| Camara model | `parabolic` | `quartic`| `full` |
|:--|:--:|:--:|:--:|
| Lens radial distortion | parabolic | parabolic + quartic | parabolic + quartic |
| Lens tangential distortion | _-_ | _-_ | parabolic + quartic |
| Square pixels | yes | yes | no |
| Decentering | no | no | yes |

The `parabolic` model is recommended by default, unless the images are highly distorted.


```python
calibrationModel = 'parabolic'
```

Run the calibration algorithm of the basis:


```python
udrone.calibrationOfBasisImages(pathFolderBasis, calibrationModel)
```

Then, run the algorithm to obtain the optimal intrinsic parameters of the camera.


```python
udrone.calibrationOfBasisImagesConstantIntrinsic(pathFolderBasis, calibrationModel)
```

As a result of the calibration, the calibration file `<basisFrame>cal.txt` is generated in the **`basis`** directory for each of the frames. This file contains the following parameters:

| Magnitudes | Variables | Units |
|:--|:--:|:--:|
| Camera position coordinates | `xc`, `yc`, `zc` | _m_ |
| Camera orientation angles | `ph`, `sg`, `ta` | _rad_ |
| Lens radial distortion (parabolic, quartic) | `k1a`, `k2a` | _-_ |
| Lens tangential distortion (parabolic, quartic) | `p1a`, `p2a` | _-_ |
| Pixel size | `sc`, `sr` | _-_ |
| Decentering | `oc`, `rr` | _pixel_ |
| Image size | `nc`, `nr` | _pixel_ |
| Calibration error | `errorT`| _pixel_ |

The different calibration files `*cal.txt` differ only in extinsec paramaters (`xc`, `yc`, `zc`, `ph`, `sg`, `ta`) and the calibration error (`errorT`). A `<basisFrame>cal0.txt` file with the manual calibration parameters for each frame of the basis will also have been generated.

## Automatic calibration

In this step, each of the frames `<frame>.png` in the folder **`frames`** will be automatically calibrated. To facilitate the verification that the GCPs have been correctly identified in each frame, images showing the reprojection of the GCPs can be generated. Set parameter `verbosePlot = True`, and to `False` otherwise. Images(`<frame>_check.png`) will be placed on a **`TMP`** folder


```python
verbosePlot = True
```

Run the algorithm to calibrate frames automatically:


```python
udrone.autoCalibrationOfFrames(pathFolderBasis, pathFolderFrames, verbosePlot)
```

For each of the frames `<frame>.png` in folder **`frames`**, a calibration file `<frame>cal.txt` with the same characteristics as the one described above will be obtained. The self-calibration process may fail for particular frames. Increasing the basis frames can improve the calibration process. 

## Planviews

Once the frames have been calibrated, a planview can be generated. The region of the planview is the one delimited by the minimum area rectangle containing the points of the plane specified in the file `xy_planview.txt`. The planview image will be oriented so that the nearest corner to the point of the first of the file  `xy_planview.txt` will be placed in the upper left corner of the image. The structure of this file is the following:
* `xy_planview.txt`: For each points one line with 
> `x-coordinate`, `y-coordinate`

A minimum number of three points is required. These points are to be given in the same coordinate system as the GCPs and on a plane at height `z0` in which the projection is made. Set the value of `z0`.



```python
z0 = 3.2
```

The resolution of the planviews is fixed by the pixels-per-meter established in the parameter `ppm`. To help verify that the points for setting the planview are correctly placed, it is possible to show such points on the frames and on the planviews. Set the parameter `verbosePlot = True`, and to `False` otherwise. The images (`<frame>_checkplw.png` and
`<frame>plw_check.png`) will be placed in a TMP folder.


```python
ppm = 1.0
verbosePlanviews = True
```

Set the folder path where planviews will be located and run the algorithm to generate the planviews:


```python
pathFolderPlanv = pathMain + os.sep + 'planviews'
udrone.planviewsFromSnaps(pathMain, pathFolderFrames, pathFolderPlanv, z0, ppm, verbosePlanviews)
```

As a result, for each of the calibrated frames `<frame>.png` in folder **`frames`**, a planview `<frame>plw.png` will be placed in the folder **`planviews`**. Note that objects outside the plane at height z0 will show apparent displacements due to real camera movement. In the same folder, the file `crxy_Planviews.txt` will be located, containing the coordinates of the corner of the planviews images:
* `crxy_Planviews.txt`: For each corner one line with 
>`pixel-column`, `pixel-row`, `x-coordinate`, `y-coordinate`, `z-coordinate`

## GCP check



To verify the quality of the GCPs used in the manual calibration of the basis frames, a RANSAC (RANdom SAmple Consensus) is performed to the GCPs using the calibration (parabolic) of four points as the model. Points of the files `<basisFrame>cdg.txt` located at the **`basis_check`** folder will be test. Set the folder and run the RANSAC algorithm:


```python
pathBasisCheck=pathMain + os.sep + 'basis_check'
udrone.checkGCPs(pathBasisCheck)
```

For each file `<basisFrame>cdg.txt`, the GPCs that should be revised or excluded will be reported.

## Contact us

Are you experiencing problems? Do you want to give us a comment? Do you need to get in touch with us? Please contact us!

To do so, we ask you to use the [Issues section](https://github.com/Ulises-ICM-UPC/UDrone/issues) instead of emailing us.

## Contributions

Contributions to this project are welcome. To do a clean pull request, please follow these [guidelines](https://github.com/MarcDiethelm/contributing/blob/master/README.md)

## License

UCalib is released under a [GPLv3 license](https://github.com/Ulises-ICM-UPC/UCalib/blob/main/LICENSE). If you use UDrone in an academic work, please cite:

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
