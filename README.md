# 3D_surveillance
This work implements a prototype of 3D person surveillance, which fuses all camera views into one consistent 3D scene. In the offline stage, 3D reconstruction of the experiment environment is performed and surveillance cameras are calibrated. In the online stage, the positions of persons are updated using object tracking algorithm, and new persons are added to the tracking list at certain frequency, by comparing the detection and tracking results. Before they are tracked, their identities (person id) are queried using person re-id algorithm. Finally all tracked persons are back-projected into the reconstructed 3D environment model as a textured rectangle, and the 3D scene containing both environment and persons are rendered in a separated thread.

## Person detection
We choose SSD as our person detector.
### Prerequisites
* [caffe & SSD](https://github.com/weiliu89/caffe/tree/ssd)
### Usage
To obtain only person detections with confidence higher than certain threshold, we should modify main function in the example script `caffe/examples/ssd/ssd_detect.py` appropriately:
1. Set confidence threshold `conf_thresh` when calling `detect` method (we set it to 0.3 in our settings).

2. In the iteration section, compare `item[-1]` with string `person` to filter out detections of other categories.

## Person tracking with re-identification
Our tracking algorithm is an improvement to original [multi-scale KCF](https://github.com/joaofaro/KCFcpp), while person re-id algorithm is based on [trinet](https://github.com/VisualComputingInstitute/triplet-reid) features. In order to integrate tracking and re-id, we decide to use Python in the top level, so we need a Python wrapper for the tracking algorithm which is originally implemented in C++. It is achieved using Cython.
### Prerequisites
* [Python 3](https://www.python.org/downloads/)

* [Cython](http://docs.cython.org/en/latest/src/quickstart/install.html)

* OpenCV [for C++](https://opencv.org/releases.html) & [for Python3](https://stackoverflow.com/questions/46610689/how-to-import-cv2-in-python3)

* Python packages: numpy, matplotlib, [tensorflow](https://www.tensorflow.org/install/?hl=zh-cn)
### Usage
1. Compile python extensions for KCF. The directory `tracking/KCF_source` contains C++ source code of KCF, while directory `tracking/KCF_wrapper` contains necessary files to compile python extensions. In the latter directory, `python/KCF.pyx` defines a python wrapper class for original C++ class `KCFTracker`, and `setup.py` sets required paths, such as OpenCV library path. If interface of `KCFTracker` is modified, the definition of its wrapper class should be modified accordingly. And the paths in `python/KCF.pyx` and `setup.py` should be set according to your environment. After all necessary changes are made, just compile the extensions by `python setup.py build_ext --inplace` (If you don't want to compile it yourself, you can simply copy the compiled extensions in `tracking/KCF_wrapper/build/lib.win-amd64-3.6/` to python library directory).

2. Download the [network weights of trinet](https://github.com/VisualComputingInstitute/triplet-reid/releases/tag/250eb1) and unzip it. Next create a new directory `checkpoint` inside `tracking/trinet`, and put the unzipped files in it.

3. Set all required paths in script `tracking/run.py` and just type `python run.py` to execute the program.

## Scene rendering
The rendering of 3D scene is implemented using a C++ 3D graphics library called OSG (OpenSceneGraph).
### Prerequisites
* [OSG](http://www.openscenegraph.org/index.php/documentation/platform-specifics/windows/37-visual-studio)

* OpenCV (assume it is already installed in the last section)

* [Eigen](http://eigen.tuxfamily.org/dox/GettingStarted.html)
### Usage
It is easy to compile and run it in visual studio, just by adding those sources in `rendering` directory into an empty VS project. Note the paths to those required third-party libraries should be set in VS, and paths to dynamic link libraries of OSG (including thrid-party dlls) should be added to SYSTEM PATH.
